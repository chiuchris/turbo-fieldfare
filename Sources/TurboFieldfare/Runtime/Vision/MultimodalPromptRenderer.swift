import Foundation

public enum MultimodalContentPart: Sendable, Equatable {
    case text(String)
    case image(id: UUID)
}

/// A continuation turn's content, before image token counts are known.
public enum MultimodalContinuationPart: Sendable, Equatable {
    case text(String)
    case image
}

/// Tokens for a continuation turn. Ranges are relative to this turn, which is
/// also the suffix handed to the runner, so they need no rebasing.
public struct MultimodalContinuationTokens: Sendable, Equatable {
    public let effectiveTokenIDs: [Int32]
    public let embeddingTokenIDs: [Int32]
    public let imageTokenRanges: [Range<Int>]
}

public struct MultimodalMessage: Sendable, Equatable {
    public let role: GFTokenizer.Role
    public let content: [MultimodalContentPart]
    public let toolCalls: [GFTokenizer.HistoricalToolCall]
    public let toolCallID: String?
    public let name: String?

    public init(role: GFTokenizer.Role,
                content: [MultimodalContentPart],
                toolCalls: [GFTokenizer.HistoricalToolCall] = [],
                toolCallID: String? = nil,
                name: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }
}

public enum MultimodalPromptRendererError: Error, Equatable {
    case emptyMessages
    case emptyContent
    case reservedImageMarker
    case missingImage(UUID)
    case unexpectedImage(UUID)
    case placeholderMismatch
}

public enum MultimodalPromptRenderer {
    public static let placeholder = "<|image|>"
    public static let imageTokenID = Int32(258_880)
    public static let beginImageTokenID = Int32(255_999)
    public static let endImageTokenID = Int32(258_882)
    public static let qwenImageTokenID = Int32(248_056)
    public static let qwenVisionStartTokenID = Int32(248_053)
    public static let qwenVisionEndTokenID = Int32(248_054)

    struct ImageMarkers {
        let placeholder: String
        let imageTokenID: Int32
        let beginToken: String
        let beginTokenID: Int32
        let endToken: String
        let endTokenID: Int32
        var reservedText: [String] { [placeholder, beginToken, endToken] }
    }

    public static func render(
        messages: [MultimodalMessage],
        featuresByID: [UUID: VisionFeatures],
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition] = []
    ) throws -> MultimodalPrefillInput {
        guard tokenizer.family == .gemma4 else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        return try render(
            messages: messages,
            payloads: featuresByID.mapValues { $0 as any MultimodalVisionFeaturePayload },
            tokenizer: tokenizer,
            tools: tools)
    }

    public static func renderQwen(
        messages: [MultimodalMessage],
        featuresByID: [UUID: QwenVisionFeatures],
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition] = []
    ) throws -> MultimodalPrefillInput {
        guard tokenizer.family == .qwen36 else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        return try render(
            messages: messages,
            payloads: featuresByID.mapValues { $0 as any MultimodalVisionFeaturePayload },
            tokenizer: tokenizer,
            tools: tools)
    }

    private static func render(
        messages: [MultimodalMessage],
        payloads: [UUID: any MultimodalVisionFeaturePayload],
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition]
    ) throws -> MultimodalPrefillInput {
        guard !messages.isEmpty else { throw MultimodalPromptRendererError.emptyMessages }
        let markers = try imageMarkers(for: tokenizer.family) {
            tokenizer.encode($0, addBOS: false)
        }
        var orderedImages: [(UUID, any MultimodalVisionFeaturePayload)] = []
        let tokenizerMessages = try messages.map { message in
            guard !message.content.isEmpty || !message.toolCalls.isEmpty else {
                throw MultimodalPromptRendererError.emptyContent
            }
            var text = ""
            for part in message.content {
                switch part {
                case .text(let value):
                    guard !markers.reservedText.contains(where: value.contains) else {
                        throw MultimodalPromptRendererError.reservedImageMarker
                    }
                    text += value
                case .image(let id):
                    guard let features = payloads[id] else {
                        throw MultimodalPromptRendererError.missingImage(id)
                    }
                    text += markers.placeholder
                    orderedImages.append((id, features))
                }
            }
            return GFTokenizer.Message(
                role: message.role,
                content: text,
                toolCalls: message.toolCalls,
                toolCallID: message.toolCallID,
                name: message.name)
        }
        guard Set(orderedImages.map(\.0)) == Set(payloads.keys) else {
            let used = Set(orderedImages.map(\.0))
            throw MultimodalPromptRendererError.unexpectedImage(
            payloads.keys.first { !used.contains($0) }!)
        }

        let usesToolTemplate = !tools.isEmpty || tokenizerMessages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
        let templateTokens: [Int32]
        if usesToolTemplate {
            templateTokens = try tokenizer.encodeToolChat(
                messages: tokenizerMessages,
                tools: tools)
        } else {
            let rendered = try tokenizer.applyChatTemplate(tokenizerMessages)
            templateTokens = tokenizer.encode(rendered, addBOS: false)
        }
        let placeholders = templateTokens.indices.filter {
            templateTokens[$0] == markers.imageTokenID
        }
        guard placeholders.count == orderedImages.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        var effective: [Int32] = []
        var embedding: [Int32] = []
        var spans: [MultimodalImageSpan] = []
        effective.reserveCapacity(
            templateTokens.count + orderedImages.reduce(0) { $0 + $1.1.tokenCount + 1 })
        embedding.reserveCapacity(effective.capacity)
        var imageIndex = 0
        for token in templateTokens {
            guard token == markers.imageTokenID else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let features = orderedImages[imageIndex].1
            effective.append(markers.beginTokenID)
            embedding.append(markers.beginTokenID)
            let lower = effective.count
            effective.append(contentsOf: repeatElement(
                markers.imageTokenID, count: features.tokenCount))
            embedding.append(contentsOf: repeatElement(Int32(0), count: features.tokenCount))
            spans.append(MultimodalImageSpan(
                tokenRange: lower..<effective.count,
                features: features))
            effective.append(markers.endTokenID)
            embedding.append(markers.endTokenID)
            imageIndex += 1
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageSpans: spans)
    }

    static func imageMarkers(
        for family: GFTokenizer.Family,
        encodeToken: (String) -> [Int32]
    ) throws -> ImageMarkers {
        switch family {
        case .gemma4:
            return ImageMarkers(
                placeholder: placeholder,
                imageTokenID: imageTokenID,
                beginToken: "<|image>",
                beginTokenID: beginImageTokenID,
                endToken: "<image|>",
                endTokenID: endImageTokenID)
        case .qwen36:
            let placeholder = "<|image_pad|>"
            let beginToken = "<|vision_start|>"
            let endToken = "<|vision_end|>"
            let resolvedImageID = encodeToken(placeholder)
            guard resolvedImageID.count == 1,
                  resolvedImageID[0] == qwenImageTokenID else {
                throw GFTokenizerError.specialTokenMismatch(
                    token: placeholder,
                    expected: qwenImageTokenID,
                    resolved: resolvedImageID.first ?? -1)
            }
            let beginIDs = encodeToken(beginToken)
            let endIDs = encodeToken(endToken)
            guard beginIDs.count == 1,
                  beginIDs[0] == qwenVisionStartTokenID else {
                throw GFTokenizerError.specialTokenMismatch(
                    token: beginToken,
                    expected: qwenVisionStartTokenID,
                    resolved: beginIDs.first ?? -1)
            }
            guard endIDs.count == 1, endIDs[0] == qwenVisionEndTokenID else {
                throw GFTokenizerError.specialTokenMismatch(
                    token: endToken,
                    expected: qwenVisionEndTokenID,
                    resolved: endIDs.first ?? -1)
            }
            return ImageMarkers(
                placeholder: placeholder,
                imageTokenID: qwenImageTokenID,
                beginToken: beginToken,
                beginTokenID: qwenVisionStartTokenID,
                endToken: endToken,
                endTokenID: qwenVisionEndTokenID)
        }
    }
}
