import Foundation

public struct QwenExternalDraftMetadata: Sendable, Equatable {
    public let architectureFamily: String
    public let vocabularySize: Int
    public let specialTokenIDs: [String: Int]

    public init(architectureFamily: String,
                vocabularySize: Int,
                specialTokenIDs: [String: Int]) {
        self.architectureFamily = architectureFamily
        self.vocabularySize = vocabularySize
        self.specialTokenIDs = specialTokenIDs
    }
}

public enum QwenExternalDraftCompatibilityError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case unsupportedTarget(ModelFamily)
    case vocabularyMismatch(expected: Int, actual: Int)
    case specialTokenMismatch(expected: [String: Int], actual: [String: Int])
    case invalidBlockSize(Int)

    public var description: String {
        switch self {
        case .unsupportedTarget(let modelFamily):
            return "external drafting is unsupported for target family \(modelFamily)"
        case .vocabularyMismatch(let expected, let actual):
            return "draft vocabulary \(actual) does not match target vocabulary \(expected)"
        case .specialTokenMismatch(let expected, let actual):
            return "draft special-token IDs \(actual) do not match target IDs \(expected)"
        case .invalidBlockSize(let blockSize):
            return "external draft block size must be between 1 and 8, got \(blockSize)"
        }
    }
}

public enum QwenExternalDraftCompatibility {
    public static let maximumBlockSize = 8

    public static func validate(
        draft: QwenExternalDraftMetadata,
        targetFamily: ModelFamily,
        targetVocabularySize: Int,
        targetSpecialTokenIDs: [String: Int],
        blockSize: Int) throws {
        guard targetFamily == .qwen36MoeText
                || targetFamily == .qwen38FlashNextText else {
            throw QwenExternalDraftCompatibilityError.unsupportedTarget(targetFamily)
        }
        guard blockSize > 0 && blockSize <= maximumBlockSize else {
            throw QwenExternalDraftCompatibilityError.invalidBlockSize(blockSize)
        }
        guard draft.vocabularySize == targetVocabularySize else {
            throw QwenExternalDraftCompatibilityError.vocabularyMismatch(
                expected: targetVocabularySize,
                actual: draft.vocabularySize)
        }
        guard draft.specialTokenIDs == targetSpecialTokenIDs else {
            throw QwenExternalDraftCompatibilityError.specialTokenMismatch(
                expected: targetSpecialTokenIDs,
                actual: draft.specialTokenIDs)
        }
    }
}
