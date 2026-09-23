import Foundation
import Metal
import TurboFieldfareFormat

extension Qwen38MTPExecutionGeometry {
    static var switchMoE: Qwen38MTPSwitchMoEGeometry {
        .qwen
    }
}

enum Qwen38MTPHyperConnectionBranch {
    case mixer
    case attention
    case mlp

    var normRole: Qwen38MTPRole {
        switch self {
        case .mixer:
            return .hyperConnectionMixerNorm
        case .attention:
            return .attentionHyperConnectionNorm
        case .mlp:
            return .mlpHyperConnectionNorm
        }
    }

    var inputMixDownRole: Qwen38MTPRole {
        switch self {
        case .mixer:
            return .hyperConnectionMixerInputMixDown
        case .attention:
            return .attentionHyperConnectionInputMixDown
        case .mlp:
            return .mlpHyperConnectionInputMixDown
        }
    }

    var inputMixUpRole: Qwen38MTPRole {
        switch self {
        case .mixer:
            return .hyperConnectionMixerInputMixUp
        case .attention:
            return .attentionHyperConnectionInputMixUp
        case .mlp:
            return .mlpHyperConnectionInputMixUp
        }
    }

    var blockInjectRole: Qwen38MTPRole? {
        switch self {
        case .mixer:
            return nil
        case .attention:
            return .attentionHyperConnectionBlockInject
        case .mlp:
            return .mlpHyperConnectionBlockInject
        }
    }
}

struct Qwen38MTPHyperConnectionWeights {
    let weights: Qwen38HyperConnectionWeights
    let geometry: Qwen38HyperConnectionGeometry

    init(mtp: Qwen38MTPWeights,
         branch: Qwen38MTPHyperConnectionBranch) throws {
        let mtpGeometry = Qwen38MTPExecutionGeometry.qwen
        let geometry = Qwen38HyperConnectionGeometry(
            streamCount: UInt32(mtpGeometry.streamCount),
            hiddenSize: UInt32(mtpGeometry.hiddenSize),
            lowRankSize: UInt32(mtpGeometry.lowRankSize))
        self.geometry = geometry
        self.weights = try Qwen38HyperConnectionWeights(
            norm: mtp.tensor(role: branch.normRole),
            inputMixDown: mtp.tensor(role: branch.inputMixDownRole),
            inputMixUp: mtp.tensor(role: branch.inputMixUpRole),
            blockInject: try branch.blockInjectRole.map {
                try mtp.tensor(role: $0)
            },
            geometry: geometry)
    }
}

final class Qwen38MTPHyperConnectionExecutor {
    let geometry: Qwen38HyperConnectionGeometry
    private let hyperConnection: Qwen38HyperConnection

    init(context: MetalContext) throws {
        let mtpGeometry = Qwen38MTPExecutionGeometry.qwen
        let geometry = Qwen38HyperConnectionGeometry(
            streamCount: UInt32(mtpGeometry.streamCount),
            hiddenSize: UInt32(mtpGeometry.hiddenSize),
            lowRankSize: UInt32(mtpGeometry.lowRankSize))
        self.geometry = geometry
        self.hyperConnection = try Qwen38HyperConnection(
            context: context,
            geometry: geometry)
    }

    func encodePrepare(
        commandBuffer: MTLCommandBuffer,
        hyperInput: MTLBuffer,
        weights: Qwen38MTPHyperConnectionWeights,
        scratch: Qwen38HyperConnectionScratch,
        mixedInput: MTLBuffer,
        tokenCount: UInt32,
        epsilon: Float,
        normalizedIsFloat: Bool = false,
        mixedInputIsFloat: Bool? = nil
    ) {
        precondition(tokenCount > 0)
        precondition(weights.geometry == geometry)
        hyperConnection.encodePrepare(
            commandBuffer: commandBuffer,
            hyperInput: hyperInput,
            weights: weights.weights,
            scratch: scratch,
            mixedInput: mixedInput,
            tokenCount: tokenCount,
            epsilon: epsilon,
            normalizedIsFloat: normalizedIsFloat,
            mixedInputIsFloat: mixedInputIsFloat)
    }

    func encodeInject(
        commandBuffer: MTLCommandBuffer,
        hyperInput: MTLBuffer,
        branchOutput: MTLBuffer,
        injectionWeights: MTLBuffer,
        output: MTLBuffer,
        tokenCount: UInt32
    ) {
        hyperConnection.encodeInject(
            commandBuffer: commandBuffer,
            hyperInput: hyperInput,
            branchOutput: branchOutput,
            injectionWeights: injectionWeights,
            output: output,
            tokenCount: tokenCount)
    }
}

struct Qwen38MTPAttentionWeights {
    let query: TensorView
    let key: TensorView
    let value: TensorView
    let output: TensorView
    let queryNorm: TensorView
    let keyNorm: TensorView

    init(mtp: Qwen38MTP) throws {
        self.query = try mtp.tensor(role: .queryProjection)
        self.key = try mtp.tensor(role: .keyProjection)
        self.value = try mtp.tensor(role: .valueProjection)
        self.output = try mtp.tensor(role: .outputProjection)
        self.queryNorm = try mtp.tensor(role: .queryNorm)
        self.keyNorm = try mtp.tensor(role: .keyNorm)
    }
}

struct Qwen38MTPAttentionScratch {
    let qsaProjectedRows: MTLBuffer
    let qsaBlockScores: MTLBuffer
    let qsaSelectionState: MTLBuffer
    let qsaTokenMask: MTLBuffer
    let queryPositions: MTLBuffer
    let visibleTokenCounts: MTLBuffer
    let projection: MTLBuffer
    let query: MTLBuffer
    let queryGate: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let normalizedQuery: MTLBuffer
    let normalizedKey: MTLBuffer
    let attentionOutput: MTLBuffer
    let prefillAttentionOutput: MTLBuffer
    let gatedAttention: MTLBuffer

    init(device: MTLDevice, maxContext: Int) throws {
        guard maxContext > 0 else {
            throw ModelError.archMismatch(
                field: "mtp.attention.maxContext",
                expected: "positive",
                actual: "\(maxContext)")
        }
        func makeBuffer(_ elements: Int,
                        stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }

        let qsa = Qwen38QSAGeometry.qwen
        let full = QwenFullAttentionGeometry(
            queryHeads: Qwen38MTPExecutionGeometry.qwen.fullAttentionQueryHeads,
            keyValueHeads: Qwen38MTPExecutionGeometry.qwen.fullAttentionKeyValueHeads,
            headDimension: Qwen38MTPExecutionGeometry.qwen.fullAttentionHeadDimension,
            rotaryDimension: 64,
            ropeTheta: 10_000_000)
        let qsaBlockCount = max(1, maxContext / Int(qsa.compressRatio))
        let qsaTokenMaskBytes = maxContext * maxContext

        self.qsaProjectedRows = try makeBuffer(Int(qsa.projectionWidth))
        self.qsaBlockScores = try makeBuffer(
            qsaBlockCount, stride: MemoryLayout<Float>.stride)
        self.qsaSelectionState = try makeBuffer(
            Qwen38QSASelector.stateBytesPerQuery, stride: 1)
        self.qsaTokenMask = try makeBuffer(qsaTokenMaskBytes, stride: 1)
        self.queryPositions = try makeBuffer(1, stride: MemoryLayout<UInt32>.stride)
        self.visibleTokenCounts = try makeBuffer(1, stride: MemoryLayout<UInt32>.stride)
        self.projection = try makeBuffer(full.queryWidth * 2)
        self.query = try makeBuffer(full.queryWidth)
        self.queryGate = try makeBuffer(full.queryWidth)
        self.key = try makeBuffer(full.keyValueWidth)
        self.value = try makeBuffer(full.keyValueWidth)
        self.normalizedQuery = try makeBuffer(full.queryWidth)
        self.normalizedKey = try makeBuffer(full.keyValueWidth)
        self.attentionOutput = try makeBuffer(full.queryWidth)
        self.prefillAttentionOutput = try makeBuffer(full.queryWidth)
        self.gatedAttention = try makeBuffer(full.queryWidth)
    }

    var qsaSelection: Qwen38QSASelectionScratch {
        Qwen38QSASelectionScratch(
            state: qsaSelectionState,
            tokenMask: qsaTokenMask)
    }
}

final class Qwen38MTPAttentionExecutor {
    let geometry: QwenFullAttentionGeometry
    private let projection: Qwen38PLEProjection
    private let qsa: Qwen38QSALayerExecutor
    private let attention: QwenFullAttention

    init(context: MetalContext) throws {
        let mtpGeometry = Qwen38MTPExecutionGeometry.qwen
        self.geometry = QwenFullAttentionGeometry(
            queryHeads: mtpGeometry.fullAttentionQueryHeads,
            keyValueHeads: mtpGeometry.fullAttentionKeyValueHeads,
            headDimension: mtpGeometry.fullAttentionHeadDimension,
            rotaryDimension: 64,
            ropeTheta: 10_000_000)
        self.projection = try Qwen38PLEProjection(context: context)
        self.qsa = try Qwen38QSALayerExecutor(context: context)
        self.attention = try QwenFullAttention(
            context: context,
            geometry: geometry)
    }

    static func rotaryPosition(for statePosition: Int) -> Int {
        statePosition
    }

    func encode(commandBuffer: MTLCommandBuffer,
                state: Qwen38MTPState,
                weights: Qwen38MTPAttentionWeights,
                input: MTLBuffer,
                output: MTLBuffer,
                scratch: Qwen38MTPAttentionScratch,
                position: Int,
                epsilon: Float) throws {
        guard position == state.position else {
            throw PrefillError.prefillCursorMismatch(
                "MTP attention position \(position) != state position \(state.position)")
        }
        guard state.qsa.rawKeyCache.count == position,
              state.fullAttention.count == position else {
            throw PrefillError.prefillCursorMismatch(
                "MTP attention caches do not match position \(position)")
        }
        guard position < state.maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "MTP attention position \(position) exceeds maxContext")
        }
        let rotaryPosition = Self.rotaryPosition(for: position)
        let positionPointer = scratch.queryPositions.contents()
            .assumingMemoryBound(to: UInt32.self)
        positionPointer[0] = UInt32(rotaryPosition)
        let visiblePointer = scratch.visibleTokenCounts.contents()
            .assumingMemoryBound(to: UInt32.self)
        visiblePointer[0] = UInt32(position + 1)
        let nextKeyCount = position + 1
        let reuseKeyCount = state.qsaSelectionKeyCount
        let shouldReuseSelection = reuseKeyCount > 0
        let shouldCaptureSelection = !shouldReuseSelection
            && nextKeyCount >= Int(qsa.geometry.tokenBudget)

        qsa.encode(
            commandBuffer: commandBuffer,
            state: state.qsa,
            hiddenStates: input,
            queryPositions: scratch.queryPositions,
            visibleTokenCounts: scratch.visibleTokenCounts,
            scratch: Qwen38QSALayerExecutionScratch(
                projectedRows: scratch.qsaProjectedRows,
                blockScores: scratch.qsaBlockScores,
                selection: Qwen38QSASelectionScratch(
                    state: scratch.qsaSelection.state,
                    tokenMask: scratch.qsaSelection.tokenMask,
                    reusableTokenMask: shouldReuseSelection
                        ? state.qsaSelectionMask : nil,
                    reusableKeyCount: UInt32(reuseKeyCount),
                    captureTokenMask: shouldCaptureSelection
                        ? state.qsaSelectionMask : nil)),
            tokenCount: 1,
            inputWidth: UInt32(Qwen38MTPExecutionGeometry.qwen.hiddenSize),
            epsilon: epsilon)
        if shouldCaptureSelection {
            state.recordQSASelection(keyCount: nextKeyCount)
        }
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.query,
            input: input,
            output: scratch.projection,
            outputWidth: geometry.queryWidth * 2,
            inputWidth: Qwen38MTPExecutionGeometry.qwen.hiddenSize)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.key,
            input: input,
            output: scratch.key,
            outputWidth: geometry.keyValueWidth,
            inputWidth: Qwen38MTPExecutionGeometry.qwen.hiddenSize)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.value,
            input: input,
            output: scratch.value,
            outputWidth: geometry.keyValueWidth,
            inputWidth: Qwen38MTPExecutionGeometry.qwen.hiddenSize)
        attention.encodeSplitQueryGate(
            commandBuffer: commandBuffer,
            projection: scratch.projection,
            query: scratch.query,
            gate: scratch.queryGate)
        attention.encodeQueryKey(
            commandBuffer: commandBuffer,
            query: scratch.query,
            key: scratch.key,
            queryNorm: weights.queryNorm.buffer,
            queryNormOffset: Int(weights.queryNorm.offset),
            keyNorm: weights.keyNorm.buffer,
            keyNormOffset: Int(weights.keyNorm.offset),
            normalizedQuery: scratch.normalizedQuery,
            normalizedKey: scratch.normalizedKey,
            position: UInt32(rotaryPosition),
            epsilon: epsilon,
            centeredWeights: true)
        state.fullAttention.append(
            commandBuffer: commandBuffer,
            key: scratch.normalizedKey,
            value: scratch.value)
        attention.encode(
            commandBuffer: commandBuffer,
            query: scratch.normalizedQuery,
            keyValueCache: state.fullAttention,
            output: scratch.attentionOutput,
            tokenMask: scratch.qsaTokenMask)
        attention.encodeOutputGate(
            commandBuffer: commandBuffer,
            attention: scratch.attentionOutput,
            gate: scratch.queryGate,
            output: scratch.gatedAttention)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.output,
            input: scratch.gatedAttention,
            output: output,
            outputWidth: Qwen38MTPExecutionGeometry.qwen.hiddenSize,
            inputWidth: geometry.queryWidth)
    }

    func encodeBatch(commandBuffer: MTLCommandBuffer,
                     query: MTLBuffer,
                     cache: QwenFullAttentionKVCache,
                     output: MTLBuffer,
                     startPosition: UInt32,
                     tokenCount: UInt32,
                     tokenMask: MTLBuffer? = nil) {
        attention.encodeBatch(
            commandBuffer: commandBuffer,
            query: query,
            cache: cache,
            output: output,
            startPosition: startPosition,
            tokenCount: tokenCount,
            tokenMask: tokenMask)
    }

    private func encodeProjection(commandBuffer: MTLCommandBuffer,
                                  weights: TensorView,
                                  input: MTLBuffer,
                                  output: MTLBuffer,
                                  outputWidth: Int,
                                  inputWidth: Int) {
        let format = try? Qwen38PLEQuantizedProjection(
            view: weights,
            rows: UInt32(outputWidth),
            columns: UInt32(inputWidth),
            field: "MTP attention projection")
        guard let format else {
            preconditionFailure("invalid MTP attention projection metadata")
        }
        projection.encode(
            commandBuffer: commandBuffer,
            projection: format,
            input: input,
            output: output,
            tokenCount: 1,
            outputWidth: UInt32(outputWidth),
            inputWidth: UInt32(inputWidth))
    }
}

struct Qwen38MTPDraftScratch {
    let fusion: Qwen38MTPInputFusionScratch
    let attention: Qwen38MTPAttentionScratch
    let attentionHyper: Qwen38HyperConnectionScratch
    let mlpHyper: Qwen38HyperConnectionScratch
    let finalMixer: Qwen38HyperConnectionScratch
    let fusionOutput: MTLBuffer
    let attentionInput: MTLBuffer
    let attentionOutput: MTLBuffer
    let afterAttention: MTLBuffer
    let mlpInput: MTLBuffer
    let mlpOutput: MTLBuffer
    let finalHyper: MTLBuffer
    let finalInput: MTLBuffer
    let sharedOutput: MTLBuffer
    let sharedGate: MTLBuffer
    let sharedUp: MTLBuffer
    let sharedAct: MTLBuffer

    init(device: MTLDevice, maxContext: Int) throws {
        let hiddenSize = Qwen38MTPExecutionGeometry.qwen.hiddenSize
        let hyperWidth = Qwen38MTPExecutionGeometry.qwen.streamCount * hiddenSize
        let intermediateSize = Qwen38MTPSwitchMoEGeometry.qwen.intermediateSize
        func makeBuffer(_ elements: Int,
                        stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: elements * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }
        self.fusion = try Qwen38MTPInputFusionScratch(device: device)
        self.attention = try Qwen38MTPAttentionScratch(
            device: device, maxContext: maxContext)
        func makeHyperScratch() throws -> Qwen38HyperConnectionScratch {
            Qwen38HyperConnectionScratch(
                normalized: try makeBuffer(
                    hyperWidth, stride: MemoryLayout<Float>.stride),
                lowRank: try makeBuffer(320),
                activatedLowRank: try makeBuffer(320),
                mixLogits: try makeBuffer(hyperWidth),
                injectionLogits: try makeBuffer(4),
                injectionWeights: try makeBuffer(4))
        }
        self.attentionHyper = try makeHyperScratch()
        self.mlpHyper = try makeHyperScratch()
        self.finalMixer = try makeHyperScratch()
        self.fusionOutput = try makeBuffer(hyperWidth)
        self.attentionInput = try makeBuffer(hiddenSize)
        self.attentionOutput = try makeBuffer(hiddenSize)
        self.afterAttention = try makeBuffer(hyperWidth)
        self.mlpInput = try makeBuffer(hiddenSize)
        self.mlpOutput = try makeBuffer(hiddenSize)
        self.finalHyper = try makeBuffer(hyperWidth)
        self.finalInput = try makeBuffer(hiddenSize)
        self.sharedOutput = try makeBuffer(hiddenSize)
        self.sharedGate = try makeBuffer(intermediateSize)
        self.sharedUp = try makeBuffer(intermediateSize)
        self.sharedAct = try makeBuffer(intermediateSize)
    }
}

enum Qwen38MTPDiagnosticMode: Sendable, Equatable {
    case off
    case on

    var isEnabled: Bool {
        self == .on
    }
}

struct Qwen38MTPDraftBlock: Sendable, Equatable {
    static let maxTokenCount = 4

    let tokens: [Int32]
    let startPosition: Int
    let endPosition: Int

    init(tokens: [Int32], startPosition: Int) throws {
        guard !tokens.isEmpty else {
            throw PrefillError.chunkedUnsupported(
                "MTP draft block requires at least one token")
        }
        guard tokens.count <= Self.maxTokenCount else {
            throw PrefillError.chunkedUnsupported(
                "MTP draft block supports at most \(Self.maxTokenCount) tokens")
        }
        guard startPosition >= 0 else {
            throw PrefillError.prefillCursorMismatch(
                "MTP draft block start position must be non-negative")
        }
        let (endPosition, overflow) = startPosition.addingReportingOverflow(tokens.count)
        guard !overflow else {
            throw PrefillError.prefillCursorMismatch(
                "MTP draft block end position exceeds integer capacity")
        }
        self.tokens = tokens
        self.startPosition = startPosition
        self.endPosition = endPosition
    }

    var tokenCount: Int {
        endPosition - startPosition
    }
}

final class Qwen38MTPDraftExecutor {
    private let model: Model
    private let context: MetalContext
    private let inputFusion: Qwen38MTPInputFusion
    private let inputFusionWeights: Qwen38MTPInputFusionWeights
    private let attention: Qwen38MTPAttentionExecutor
    private let attentionWeights: Qwen38MTPAttentionWeights
    private let attentionHyper: Qwen38MTPHyperConnectionExecutor
    private let attentionHyperWeights: Qwen38MTPHyperConnectionWeights
    private let mlpHyper: Qwen38MTPHyperConnectionExecutor
    private let mlpHyperWeights: Qwen38MTPHyperConnectionWeights
    private let finalMixer: Qwen38MTPHyperConnectionExecutor
    private let finalMixerWeights: Qwen38MTPHyperConnectionWeights
    private let switchMoE: Qwen38MTPSwitchMoEExecutor
    private let switchMoEWeights: Qwen38MTPSwitchMoEWeights
    private let head: QwenUntiedLMHead
    private let scratch: Qwen38MTPDraftScratch
    private let weights: Qwen38MTP
    private let diagnosticMode: Qwen38MTPDiagnosticMode

    init(model: Model,
         context: MetalContext,
         maxContext: Int,
            diagnosticMode: Qwen38MTPDiagnosticMode = .off,
            fcOrientation: Qwen38MTPFCOrientation = .normal) throws {
        self.model = model
        self.context = context
        self.weights = try Qwen38MTP(model: model)
        self.inputFusion = try Qwen38MTPInputFusion(
            context: context,
            fcOrientation: fcOrientation)
        self.inputFusionWeights = try Qwen38MTPInputFusionWeights(mtp: weights)
        self.attention = try Qwen38MTPAttentionExecutor(context: context)
        self.attentionWeights = try Qwen38MTPAttentionWeights(mtp: weights)
        self.attentionHyper = try Qwen38MTPHyperConnectionExecutor(context: context)
        self.attentionHyperWeights = try Qwen38MTPHyperConnectionWeights(
            mtp: weights.weights, branch: .attention)
        self.mlpHyper = try Qwen38MTPHyperConnectionExecutor(context: context)
        self.mlpHyperWeights = try Qwen38MTPHyperConnectionWeights(
            mtp: weights.weights, branch: .mlp)
        self.finalMixer = try Qwen38MTPHyperConnectionExecutor(context: context)
        self.finalMixerWeights = try Qwen38MTPHyperConnectionWeights(
            mtp: weights.weights, branch: .mixer)
        self.switchMoE = try Qwen38MTPSwitchMoEExecutor(context: context)
        self.switchMoEWeights = try Qwen38MTPSwitchMoEWeights(mtp: weights.weights)
        self.head = try QwenUntiedLMHead(
            context: context,
            geometry: QwenLMHeadGeometry(
                vocabularySize: model.config.vocabSize,
                hiddenSize: Qwen38MTPExecutionGeometry.qwen.hiddenSize),
            groupSize: Quantization.qwen38GroupSize,
            weightBits: model.lmHead.quantization?.bits ?? 4)
        self.scratch = try Qwen38MTPDraftScratch(
            device: context.device, maxContext: maxContext)
        self.diagnosticMode = diagnosticMode
    }

    func generate(embedding: MTLBuffer,
                  hiddenStreams: MTLBuffer,
                  targetFinalHidden: MTLBuffer? = nil,
                  state: Qwen38MTPState,
                  logits: MTLBuffer,
                  diagnosticMode: Qwen38MTPDiagnosticMode? = nil,
                  diagnosticLabel: String? = nil) throws -> Int32 {
        let diagnosticsEnabled = (diagnosticMode ?? self.diagnosticMode).isEnabled
        FileHandle.standardError.write(
            Data("mtp generate_label=\(diagnosticLabel ?? "nil")\n".utf8))
        let hiddenBytes = Qwen38MTPExecutionGeometry.qwen.hiddenSize
            * MemoryLayout<Float16>.stride
        let hyperBytes = Qwen38MTPExecutionGeometry.qwen.streamCount
            * hiddenBytes
        guard embedding.length >= hiddenBytes,
              hiddenStreams.length >= hyperBytes,
              logits.length >= head.geometry.vocabularySize
                * MemoryLayout<Float16>.stride else {
            throw ModelError.residentBufferWrapFailed
        }
        let first = try makeCommandBuffer()
        inputFusion.encode(
            commandBuffer: first,
            embedding: embedding,
            hidden: hiddenStreams,
            weights: inputFusionWeights,
            scratch: scratch.fusion,
            epsilon: 1e-6,
            hiddenIsFloat: true)
        attentionHyper.encodePrepare(
            commandBuffer: first,
            hyperInput: scratch.fusion.output,
            weights: attentionHyperWeights,
            scratch: scratch.attentionHyper,
            mixedInput: scratch.attentionInput,
            tokenCount: 1,
            epsilon: 1e-6,
            normalizedIsFloat: true,
            mixedInputIsFloat: false)
        let attentionPosition = state.position
        let diagnosticPosition = Qwen38MTPAttentionExecutor.rotaryPosition(
            for: state.position)
        try attention.encode(
            commandBuffer: first,
            state: state,
            weights: attentionWeights,
            input: scratch.attentionInput,
            output: scratch.attentionOutput,
            scratch: scratch.attention,
            position: state.position,
            epsilon: 1e-6)
        if diagnosticsEnabled {
            attention.encodeBatch(
                commandBuffer: first,
                query: scratch.attention.normalizedQuery,
                cache: state.fullAttention,
                output: scratch.attention.prefillAttentionOutput,
                startPosition: UInt32(attentionPosition),
                tokenCount: 1,
                tokenMask: scratch.attention.qsaTokenMask)
        }
        attentionHyper.encodeInject(
            commandBuffer: first,
            hyperInput: scratch.fusion.output,
            branchOutput: scratch.attentionOutput,
            injectionWeights: scratch.attentionHyper.injectionWeights,
            output: scratch.afterAttention,
            tokenCount: 1)
        mlpHyper.encodePrepare(
            commandBuffer: first,
            hyperInput: scratch.afterAttention,
            weights: mlpHyperWeights,
            scratch: scratch.mlpHyper,
            mixedInput: scratch.mlpInput,
            tokenCount: 1,
            epsilon: 1e-6,
            normalizedIsFloat: true,
            mixedInputIsFloat: false)
        switchMoE.encodeRouter(
            commandBuffer: first,
            weights: switchMoEWeights,
            hidden: scratch.mlpInput)
        try commitAndWait(first)
        if diagnosticsEnabled {
            switchMoE.emitRouterDiagnostics(
                weights: switchMoEWeights,
                hidden: scratch.mlpInput)
            var stages: [(String, MTLBuffer, Int)] = [
                ("normalized_embedding", scratch.fusion.normalizedEmbedding, 2_560),
                ("normalized_hidden", scratch.fusion.normalizedHidden, 10_240),
                ("projected_embedding", scratch.fusion.projectedEmbedding, 2_560),
                ("expanded_embedding", scratch.fusion.expandedEmbedding, 10_240),
                ("projected_hidden", scratch.fusion.projectedHidden, 10_240),
                ("fusion", scratch.fusion.output, 10_240),
                ("attention_input", scratch.attentionInput, 2_560),
                ("attention_projection", scratch.attention.projection, 12_288),
                ("attention_query", scratch.attention.query, 6_144),
                ("attention_gate", scratch.attention.queryGate, 6_144),
                ("attention_key", scratch.attention.key, 512),
                ("attention_value", scratch.attention.value, 512),
                ("attention_normalized_query", scratch.attention.normalizedQuery, 6_144),
                ("attention_normalized_key", scratch.attention.normalizedKey, 512),
                ("attention_output", scratch.attention.attentionOutput, 6_144),
                ("attention_gated_output", scratch.attention.gatedAttention, 6_144),
                ("attention_output_projection", scratch.attentionOutput, 2_560),
                ("after_attention", scratch.afterAttention, 10_240),
                ("mlp_input", scratch.mlpInput, 2_560)]
            if let targetFinalHidden {
                stages.insert(("target_post_final_mixer", targetFinalHidden, 2_560), at: 0)
                stages.insert(("target_pre_final_mixer", hiddenStreams, 10_240), at: 0)
            }
            emitDiagnostics(
                stages,
                prefix: diagnosticLabel,
                floatLabels: ["target_pre_final_mixer", "normalized_hidden"])
            emitStreamDiagnostics(
                label: "fusion",
                buffer: scratch.fusion.output)
            if targetFinalHidden != nil {
                emitStreamDiagnostics(
                    label: "target_pre_final_mixer",
                    buffer: hiddenStreams,
                    isFloat: true)
            }
            emitInputFusionDiagnostics(
                weights: inputFusionWeights,
                scratch: scratch.fusion,
                hidden: hiddenStreams)
            emitAttentionNormalizationDiagnostics(
                weights: attentionWeights,
                scratch: scratch.attention,
                position: diagnosticPosition)
            emitAttentionPathDiagnostics(
                split: scratch.attention.attentionOutput,
                prefill: scratch.attention.prefillAttentionOutput)
            emitAttentionReferenceDiagnostics(
                query: scratch.attention.normalizedQuery,
                cache: state.fullAttention,
                tokenMask: scratch.attention.qsaTokenMask,
                actual: scratch.attention.attentionOutput)
            emitAttentionOutputDiagnostics(
                weights: attentionWeights,
                scratch: scratch.attention,
                output: scratch.attentionOutput)
            emitQSASelectionDiagnostics(state: state, scratch: scratch.attention)
        }

        let routes = switchMoE.selectedRoutes()
        if diagnosticsEnabled {
            emitRouteDiagnostics(experts: routes.experts, weights: routes.weights)
        }
        let expertViews = try switchMoEWeights.expertViews(indices: routes.experts)
        let second = try makeCommandBuffer()
        try switchMoE.encodeMLP(
            commandBuffer: second,
            weights: switchMoEWeights,
            input: scratch.mlpInput,
            output: scratch.mlpOutput,
            sharedOutput: scratch.sharedOutput,
            scratchGate: scratch.sharedGate,
            scratchUp: scratch.sharedUp,
            scratchAct: scratch.sharedAct,
            experts: expertViews)
        mlpHyper.encodeInject(
            commandBuffer: second,
            hyperInput: scratch.afterAttention,
            branchOutput: scratch.mlpOutput,
            injectionWeights: scratch.mlpHyper.injectionWeights,
            output: scratch.finalHyper,
            tokenCount: 1)
        finalMixer.encodePrepare(
            commandBuffer: second,
            hyperInput: scratch.finalHyper,
            weights: finalMixerWeights,
            scratch: scratch.finalMixer,
            mixedInput: scratch.finalInput,
            tokenCount: 1,
            epsilon: 1e-6,
            normalizedIsFloat: true,
            mixedInputIsFloat: false)
        let lmHead = model.lmHead
        head.encode(
            commandBuffer: second,
            weights: lmHead.buffer,
            weightsOffset: Int(lmHead.offset),
            scales: lmHead.buffer,
            scalesOffset: Int(lmHead.scaleOffset),
            biases: lmHead.buffer,
            biasesOffset: Int(lmHead.biasOffset),
            hidden: scratch.finalInput,
            logits: logits)
        try commitAndWait(second)
        if diagnosticsEnabled,
           let targetFinalHidden,
           targetFinalHidden.length >= hiddenBytes {
            let actualValues = scratch.finalInput.contents()
                .assumingMemoryBound(to: Float16.self)
            let targetValues = targetFinalHidden.contents()
                .assumingMemoryBound(to: Float16.self)
            var maximumError: Float = 0
            var sumSquares = 0.0
            var dot = 0.0
            var actualSquares = 0.0
            var targetSquares = 0.0
            for index in 0..<Qwen38MTPExecutionGeometry.qwen.hiddenSize {
                let actual = Float(actualValues[index])
                let target = Float(targetValues[index])
                let error = actual - target
                maximumError = max(maximumError, abs(error))
                sumSquares += Double(error) * Double(error)
                dot += Double(actual) * Double(target)
                actualSquares += Double(actual) * Double(actual)
                targetSquares += Double(target) * Double(target)
            }
            let cosineDenominator = sqrt(actualSquares * targetSquares)
            let cosine = cosineDenominator > 0 ? dot / cosineDenominator : 0
            let line = String(
                format: "mtp final_input_reference n=%d max_abs=%+.6e rms_error=%.6e cosine=%.6e\\n",
                Qwen38MTPExecutionGeometry.qwen.hiddenSize,
                maximumError,
                sqrt(sumSquares / Double(Qwen38MTPExecutionGeometry.qwen.hiddenSize)),
                cosine)
            FileHandle.standardError.write(Data(line.utf8))
        }
        let normalizedFeedback = scratch.finalMixer.normalized
        state.storeFeedback(from: normalizedFeedback)
        state.advance(by: 1)
        if diagnosticsEnabled {
            guard let feedback = state.feedback else {
                preconditionFailure("MTP feedback was not materialized")
            }
            emitLMHeadDiagnostics(
                weights: lmHead,
                hidden: scratch.finalInput,
                logits: logits)
            switchMoE.emitSharedExpertDiagnostics(
                weights: switchMoEWeights,
                input: scratch.mlpInput,
                gate: scratch.sharedGate,
                up: scratch.sharedUp,
                act: scratch.sharedAct,
                output: scratch.sharedOutput)
            switchMoE.emitRoutedExpertDiagnostics(
                experts: expertViews,
                input: scratch.mlpInput,
                sharedOutput: scratch.sharedOutput,
                output: scratch.mlpOutput)
            emitAttentionHyperConnectionDiagnostics(
                branch: "attention",
                hyperInput: scratch.fusion.output,
                scratch: scratch.attentionHyper,
                weights: attentionHyperWeights,
                mixedInput: scratch.attentionInput,
                branchOutput: scratch.attentionOutput,
                injectedOutput: scratch.afterAttention)
            emitAttentionHyperConnectionDiagnostics(
                branch: "mlp",
                hyperInput: scratch.afterAttention,
                scratch: scratch.mlpHyper,
                weights: mlpHyperWeights,
                mixedInput: scratch.mlpInput,
                branchOutput: scratch.mlpOutput,
                injectedOutput: scratch.finalHyper)
            emitAttentionHyperConnectionDiagnostics(
                branch: "final",
                hyperInput: scratch.finalHyper,
                scratch: scratch.finalMixer,
                weights: finalMixerWeights,
                mixedInput: scratch.finalInput)
            emitDiagnostics([
                ("mlp_output", scratch.mlpOutput, 2_560),
                ("shared_output", scratch.sharedOutput, 2_560),
                ("final_hyper", scratch.finalHyper, 10_240),
                ("final_mixer_normalized", scratch.finalMixer.normalized, 10_240),
                ("final_input", scratch.finalInput, 2_560),
                ("feedback", feedback, 10_240),
                ("logits", logits, head.geometry.vocabularySize)],
                prefix: diagnosticLabel,
                floatLabels: ["final_mixer_normalized", "feedback"])
            let finalMixerNormalizedValues = scratch.finalMixer.normalized.contents()
                .assumingMemoryBound(to: Float.self)
            let feedbackValues = feedback.contents()
                .assumingMemoryBound(to: Float.self)
            var mismatchCount = 0
            var maximumError: Float = 0
            var sumSquares = 0.0
            for index in 0..<10_240 {
                let error = Float(feedbackValues[index]) - Float(finalMixerNormalizedValues[index])
                if error != 0 {
                    mismatchCount += 1
                }
                maximumError = max(maximumError, abs(error))
                sumSquares += Double(error) * Double(error)
            }
            let rmsError = sqrt(sumSquares / 10_240)
            let line = String(
                format: "mtp feedback_copy n=%d mismatches=%d max_abs=%+.6e rms_error=%.6e\\n",
                10_240, mismatchCount, maximumError, rmsError)
            FileHandle.standardError.write(Data(line.utf8))
        }
        let values = logits.contents().assumingMemoryBound(to: Float16.self)
        var bestIndex = 0
        var bestValue = values[0]
        for index in 1..<head.geometry.vocabularySize where values[index] > bestValue {
            bestIndex = index
            bestValue = values[index]
        }
        return Int32(bestIndex)
    }

    func generateBlock(
        initialToken: Int32,
        hiddenStreams: MTLBuffer,
        state: Qwen38MTPState,
        logits: MTLBuffer,
        tokenCount: Int,
        embeddingForToken: (Int32) throws -> MTLBuffer
    ) throws -> Qwen38MTPDraftBlock {
        guard tokenCount > 0 && tokenCount <= Qwen38MTPDraftBlock.maxTokenCount else {
            throw PrefillError.chunkedUnsupported(
                "MTP draft block supports between 1 and \(Qwen38MTPDraftBlock.maxTokenCount) tokens")
        }
        guard initialToken >= 0,
              initialToken < Int32(model.config.vocabSize) else {
            throw PrefillError.chunkedUnsupported(
                "MTP draft block input token must be a valid vocabulary ID")
        }
        let checkpoint = state.snapshot()
        let startPosition = state.position
        do {
            var tokens: [Int32] = []
            tokens.reserveCapacity(tokenCount)
            var inputToken = initialToken
            var inputStreams = hiddenStreams
            for _ in 0..<tokenCount {
                let embedding = try embeddingForToken(inputToken)
                let proposedToken = try generate(
                    embedding: embedding,
                    hiddenStreams: inputStreams,
                    state: state,
                    logits: logits)
                tokens.append(proposedToken)
                inputToken = proposedToken
                guard let feedback = state.feedback else {
                    throw ModelError.residentBufferWrapFailed
                }
                inputStreams = feedback
            }
            return try Qwen38MTPDraftBlock(
                tokens: tokens,
                startPosition: startPosition)
        } catch {
            state.restore(checkpoint)
            throw error
        }
    }

    private func emitAttentionOutputDiagnostics(
        weights: Qwen38MTPAttentionWeights,
        scratch: Qwen38MTPAttentionScratch,
        output: MTLBuffer
    ) {
        let geometry = Qwen38MTPExecutionGeometry.qwen
        let queryWidth = geometry.fullAttentionQueryHeads
            * geometry.fullAttentionHeadDimension
        let hiddenSize = geometry.hiddenSize
        let groupSize = 32
        let attentionValues = scratch.attentionOutput.contents()
            .assumingMemoryBound(to: Float16.self)
        let gateValues = scratch.queryGate.contents()
            .assumingMemoryBound(to: Float16.self)
        var expectedGated = [Float16](repeating: 0, count: queryWidth)
        var expectedGateOutput = [Float](repeating: 0, count: queryWidth)
        for index in 0..<queryWidth {
            let gate = Float(gateValues[index])
            let sigmoid = 1 / (1 + exp(-gate))
            let gated = Float16(Float(attentionValues[index]) * sigmoid)
            expectedGated[index] = gated
            expectedGateOutput[index] = Float(gated)
        }
        emitHyperConnectionError(
            branch: "attention",
            relation: "output_gate",
            expected: expectedGateOutput,
            actual: scratch.gatedAttention,
            count: queryWidth)

        var expectedProjection = [Float](repeating: 0, count: hiddenSize)
        expectedGated.withUnsafeBufferPointer { gatedBuffer in
            for row in 0..<hiddenSize {
                let projected = Self.affineQ4Dot(
                    weights.output,
                    row: row,
                    input: gatedBuffer.baseAddress!,
                    count: queryWidth,
                    groupSize: groupSize)
                expectedProjection[row] = Float(Float16(projected))
            }
        }
        emitHyperConnectionError(
            branch: "attention",
            relation: "output_projection",
            expected: expectedProjection,
            actual: output,
            count: hiddenSize)
    }

    private func emitInputFusionDiagnostics(
        weights: Qwen38MTPInputFusionWeights,
        scratch: Qwen38MTPInputFusionScratch,
        hidden: MTLBuffer
    ) {
        let geometry = Qwen38MTPExecutionGeometry.qwen
        let hiddenSize = geometry.hiddenSize
        let streamCount = geometry.streamCount
        let hyperCount = hiddenSize * streamCount
        let normalizedEmbedding = scratch.normalizedEmbedding.contents()
            .assumingMemoryBound(to: Float16.self)
        let hiddenValues = hidden.contents()
            .assumingMemoryBound(to: Float.self)
        let hiddenNormValues = weights.hiddenNorm.buffer.contents()
            .assumingMemoryBound(to: UInt16.self)
        let hiddenNormBase = Int(weights.hiddenNorm.offset)
            / MemoryLayout<UInt16>.stride

        var expectedEmbedding = [Float](repeating: 0, count: hiddenSize)
        for row in 0..<hiddenSize {
            expectedEmbedding[row] = Float(Float16(Self.affineQ4Dot(
                weights.embeddingProjection,
                row: row,
                input: normalizedEmbedding,
                inputOffset: 0,
                count: hiddenSize,
                groupSize: 32)))
        }
        emitHyperConnectionError(
            branch: "fusion",
            relation: "embedding_projection",
            expected: expectedEmbedding,
            actual: scratch.projectedEmbedding,
            count: hiddenSize)

        var expectedNormalizedHidden = [Float](repeating: 0, count: hyperCount)
        var sum: Float = 0
        for index in 0..<hyperCount {
            let value = hiddenValues[index]
            sum = fma(value, value, sum)
        }
        let inverse = 1 / sqrt(sum / Float(hyperCount) + 1e-6)
        for index in 0..<hyperCount {
            let checkpointWeight = Quantization.bf16ToFloat(
                hiddenNormValues[hiddenNormBase + index])
            expectedNormalizedHidden[index] =
                hiddenValues[index] * inverse * (1 + checkpointWeight)
        }
        emitHyperConnectionError(
            branch: "fusion",
            relation: "hidden_zero_centered_norm",
            expected: expectedNormalizedHidden,
            actual: scratch.normalizedHidden,
            count: hyperCount,
            actualIsFloat: true)

        let normalizedHidden = scratch.normalizedHidden.contents()
            .assumingMemoryBound(to: Float.self)
        var expectedHidden = [Float](repeating: 0, count: hyperCount)
        for stream in 0..<streamCount {
            let streamOffset = stream * hiddenSize
            for row in 0..<hiddenSize {
                expectedHidden[streamOffset + row] = Float(Float16(Self.affineQ4Dot(
                    weights.hiddenProjection,
                    row: row,
                    input: normalizedHidden,
                    inputOffset: streamOffset,
                    count: hiddenSize,
                    groupSize: 32)))
            }
        }
        emitHyperConnectionError(
            branch: "fusion",
            relation: "hidden_projection",
            expected: expectedHidden,
            actual: scratch.projectedHidden,
            count: hyperCount)

        let embedding = scratch.projectedEmbedding.contents()
            .assumingMemoryBound(to: Float16.self)
        var expectedFusion = [Float](repeating: 0, count: hyperCount)
        for stream in 0..<streamCount {
            let streamOffset = stream * hiddenSize
            for row in 0..<hiddenSize {
                expectedFusion[streamOffset + row] = Float(Float16(
                    Float(embedding[row]) + expectedHidden[streamOffset + row]))
            }
        }
        emitHyperConnectionError(
            branch: "fusion",
            relation: "residual_add",
            expected: expectedFusion,
            actual: scratch.output,
            count: hyperCount)
    }

    private static func affineQ4Dot(
        _ view: TensorView,
        row: Int,
        input: UnsafePointer<Float16>,
        inputOffset: Int = 0,
        count: Int,
        groupSize: Int
    ) -> Float {
        affineQ4Dot(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset),
            row: row,
            input: input,
            inputOffset: inputOffset,
            count: count,
            groupSize: groupSize)
    }

    private static func affineQ4Dot(
        _ projection: Qwen38PLEQuantizedProjection,
        row: Int,
        input: UnsafePointer<Float16>,
        inputOffset: Int = 0,
        count: Int,
        groupSize: Int
    ) -> Float {
        affineQ4Dot(
            weights: projection.weights,
            weightsOffset: projection.weightsOffset,
            scales: projection.scales,
            scalesOffset: projection.scalesOffset,
            biases: projection.biases,
            biasesOffset: projection.biasesOffset,
            row: row,
            input: input,
            inputOffset: inputOffset,
            count: count,
            groupSize: groupSize)
    }

    private static func affineQ4Dot(
        _ projection: Qwen38PLEQuantizedProjection,
        row: Int,
        input: UnsafePointer<Float>,
        inputOffset: Int = 0,
        count: Int,
        groupSize: Int
    ) -> Float {
        affineQ4Dot(
            weights: projection.weights,
            weightsOffset: projection.weightsOffset,
            scales: projection.scales,
            scalesOffset: projection.scalesOffset,
            biases: projection.biases,
            biasesOffset: projection.biasesOffset,
            row: row,
            input: input,
            inputOffset: inputOffset,
            count: count,
            groupSize: groupSize)
    }

    private static func affineQ4Dot(
        weights weightBuffer: MTLBuffer,
        weightsOffset: Int,
        scales scaleBuffer: MTLBuffer,
        scalesOffset: Int,
        biases biasBuffer: MTLBuffer,
        biasesOffset: Int,
        row: Int,
        input: UnsafePointer<Float16>,
        inputOffset: Int,
        count: Int,
        groupSize: Int
    ) -> Float {
        let weights = weightBuffer.contents()
            .advanced(by: weightsOffset)
            .assumingMemoryBound(to: UInt8.self)
        let scales = scaleBuffer.contents()
            .advanced(by: scalesOffset)
            .assumingMemoryBound(to: UInt16.self)
        let biases = biasBuffer.contents()
            .advanced(by: biasesOffset)
            .assumingMemoryBound(to: UInt16.self)
        let groups = count / groupSize
        let rowBytes = count / 2
        var result: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(scales[row * groups + group])
            let bias = Quantization.bf16ToFloat(biases[row * groups + group])
            let byteBase = row * rowBytes + group * (groupSize / 2)
            for index in 0..<groupSize {
                let byte = weights[byteBase + index / 2]
                let quantized = index.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                result += (Float(quantized) * scale + bias)
                    * Float(input[inputOffset + group * groupSize + index])
            }
        }
        return result
    }

    private static func affineQ4Dot(
        weights weightBuffer: MTLBuffer,
        weightsOffset: Int,
        scales scaleBuffer: MTLBuffer,
        scalesOffset: Int,
        biases biasBuffer: MTLBuffer,
        biasesOffset: Int,
        row: Int,
        input: UnsafePointer<Float>,
        inputOffset: Int,
        count: Int,
        groupSize: Int
    ) -> Float {
        let weights = weightBuffer.contents()
            .advanced(by: weightsOffset)
            .assumingMemoryBound(to: UInt8.self)
        let scales = scaleBuffer.contents()
            .advanced(by: scalesOffset)
            .assumingMemoryBound(to: UInt16.self)
        let biases = biasBuffer.contents()
            .advanced(by: biasesOffset)
            .assumingMemoryBound(to: UInt16.self)
        let groups = count / groupSize
        let rowBytes = count / 2
        var result: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(scales[row * groups + group])
            let bias = Quantization.bf16ToFloat(biases[row * groups + group])
            let byteBase = row * rowBytes + group * (groupSize / 2)
            for index in 0..<groupSize {
                let byte = weights[byteBase + index / 2]
                let quantized = index.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                result += (Float(quantized) * scale + bias)
                    * input[inputOffset + group * groupSize + index]
            }
        }
        return result
    }

    private func emitAttentionNormalizationDiagnostics(
        weights: Qwen38MTPAttentionWeights,
        scratch: Qwen38MTPAttentionScratch,
        position: Int
    ) {
        let geometry = Qwen38MTPExecutionGeometry.qwen
        let headDimension = geometry.fullAttentionHeadDimension
        let queryHeads = geometry.fullAttentionQueryHeads
        let keyValueHeads = geometry.fullAttentionKeyValueHeads
        let rotaryPairs = 64 / 2
        let queryValues = scratch.query.contents().assumingMemoryBound(to: Float16.self)
        let keyValues = scratch.key.contents().assumingMemoryBound(to: Float16.self)
        let normalizedQuery = scratch.normalizedQuery.contents()
            .assumingMemoryBound(to: Float16.self)
        let normalizedKey = scratch.normalizedKey.contents()
            .assumingMemoryBound(to: Float16.self)
        let queryNorm = weights.queryNorm.buffer.contents()
            .assumingMemoryBound(to: UInt16.self)
        let keyNorm = weights.keyNorm.buffer.contents()
            .assumingMemoryBound(to: UInt16.self)
        let queryNormBase = Int(weights.queryNorm.offset) / MemoryLayout<UInt16>.stride
        let keyNormBase = Int(weights.keyNorm.offset) / MemoryLayout<UInt16>.stride

        func expected(
            values: UnsafeMutablePointer<Float16>,
            norm: UnsafeMutablePointer<UInt16>,
            normBase: Int,
            headCount: Int
        ) -> [Float] {
            var result = [Float16](repeating: 0, count: headCount * headDimension)
            for head in 0..<headCount {
                let base = head * headDimension
                var sum: Float = 0
                for feature in 0..<headDimension {
                    let value = Float(values[base + feature])
                    sum = fma(value, value, sum)
                }
                let inverse = 1 / sqrt(sum / Float(headDimension) + 1e-6)
                for feature in 0..<headDimension {
                    let scale = 1 + Quantization.bf16ToFloat(
                        norm[normBase + feature])
                    result[base + feature] = Float16(
                        Float(values[base + feature]) * inverse * scale)
                }
                for pair in 0..<rotaryPairs {
                    let exponent = -Float(2 * pair) / Float(headDimension)
                    let angle = Float(position) * pow(10_000_000, exponent)
                    let cosine = cos(angle)
                    let sine = sin(angle)
                    let lower = Float(result[base + pair])
                    let upper = Float(result[base + headDimension / 2 + pair])
                    result[base + pair] = Float16(lower * cosine - upper * sine)
                    result[base + headDimension / 2 + pair] = Float16(
                        lower * sine + upper * cosine)
                }
            }
            return result.map(Float.init)
        }

        func emitError(label: String, expected: [Float], actual: UnsafeMutablePointer<Float16>) {
            var maximum: Float = 0
            var sumSquares = 0.0
            for index in expected.indices {
                let error = Float(actual[index]) - expected[index]
                maximum = max(maximum, abs(error))
                sumSquares += Double(error) * Double(error)
            }
            let rms = sqrt(sumSquares / Double(expected.count))
            let line = String(
                format: "mtp attention_stage=%@ n=%d max_abs=%+.6e rms_error=%.6e\\n",
                label, expected.count, maximum, rms)
            FileHandle.standardError.write(Data(line.utf8))
        }

        emitError(
            label: "normalized_query",
            expected: expected(
                values: queryValues,
                norm: queryNorm,
                normBase: queryNormBase,
                headCount: queryHeads),
            actual: normalizedQuery)
        emitError(
            label: "normalized_key",
            expected: expected(
                values: keyValues,
                norm: keyNorm,
                normBase: keyNormBase,
                headCount: keyValueHeads),
            actual: normalizedKey)
    }

    private func emitLMHeadDiagnostics(weights: TensorView,
                                       hidden: MTLBuffer,
                                       logits: MTLBuffer) {
        let vocabularySize = head.geometry.vocabularySize
        let hiddenSize = head.geometry.hiddenSize
        let nativeValues = logits.contents().assumingMemoryBound(to: Float16.self)
        let hiddenValues = hidden.contents().assumingMemoryBound(to: Float16.self)
        var nativeIndex = 0
        var nativeValue = nativeValues[0]
        for index in 1..<vocabularySize where nativeValues[index] > nativeValue {
            nativeIndex = index
            nativeValue = nativeValues[index]
        }

        var referenceIndex = 0
        var referenceValue = -Float.infinity
        var nativeReference: Float = 0
        for row in 0..<vocabularySize {
            let value = Float(Float16(Self.affineDot(
                weights,
                row: row,
                input: hiddenValues,
                count: hiddenSize)))
            if value > referenceValue {
                referenceIndex = row
                referenceValue = value
            }
            if row == nativeIndex {
                nativeReference = value
            }
        }
        let line = String(
            format: "mtp head_reference native_gpu=%d mtp_cpu=%d native_logit=%+.6e mtp_cpu_native=%+.6e\\n",
            nativeIndex,
            referenceIndex,
            Float(nativeValue),
            nativeReference)
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static func affineDot(
        _ view: TensorView,
        row: Int,
        input: UnsafePointer<Float16>,
        count: Int
    ) -> Float {
        let quantization = view.quantization
        let bits = quantization?.bits ?? 4
        let groupSize = quantization?.groupSize ?? Quantization.qwen38GroupSize
        precondition(bits == 4 || bits == 8,
                     "unsupported LM-head diagnostic bit width: \(bits)")
        precondition(count % groupSize == 0,
                     "LM-head width must be divisible by its quantization group size")

        let weights = view.buffer.contents()
            .advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt8.self)
        let scales = view.buffer.contents()
            .advanced(by: Int(view.scaleOffset))
            .assumingMemoryBound(to: UInt16.self)
        let biases = view.buffer.contents()
            .advanced(by: Int(view.biasOffset))
            .assumingMemoryBound(to: UInt16.self)
        let groups = count / groupSize
        let rowBytes = bits == 8 ? count : count / 2
        var result: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(scales[row * groups + group])
            let bias = Quantization.bf16ToFloat(biases[row * groups + group])
            let byteBase = row * rowBytes + group * (bits == 8 ? groupSize : groupSize / 2)
            for index in 0..<groupSize {
                let byte = weights[byteBase + (bits == 8 ? index : index / 2)]
                let quantized = bits == 8
                    ? byte
                    : (index.isMultiple(of: 2) ? byte & 0x0f : byte >> 4)
                result += (Float(quantized) * scale + bias) * Float(input[group * groupSize + index])
            }
        }
        return result
    }

    private func emitDiagnostics(_ stages: [(String, MTLBuffer, Int)],
                                 prefix: String? = nil,
                                 floatLabels: Set<String> = []) {
        for (label, buffer, count) in stages {
            let qualifiedLabel = prefix.map { "\($0).\(label)" } ?? label
            var minimum = Float.infinity
            var maximum = -Float.infinity
            var sum = 0.0
            var sumSquares = 0.0
            let isFloat = floatLabels.contains(label)
            for index in 0..<count {
                let value = isFloat
                    ? buffer.contents().assumingMemoryBound(to: Float.self)[index]
                    : Float(buffer.contents().assumingMemoryBound(to: Float16.self)[index])
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                sum += Double(value)
                sumSquares += Double(value) * Double(value)
            }
            let mean = sum / Double(count)
            let rms = sqrt(sumSquares / Double(count))
            let line = String(
                format: "mtp stage=%@ n=%d min=%+.6e max=%+.6e mean=%+.6e rms=%.6e\\n",
                qualifiedLabel, count, minimum, maximum, mean, rms)
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    private func emitStreamDiagnostics(label: String,
                                       buffer: MTLBuffer,
                                       streamCount: Int = 4,
                                       hiddenSize: Int = 2_560,
                                       isFloat: Bool = false) {
        let floatValues = buffer.contents().assumingMemoryBound(to: Float.self)
        let halfValues = buffer.contents().assumingMemoryBound(to: Float16.self)
        for stream in 0..<streamCount {
            let start = stream * hiddenSize
            var minimum = Float.infinity
            var maximum = -Float.infinity
            var sum = 0.0
            var sumSquares = 0.0
            for index in start..<(start + hiddenSize) {
                let value = isFloat
                    ? floatValues[index]
                    : Float(halfValues[index])
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                sum += Double(value)
                sumSquares += Double(value) * Double(value)
            }
            let mean = sum / Double(hiddenSize)
            let rms = sqrt(sumSquares / Double(hiddenSize))
            let line = String(
                format: "mtp stream=%@ index=%d n=%d min=%+.6e max=%+.6e mean=%+.6e rms=%.6e\\n",
                label, stream, hiddenSize, minimum, maximum, mean, rms)
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    private func emitAttentionHyperConnectionDiagnostics(
        branch: String = "attention",
        hyperInput: MTLBuffer,
        scratch: Qwen38HyperConnectionScratch,
        weights: Qwen38MTPHyperConnectionWeights,
        mixedInput: MTLBuffer,
        branchOutput: MTLBuffer? = nil,
        injectedOutput: MTLBuffer? = nil
    ) {
        let geometry = weights.geometry
        let hiddenSize = Int(geometry.hiddenSize)
        let streamCount = Int(geometry.streamCount)
        let lowRankSize = Int(geometry.lowRankSize)
        let hyperCount = streamCount * hiddenSize
        let inputValues = hyperInput.contents().assumingMemoryBound(to: Float16.self)
        let normalizedValues = scratch.normalized.contents()
            .assumingMemoryBound(to: Float.self)
        let normValues = weights.weights.norm.contents()
            .assumingMemoryBound(to: UInt16.self)
        let normBase = weights.weights.normOffset / MemoryLayout<UInt16>.stride
        var expectedNormalized = [Float](repeating: 0, count: hyperCount)
        for stream in 0..<streamCount {
            let streamBase = stream * hiddenSize
            var sum: Float = 0
            for feature in 0..<hiddenSize {
                let value = Float(inputValues[streamBase + feature])
                sum = fma(value, value, sum)
            }
            let inverse = 1 / sqrt(sum / Float(hiddenSize) + 1e-6)
            for feature in 0..<hiddenSize {
                let scale = Quantization.bf16ToFloat(
                    normValues[normBase + streamBase + feature])
                expectedNormalized[streamBase + feature] = Float(Float16(
                    Float(inputValues[streamBase + feature]) * inverse)) * scale
            }
        }
        emitHyperConnectionError(
            branch: branch,
            relation: "normalized",
            expected: expectedNormalized,
            actual: scratch.normalized,
            count: hyperCount,
            actualIsFloat: true)

        let lowRankValues = scratch.lowRank.contents()
            .assumingMemoryBound(to: Float16.self)
        var expectedActivated = [Float](repeating: 0, count: lowRankSize)
        for index in 0..<lowRankSize {
            let value = Float(lowRankValues[index]) / Float(streamCount)
            expectedActivated[index] = value / (1 + exp(-value))
        }
        emitHyperConnectionError(
            branch: branch,
            relation: "low_rank_silu",
            expected: expectedActivated,
            actual: scratch.activatedLowRank,
            count: lowRankSize)

        let mixLogits = scratch.mixLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        var expectedMixed = [Float](repeating: 0, count: hiddenSize)
        for feature in 0..<hiddenSize {
            var mixed: Float = 0
            for stream in 0..<streamCount {
                let index = stream * hiddenSize + feature
                let gate = 1 / (1 + exp(-Float(mixLogits[index])))
                mixed = fma(gate, Float(normalizedValues[index]), mixed)
            }
            expectedMixed[feature] = mixed / Float(streamCount)
        }
        emitHyperConnectionError(
            branch: branch,
            relation: "mix_streams",
            expected: expectedMixed,
            actual: mixedInput,
            count: hiddenSize)

        if weights.weights.blockInject != nil,
           let branchOutput,
           let injectedOutput {
            let injectionLogits = scratch.injectionLogits.contents()
                .assumingMemoryBound(to: Float16.self)
            var expectedInjectionWeights = [Float](repeating: 0, count: streamCount)
            for stream in 0..<streamCount {
                let value = Float(injectionLogits[stream]) / Float(streamCount)
                expectedInjectionWeights[stream] = 2 / (1 + exp(-value))
            }
            emitHyperConnectionError(
                branch: branch,
                relation: "injection_weights",
                expected: expectedInjectionWeights,
                actual: scratch.injectionWeights,
                count: streamCount)

            let branchValues = branchOutput.contents()
                .assumingMemoryBound(to: Float16.self)
            let injectionWeights = scratch.injectionWeights.contents()
                .assumingMemoryBound(to: Float16.self)
            var expectedInjected = [Float](repeating: 0, count: hyperCount)
            for stream in 0..<streamCount {
                for feature in 0..<hiddenSize {
                    let index = stream * hiddenSize + feature
                    expectedInjected[index] = Float(inputValues[index])
                        + Float(branchValues[feature])
                        * Float(injectionWeights[stream])
                }
            }
            emitHyperConnectionError(
                branch: branch,
                relation: "inject_streams",
                expected: expectedInjected,
                actual: injectedOutput,
                count: hyperCount)
        }
    }

    private func emitHyperConnectionError(
        branch: String,
        relation: String,
        expected: [Float],
        actual: MTLBuffer,
        count: Int,
        actualIsFloat: Bool = false
    ) {
        let floatValues = actual.contents().assumingMemoryBound(to: Float.self)
        let halfValues = actual.contents().assumingMemoryBound(to: Float16.self)
        var maximumError: Float = 0
        var sumSquares = 0.0
        for index in 0..<count {
            let actualValue = actualIsFloat
                ? floatValues[index]
                : Float(halfValues[index])
            let error = actualValue - expected[index]
            maximumError = max(maximumError, abs(error))
            sumSquares += Double(error) * Double(error)
        }
        let rmsError = sqrt(sumSquares / Double(count))
        let line = String(
            format: "mtp hyper=%@ relation=%@ n=%d max_abs=%+.6e rms_error=%.6e\\n",
            branch, relation, count, maximumError, rmsError)
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func emitAttentionPathDiagnostics(
        split: MTLBuffer,
        prefill: MTLBuffer
    ) {
        let splitValues = split.contents().assumingMemoryBound(to: Float16.self)
        let prefillValues = prefill.contents().assumingMemoryBound(to: Float16.self)
        let count = Qwen38MTPExecutionGeometry.qwen.fullAttentionQueryHeads
            * Qwen38MTPExecutionGeometry.qwen.fullAttentionHeadDimension
        var maximumError: Float = 0
        var sumSquares = 0.0
        for index in 0..<count {
            let error = Float(splitValues[index]) - Float(prefillValues[index])
            maximumError = max(maximumError, abs(error))
            sumSquares += Double(error) * Double(error)
        }
        let rmsError = sqrt(sumSquares / Double(count))
        let line = String(
            format: "mtp attention_paths n=%d max_abs=%+.6e rms_error=%.6e\\n",
            count, maximumError, rmsError)
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func emitAttentionReferenceDiagnostics(
        query: MTLBuffer,
        cache: QwenFullAttentionKVCache,
        tokenMask: MTLBuffer,
        actual: MTLBuffer
    ) {
        let actualValues = actual.contents().assumingMemoryBound(to: Float16.self)
        for (label, mask) in [("masked", tokenMask as MTLBuffer?), ("unmasked", nil)] {
            let expected = referenceAttentionOutput(
                query: query,
                cache: cache,
                tokenMask: mask)
            var maximumError: Float = 0
            var sumSquares = 0.0
            for index in 0..<expected.count {
                let error = Float(actualValues[index]) - expected[index]
                maximumError = max(maximumError, abs(error))
                sumSquares += Double(error) * Double(error)
            }
            let rmsError = sqrt(sumSquares / Double(expected.count))
            let line = String(
                format: "mtp attention_reference mask=%@ n=%d max_abs=%+.6e rms_error=%.6e\\n",
                label, expected.count, maximumError, rmsError)
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    private func referenceAttentionOutput(
        query: MTLBuffer,
        cache: QwenFullAttentionKVCache,
        tokenMask: MTLBuffer?
    ) -> [Float] {
        let geometry = cache.geometry
        let queryValues = query.contents().assumingMemoryBound(to: Float16.self)
        let keyValues = cache.key.contents().assumingMemoryBound(to: Float16.self)
        let valueValues = cache.value.contents().assumingMemoryBound(to: Float16.self)
        let maskValues = tokenMask?.contents().assumingMemoryBound(to: UInt8.self)
        let queryHeads = geometry.queryHeads
        let headDimension = geometry.headDimension
        let keyValueHeads = geometry.keyValueHeads
        let keyCount = cache.count
        let queryGroupSize = queryHeads / keyValueHeads
        var output = [Float](repeating: 0, count: geometry.queryWidth)

        for queryHead in 0..<queryHeads {
            let keyValueHead = queryHead / queryGroupSize
            let queryBase = queryHead * headDimension
            var scores = [Float](repeating: -.infinity, count: keyCount)
            var maximumScore = -Float.infinity
            for position in 0..<keyCount
                where maskValues == nil || maskValues![position] != 0 {
                let keyBase = (position * keyValueHeads + keyValueHead) * headDimension
                var dot: Float = 0
                for feature in 0..<headDimension {
                    dot += Float(queryValues[queryBase + feature])
                        * Float(keyValues[keyBase + feature])
                }
                let score = dot * geometry.attentionScale
                scores[position] = score
                maximumScore = max(maximumScore, score)
            }
            guard maximumScore.isFinite else { continue }

            var denominator: Float = 0
            for position in 0..<keyCount where scores[position].isFinite {
                denominator += exp(scores[position] - maximumScore)
            }
            guard denominator > 0 else { continue }
            for feature in 0..<headDimension {
                var weightedValue: Float = 0
                for position in 0..<keyCount where scores[position].isFinite {
                    let keyValueBase = (position * keyValueHeads + keyValueHead)
                        * headDimension
                    let weight = exp(scores[position] - maximumScore) / denominator
                    weightedValue += weight * Float(valueValues[keyValueBase + feature])
                }
                output[queryBase + feature] = weightedValue
            }
        }
        return output
    }

    private func emitQSASelectionDiagnostics(
        state: Qwen38MTPState,
        scratch: Qwen38MTPAttentionScratch
    ) {
        let visibleCount = scratch.visibleTokenCounts.contents()
            .assumingMemoryBound(to: UInt32.self)[0]
        let rotaryPosition = scratch.queryPositions.contents()
            .assumingMemoryBound(to: UInt32.self)[0]
        let keyCount = state.qsa.rawKeyCache.count
        let selectorState = scratch.qsaSelectionState.contents()
            .assumingMemoryBound(to: UInt32.self)
        let threshold = Float(bitPattern: selectorState[0])
        let remaining = selectorState[1]
        let visibleBlocks = selectorState[2]
        let blockTopK = selectorState[3]
        let selectedPositions = scratch.qsaTokenMask.contents()
            .assumingMemoryBound(to: UInt8.self)
        let cachePositions = state.qsa.rawKeyCache.positions.contents()
            .assumingMemoryBound(to: UInt32.self)
        var selected = [String]()
        for index in 0..<keyCount where selectedPositions[index] != 0 {
            selected.append(String(cachePositions[index]))
        }
        let line = "mtp qsa state=\(state.position) rotary=\(rotaryPosition) "
            + "visible=\(visibleCount) key=\(keyCount) "
            + "blocks=\(visibleBlocks) topk=\(blockTopK) "
            + "remaining=\(remaining) threshold=\(threshold) "
            + "selected=\(selected.joined(separator: ","))\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func emitRouteDiagnostics(experts: [Int], weights: [Float]) {
        let routes = zip(experts, weights)
            .map { route in "\(route.0):\(route.1)" }
            .joined(separator: ",")
        FileHandle.standardError.write(Data("mtp routes=\(routes)\n".utf8))
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        return commandBuffer
    }

    private func commitAndWait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)
    }
}
