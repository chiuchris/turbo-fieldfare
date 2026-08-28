import Foundation
import Metal

private struct Qwen38PromptStateSnapshot {
    let position: Int
    let ngramContext: [Int64]
    let runtimeState: Qwen38RuntimeStateSnapshot
}

private final class Qwen38RunnerScratch {
    let hiddenStreams: MTLBuffer
    let alternateStreams: MTLBuffer
    let mixedInput: MTLBuffer
    let attentionOutput: MTLBuffer
    let afterAttention: MTLBuffer
    let mlpOutput: MTLBuffer
    let finalHidden: MTLBuffer
    let ngramEmbedding: MTLBuffer
    let normed: MTLBuffer
    let projection: MTLBuffer
    let query: MTLBuffer
    let queryGate: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let normalizedQuery: MTLBuffer
    let normalizedKey: MTLBuffer
    let attentionOutputHeads: MTLBuffer
    let gatedAttention: MTLBuffer
    let delta: Qwen38DeltaNetScratch
    let attentionHyperConnection: Qwen38HyperConnectionScratch
    let mlpHyperConnection: Qwen38HyperConnectionScratch
    let finalHyperConnection: Qwen38HyperConnectionScratch
    let sharedOutput: MTLBuffer
    let qsaProjectedRows: MTLBuffer
    let qsaBlockScores: MTLBuffer
    let qsaSelectionState: MTLBuffer
    let qsaTokenMask: MTLBuffer
    let queryPositions: MTLBuffer
    let visibleTokenCounts: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let ple: Qwen38PLEScratch

    init(device: MTLDevice, config: ArchConfig, maxContext: Int) throws {
        func makeBuffer(_ elements: Int,
                        stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }

        let streamCount = 4
        let hiddenSize = config.hiddenSize
        let pleEmbeddingSize = config.qwen38Architecture?.pleEmbeddingSize ?? hiddenSize
        let hyperWidth = streamCount * hiddenSize
        let lowRank = 320
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        let deltaKeyWidth = config.linearNumKeyHeads * config.linearKeyHeadDim
        let deltaValueWidth = config.linearNumValueHeads * config.linearValueHeadDim
        let deltaQKVWidth = deltaKeyWidth * 2 + deltaValueWidth
        let qsa = Qwen38QSAGeometry.qwen
        let compressionRatio = Int(qsa.compressRatio)
        let blockCount = max(1, maxContext / compressionRatio)

        hiddenStreams = try makeBuffer(hyperWidth)
        alternateStreams = try makeBuffer(hyperWidth)
        mixedInput = try makeBuffer(hiddenSize)
        attentionOutput = try makeBuffer(hiddenSize)
        afterAttention = try makeBuffer(hyperWidth)
        mlpOutput = try makeBuffer(hiddenSize)
        finalHidden = try makeBuffer(hiddenSize)
        ngramEmbedding = try makeBuffer(pleEmbeddingSize)
        normed = try makeBuffer(hiddenSize)
        projection = try makeBuffer(qWidth * 2)
        query = try makeBuffer(qWidth)
        queryGate = try makeBuffer(qWidth)
        key = try makeBuffer(kvWidth)
        value = try makeBuffer(kvWidth)
        normalizedQuery = try makeBuffer(qWidth)
        normalizedKey = try makeBuffer(kvWidth)
        attentionOutputHeads = try makeBuffer(qWidth)
        gatedAttention = try makeBuffer(qWidth)
        delta = Qwen38DeltaNetScratch(
            qkv: try makeBuffer(deltaQKVWidth),
            gate: try makeBuffer(deltaValueWidth),
            betaInput: try makeBuffer(config.linearNumValueHeads),
            decayInput: try makeBuffer(config.linearNumValueHeads),
            convolution: try makeBuffer(deltaQKVWidth),
            query: try makeBuffer(deltaKeyWidth),
            key: try makeBuffer(deltaKeyWidth),
            value: try makeBuffer(deltaValueWidth),
            decay: try makeBuffer(config.linearNumValueHeads, stride: MemoryLayout<Float>.stride),
            beta: try makeBuffer(config.linearNumValueHeads, stride: MemoryLayout<Float>.stride),
            recurrent: try makeBuffer(deltaValueWidth),
            normalized: try makeBuffer(deltaValueWidth))
        attentionHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(hyperWidth),
            lowRank: try makeBuffer(lowRank),
            activatedLowRank: try makeBuffer(lowRank),
            mixLogits: try makeBuffer(hyperWidth),
            injectionLogits: try makeBuffer(streamCount),
            injectionWeights: try makeBuffer(streamCount))
        mlpHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(hyperWidth),
            lowRank: try makeBuffer(lowRank),
            activatedLowRank: try makeBuffer(lowRank),
            mixLogits: try makeBuffer(hyperWidth),
            injectionLogits: try makeBuffer(streamCount),
            injectionWeights: try makeBuffer(streamCount))
        finalHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(hyperWidth),
            lowRank: try makeBuffer(lowRank),
            activatedLowRank: try makeBuffer(lowRank),
            mixLogits: try makeBuffer(hyperWidth),
            injectionLogits: try makeBuffer(streamCount),
            injectionWeights: try makeBuffer(streamCount))
        sharedOutput = try makeBuffer(hiddenSize)
        qsaProjectedRows = try makeBuffer(Int(qsa.projectionWidth))
        qsaBlockScores = try makeBuffer(maxContext * blockCount, stride: MemoryLayout<Float>.stride)
        qsaSelectionState = try makeBuffer(Qwen38QSASelector.stateBytesPerQuery,
                                           stride: 1)
        qsaTokenMask = try makeBuffer(maxContext * maxContext, stride: 1)
        queryPositions = try makeBuffer(1, stride: MemoryLayout<UInt32>.stride)
        visibleTokenCounts = try makeBuffer(1, stride: MemoryLayout<UInt32>.stride)
        sharedGateScratch = try makeBuffer(config.intermediateSize)
        sharedUpScratch = try makeBuffer(config.intermediateSize)
        sharedActScratch = try makeBuffer(config.intermediateSize)
        ple = Qwen38PLEScratch(
            projectedKey: try makeBuffer(hyperWidth),
            value: try makeBuffer(hiddenSize),
            normalizedKey: try makeBuffer(hyperWidth),
            normalizedQuery: try makeBuffer(hyperWidth),
            gatedValue: try makeBuffer(hyperWidth),
            normalizedGatedValue: try makeBuffer(hyperWidth),
            convolution: try makeBuffer(hyperWidth))
    }
}

public final class Qwen38ForwardRunner: ForwardRunner, ContinuableLogitProducer,
    PromptStateSnapshotting, ChunkedPrefillRunner, @unchecked Sendable {
    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let embed: EmbedLookupInt4
    private let rms: RMSNorm
    private let plePipeline: Qwen38PLEPipeline
    private let projection: Qwen38PLEProjection
    private let streamOps: Qwen38GatedResidual
    private let sharedExpert: QwenSharedExpertInt4
    private let attention: QwenFullAttention
    private let deltaNet: Qwen38DeltaNetDecoder
    private let qsa: Qwen38QSALayerExecutor
    private let decoder: Qwen38DecoderLayerExecutor
    private let finalMixer: Qwen38HyperConnection
    private let moe: Qwen38MoE
    private let head: QwenUntiedLMHead
    private let runtimeState: Qwen38RuntimeState
    private let layers: [Qwen38DecoderLayerWeights]
    private let moeWeights: [Qwen38MoEWeights]
    private let scratch: Qwen38RunnerScratch
    private let finalHyperConnection: Qwen38HyperConnectionWeights
    private let pleLayer: Int
    private let pleWeights: Qwen38PLEWeights
    private let pleAddressing: Qwen38PLEAddressing
    private let ngramStreamer: PreadNgramStreamer

    public let maxContext: Int
    public private(set) var continuationPosition = 0
    private var ngramContext: [Int64] = []
    private var promptStateSnapshot: Qwen38PromptStateSnapshot?

    public init(model: Model,
                context: MetalContext,
                maxContext: Int,
                runtimeConfiguration _: RuntimeConfiguration = .production) throws {
        guard model.config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(model.config.modelFamily)")
        }
        guard maxContext > 0 else {
            throw ModelError.archMismatch(
                field: "maxContext",
                expected: "positive",
                actual: "\(maxContext)")
        }
        guard let architecture = model.config.qwen38Architecture,
              architecture.pleLayerIDs == [2] else {
            throw ModelError.archMismatch(
                field: "qwen38Architecture.pleLayerIDs",
                expected: "[2]",
                actual: "\(model.config.qwen38Architecture?.pleLayerIDs ?? [])")
        }
        let pleLayer = architecture.pleLayerIDs[0] - 1
        let pleWeights = try Qwen38PLEWeights(model: model, layer: pleLayer)
        let ngramStreamer = try model.qwen38NgramStreamer()
        let ngramHeadCount = (architecture.ngramSize - 1) * architecture.headsPerNgram
        guard ngramHeadCount > 0,
              ngramStreamer.layout.rowWidth * ngramHeadCount
                  == architecture.pleEmbeddingSize else {
            throw ModelError.archMismatch(
                field: "packedNgrams.rowWidth",
                expected: "pleEmbeddingSize / \(ngramHeadCount)",
                actual: "\(ngramStreamer.layout.rowWidth)")
        }
        let pleAddressing = Qwen38PLEAddressing(
            ngramSize: architecture.ngramSize,
            headsPerNgram: architecture.headsPerNgram,
            unigramVocabSize: Int64(model.config.vocabSize),
            ngramVocabSizeBase: Int64(architecture.ngramVocabSizeBase),
            pleLayerIndex: pleLayer,
            vocabDivisor: Int64(architecture.ngramVocabSizeDivisor))

        self.model = model
        self.context = context
        self.config = model.config
        self.maxContext = maxContext
        self.embed = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.plePipeline = try Qwen38PLEPipeline(context: context)
        self.projection = try Qwen38PLEProjection(context: context)
        self.streamOps = try Qwen38GatedResidual(context: context)
        self.sharedExpert = try QwenSharedExpertInt4(context: context)
        self.attention = try QwenFullAttention(
            context: context,
            geometry: QwenFullAttentionGeometry(
                queryHeads: model.config.numHeads,
                keyValueHeads: model.config.numFullKVHeads,
                headDimension: model.config.fullHeadDim,
                rotaryDimension: Int(
                    Double(model.config.fullHeadDim) * model.config.partialRotaryFactor),
                ropeTheta: Float(model.config.fullRopeTheta)))
        self.deltaNet = try Qwen38DeltaNetDecoder(
            context: context,
            geometry: Qwen38DeltaNetGeometry(
                hiddenSize: UInt32(model.config.hiddenSize),
                keyHeads: UInt32(model.config.linearNumKeyHeads),
                valueHeads: UInt32(model.config.linearNumValueHeads),
                keyHeadDimension: UInt32(model.config.linearKeyHeadDim),
                valueHeadDimension: UInt32(model.config.linearValueHeadDim),
                convolutionKernel: UInt32(model.config.linearConvKernelDim)))
        self.qsa = try Qwen38QSALayerExecutor(context: context)
        self.decoder = try Qwen38DecoderLayerExecutor(context: context)
        self.finalMixer = try Qwen38HyperConnection(context: context)
        self.moe = try Qwen38MoE(context: context)
        self.head = try QwenUntiedLMHead(
            context: context,
            geometry: QwenLMHeadGeometry(
                vocabularySize: model.config.vocabSize,
                hiddenSize: model.config.hiddenSize))
        self.runtimeState = try Qwen38RuntimeState(model: model, maxContext: maxContext)
        self.scratch = try Qwen38RunnerScratch(
            device: context.device,
            config: model.config,
            maxContext: maxContext)
        self.layers = try (0..<model.config.numLayers).map {
            try Qwen38DecoderLayerWeights(model: model, layer: $0)
        }
        self.moeWeights = try (0..<model.config.numLayers).map {
            try Qwen38MoEWeights(model: model, layer: $0)
        }
        self.finalHyperConnection = try model.qwen38FinalHyperConnectionWeights()
        self.pleLayer = pleLayer
        self.pleWeights = pleWeights
        self.pleAddressing = pleAddressing
        self.ngramStreamer = ngramStreamer

        _ = model.embedding
        _ = model.finalNorm
        _ = model.lmHead
    }

    public func reset() {
        continuationPosition = 0
        ngramContext.removeAll(keepingCapacity: true)
        promptStateSnapshot = nil
        runtimeState.reset()
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, expectedPosition == continuationPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected position \(expectedPosition), current \(continuationPosition)")
        }
    }

    public func savePromptState() {
        promptStateSnapshot = Qwen38PromptStateSnapshot(
            position: continuationPosition,
            ngramContext: ngramContext,
            runtimeState: runtimeState.snapshot())
    }

    public func restorePromptState(expectedPosition: Int) throws {
        guard let snapshot = promptStateSnapshot,
              snapshot.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "prompt replay expected position \(expectedPosition) has no matching snapshot")
        }
        runtimeState.restore(snapshot.runtimeState)
        ngramContext = snapshot.ngramContext
        continuationPosition = snapshot.position
    }

    public func produce(token: Int32,
                        position: Int,
                        into logits: MTLBuffer) async throws {
        try Task.checkCancellation()
        guard position == continuationPosition else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(continuationPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        guard logits.length >= config.vocabSize * MemoryLayout<Float16>.stride else {
            throw ModelError.residentBufferWrapFailed
        }

        var nextNgramContext = ngramContext
        let addresses = pleAddressing.addresses(
            tokens: [Int64(token)], context: &nextNgramContext)
        try writeNgramEmbedding(addresses: addresses[0])

        try runSync { commandBuffer in
            let embedding = model.embedding
            embed.encode(
                commandBuffer: commandBuffer,
                table: embedding.buffer,
                tableOffset: Int(embedding.offset),
                scales: embedding.buffer,
                scalesOffset: Int(embedding.scaleOffset),
                biases: embedding.buffer,
                biasesOffset: Int(embedding.biasOffset),
                out: scratch.finalHidden,
                tokenId: UInt32(bitPattern: token),
                d: UInt32(config.hiddenSize),
                outScale: 1)
            streamOps.encodeRepeatStreams(
                commandBuffer: commandBuffer,
                input: scratch.finalHidden,
                output: scratch.hiddenStreams,
                tokenCount: 1,
                streamCount: 4,
                hiddenSize: UInt32(config.hiddenSize))
        }

        var inputStreams = scratch.hiddenStreams
        var outputStreams = scratch.alternateStreams
        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            if layer == pleLayer {
                try runSync { commandBuffer in
                    plePipeline.encode(
                        commandBuffer: commandBuffer,
                        embedding: scratch.ngramEmbedding,
                        hiddenStates: inputStreams,
                        weights: pleWeights,
                        scratch: scratch.ple,
                        state: runtimeState.pleConvolution,
                        output: outputStreams,
                        tokenCount: 1,
                        streamCount: 4,
                        hiddenSize: UInt32(config.hiddenSize),
                        embeddingSize: UInt32(config.qwen38Architecture?.pleEmbeddingSize ?? config.hiddenSize),
                        epsilon: 1e-6)
                }
                swap(&inputStreams, &outputStreams)
            }
            try await encodeLayer(
                layer: layer,
                position: position,
                inputStreams: inputStreams,
                outputStreams: outputStreams)
            swap(&inputStreams, &outputStreams)
        }
        try runSync { commandBuffer in
            decoderFinalPrepare(
                commandBuffer: commandBuffer,
                input: inputStreams)
            rms.encodeBF16W(
                commandBuffer: commandBuffer,
                x: scratch.mixedInput,
                weight: model.finalNorm.buffer,
                weightOffset: Int(model.finalNorm.offset),
                out: scratch.normed,
                d: UInt32(config.hiddenSize),
                eps: 1e-6)
            let lmHead = model.lmHead
            head.encode(
                commandBuffer: commandBuffer,
                weights: lmHead.buffer,
                weightsOffset: Int(lmHead.offset),
                scales: lmHead.buffer,
                scalesOffset: Int(lmHead.scaleOffset),
                biases: lmHead.buffer,
                biasesOffset: Int(lmHead.biasOffset),
                hidden: scratch.normed,
                logits: logits)
        }
        ngramContext = nextNgramContext
        continuationPosition += 1
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode _: PrefillOutputMode,
                               config _: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard startPosition == continuationPosition else {
            throw PrefillError.prefillCursorMismatch(
                "prefill start \(startPosition) != current position \(continuationPosition)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: continuationPosition, seed: .logitsWritten)
        }
        for (offset, token) in tokens.enumerated() {
            try await produce(
                token: token,
                position: startPosition + offset,
                into: logits)
            onProgress(offset + 1)
        }
        return PrefillResult(newPosition: continuationPosition, seed: .logitsWritten)
    }

    private func encodeLayer(layer: Int,
                             position: Int,
                             inputStreams: MTLBuffer,
                             outputStreams: MTLBuffer) async throws {
        let layerScratch = Qwen38DecoderLayerScratch(
            attentionHyperConnection: scratch.attentionHyperConnection,
            attentionInput: scratch.mixedInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
        try runSync { commandBuffer in
            decoder.encodeAttentionPrepare(
                commandBuffer: commandBuffer,
                weights: layers[layer],
                hyperInput: inputStreams,
                scratch: layerScratch,
                tokenCount: 1,
                epsilon: 1e-6)
            let state = try decoder.attentionState(
                layer: layer,
                runtimeState: runtimeState)
            try encodeAttention(
                commandBuffer: commandBuffer,
                layer: layer,
                state: state,
                input: scratch.mixedInput,
                output: scratch.attentionOutput,
                position: position)
            decoder.encodeAttentionInject(
                commandBuffer: commandBuffer,
                hyperInput: inputStreams,
                scratch: layerScratch,
                tokenCount: 1)
            decoder.encodeMLPPrepare(
                commandBuffer: commandBuffer,
                weights: layers[layer],
                scratch: layerScratch,
                tokenCount: 1,
                epsilon: 1e-6)
            moe.encodeRouter(
                commandBuffer: commandBuffer,
                weights: moeWeights[layer],
                hidden: scratch.mixedInput,
                hiddenSize: UInt32(config.hiddenSize))
            moe.encodeSelection(
                commandBuffer: commandBuffer,
                weights: moeWeights[layer])
        }

        let fetched = try await moe.fetchSelectedExperts(model: model, layer: layer)
        try runSync { commandBuffer in
            try sharedExpert.encode(
                commandBuffer: commandBuffer,
                x: scratch.mixedInput,
                gate: sharedProjection(moeWeights[layer].sharedExpertGate,
                                       rows: config.intermediateSize,
                                       cols: config.hiddenSize),
                up: sharedProjection(moeWeights[layer].sharedExpertUp,
                                     rows: config.intermediateSize,
                                     cols: config.hiddenSize),
                down: sharedProjection(moeWeights[layer].sharedExpertDown,
                                       rows: config.hiddenSize,
                                       cols: config.intermediateSize),
                y: scratch.sharedOutput,
                scratchGate: scratch.sharedGateScratch,
                scratchUp: scratch.sharedUpScratch,
                scratchAct: scratch.sharedActScratch)
            let routedArguments = try moe.makeRoutedArgumentBuffer(
                experts: fetched.views)
            moe.encodeRouted(
                commandBuffer: commandBuffer,
                routedArgumentBuffer: routedArguments,
                routedOffsets: fetched.offsets,
                input: scratch.mixedInput,
                residual: scratch.sharedOutput,
                output: scratch.mlpOutput,
                hiddenSize: UInt32(config.hiddenSize),
                intermediateSize: UInt32(config.moeIntermediateSize),
                sharedExpertGateWeight: moeWeights[layer].sharedExpertGateWeight)
            decoder.encodeMLPInject(
                commandBuffer: commandBuffer,
                scratch: layerScratch,
                output: outputStreams,
                tokenCount: 1)
        }
    }

    private func encodeAttention(commandBuffer: MTLCommandBuffer,
                                 layer: Int,
                                 state: Qwen38DecoderAttentionState,
                                 input: MTLBuffer,
                                 output: MTLBuffer,
                                 position: Int) throws {
        switch state {
        case .linear:
            let weights = try Qwen38DeltaNetWeights(model: model, layer: layer)
            try deltaNet.encode(
                commandBuffer: commandBuffer,
                state: state,
                weights: weights,
                input: input,
                scratch: scratch.delta,
                output: output,
                epsilon: 1e-6)
        case .sparse(let qsaState, let cache):
            let weights = try model.qwenFullAttentionWeights(layer: layer)
            scratch.queryPositions.contents()
                .assumingMemoryBound(to: UInt32.self)[0] = UInt32(position)
            scratch.visibleTokenCounts.contents()
                .assumingMemoryBound(to: UInt32.self)[0] = UInt32(position + 1)
            let qsaScratch = Qwen38QSALayerExecutionScratch(
                projectedRows: scratch.qsaProjectedRows,
                blockScores: scratch.qsaBlockScores,
                selection: Qwen38QSASelectionScratch(
                    state: scratch.qsaSelectionState,
                    tokenMask: scratch.qsaTokenMask))
            qsa.encode(
                commandBuffer: commandBuffer,
                state: qsaState,
                hiddenStates: input,
                queryPositions: scratch.queryPositions,
                visibleTokenCounts: scratch.visibleTokenCounts,
                scratch: qsaScratch,
                tokenCount: 1,
                inputWidth: UInt32(config.hiddenSize),
                epsilon: 1e-6)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.q,
                input: input,
                output: scratch.projection,
                outputWidth: config.numHeads * config.fullHeadDim * 2)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.k,
                input: input,
                output: scratch.key,
                outputWidth: config.numFullKVHeads * config.fullHeadDim)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.v,
                input: input,
                output: scratch.value,
                outputWidth: config.numFullKVHeads * config.fullHeadDim)
            attention.encodeSplitQueryGate(
                commandBuffer: commandBuffer,
                projection: scratch.projection,
                query: scratch.query,
                gate: scratch.queryGate)
            attention.encodeQueryKey(
                commandBuffer: commandBuffer,
                query: scratch.query,
                key: scratch.key,
                queryNorm: weights.qNorm.buffer,
                queryNormOffset: Int(weights.qNorm.offset),
                keyNorm: weights.kNorm.buffer,
                keyNormOffset: Int(weights.kNorm.offset),
                normalizedQuery: scratch.normalizedQuery,
                normalizedKey: scratch.normalizedKey,
                position: UInt32(position),
                epsilon: 1e-6)
            cache.append(
                commandBuffer: commandBuffer,
                key: scratch.normalizedKey,
                value: scratch.value)
            attention.encode(
                commandBuffer: commandBuffer,
                query: scratch.normalizedQuery,
                keyValueCache: cache,
                output: scratch.attentionOutputHeads,
                tokenMask: scratch.qsaTokenMask)
            attention.encodeOutputGate(
                commandBuffer: commandBuffer,
                attention: scratch.attentionOutputHeads,
                gate: scratch.queryGate,
                output: scratch.gatedAttention)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.o,
                input: scratch.gatedAttention,
                output: output,
                outputWidth: config.hiddenSize,
                inputWidth: config.numHeads * config.fullHeadDim)
        }
    }

    private func writeNgramEmbedding(addresses: [Int64]) throws {
        guard !addresses.isEmpty else {
            throw ModelError.indexCorrupt(detail: "Qwen3.8 PLE produced no n-gram addresses")
        }
        let rows = try ngramStreamer.read(addresses: addresses)
        let embeddingSize = config.qwen38Architecture?.pleEmbeddingSize ?? 0
        guard rows.count == addresses.count,
              !rows.isEmpty,
              embeddingSize.isMultiple(of: rows.count),
              rows.allSatisfy({ $0.count == embeddingSize / rows.count }) else {
            throw ModelError.archMismatch(
                field: "packedNgrams.rowWidth",
                expected: "pleEmbeddingSize / addressCount",
                actual: "\(rows.first?.count ?? 0)")
        }
        let headWidth = embeddingSize / rows.count
        let output = scratch.ngramEmbedding.contents().assumingMemoryBound(to: UInt16.self)
        for (head, row) in rows.enumerated() {
            for (index, value) in row.enumerated() {
                output[head * headWidth + index] = Float16(value).bitPattern
            }
        }
    }

    private func decoderFinalPrepare(commandBuffer: MTLCommandBuffer,
                                     input: MTLBuffer) {
        finalMixer.encodePrepare(
            commandBuffer: commandBuffer,
            hyperInput: input,
            weights: finalHyperConnection,
            scratch: scratch.finalHyperConnection,
            mixedInput: scratch.mixedInput,
            tokenCount: 1,
            epsilon: 1e-6)
    }

    private func encodeProjection(commandBuffer: MTLCommandBuffer,
                                  weights: TensorView,
                                  input: MTLBuffer,
                                  output: MTLBuffer,
                                  outputWidth: Int,
                                  inputWidth: Int? = nil) {
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.buffer,
            weightsOffset: Int(weights.offset),
            scales: weights.buffer,
            scalesOffset: Int(weights.scaleOffset),
            biases: weights.buffer,
            biasesOffset: Int(weights.biasOffset),
            input: input,
            output: output,
            tokenCount: 1,
            outputWidth: UInt32(outputWidth),
            inputWidth: UInt32(inputWidth ?? config.hiddenSize))
    }

    private func sharedProjection(_ view: TensorView,
                                  rows: Int,
                                  cols: Int) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: view.buffer,
            scales: view.buffer,
            biases: view.buffer,
            weightsOffset: Int(view.offset),
            scalesOffset: Int(view.scaleOffset),
            biasesOffset: Int(view.biasOffset),
            rows: UInt32(rows),
            cols: UInt32(cols))
    }

    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try body(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        guard commandBuffer.status == .completed else {
            throw ModelError.residentBufferWrapFailed
        }
    }
}
