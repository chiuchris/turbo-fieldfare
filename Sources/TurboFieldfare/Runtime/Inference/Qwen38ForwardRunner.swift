import Foundation
import Metal

public struct Qwen38DecodeTimingSample: Codable, Sendable, Equatable {
    public let embeddingNanos: UInt64
    public let pleNanos: UInt64
    public let attentionRouterNanos: UInt64
    public let expertFetchNanos: UInt64
    public let expertCacheHits: Int
    public let expertCacheMisses: Int
    public let moeNanos: UInt64
    public let finalHeadNanos: UInt64
    public let gpuActiveNanos: UInt64
    public let commandBufferCount: Int

    public static let zero = Qwen38DecodeTimingSample(
        embeddingNanos: 0,
        pleNanos: 0,
        attentionRouterNanos: 0,
        expertFetchNanos: 0,
        moeNanos: 0,
        finalHeadNanos: 0,
        gpuActiveNanos: 0,
        commandBufferCount: 0)

    public init(embeddingNanos: UInt64,
                pleNanos: UInt64,
                attentionRouterNanos: UInt64,
                expertFetchNanos: UInt64,
                expertCacheHits: Int = 0,
                expertCacheMisses: Int = 0,
                moeNanos: UInt64,
                finalHeadNanos: UInt64,
                gpuActiveNanos: UInt64,
                commandBufferCount: Int) {
        self.embeddingNanos = embeddingNanos
        self.pleNanos = pleNanos
        self.attentionRouterNanos = attentionRouterNanos
        self.expertFetchNanos = expertFetchNanos
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.moeNanos = moeNanos
        self.finalHeadNanos = finalHeadNanos
        self.gpuActiveNanos = gpuActiveNanos
        self.commandBufferCount = commandBufferCount
    }
}

private struct Qwen38LayerDecodeTiming {
    let attentionRouterNanos: UInt64
    let expertFetchNanos: UInt64
    let moeNanos: UInt64
    let gpuActiveNanos: UInt64
}

private struct Qwen38LayerPrefillTiming {
    let mixerNanos: UInt64
    let expertFetchNanos: UInt64
    let routedMoENanos: UInt64
    let cacheHits: Int
    let cacheMisses: Int
    let expertReadCount: Int
    let expertReadNanos: UInt64
    let expertReadMaxNanos: UInt64
}

private struct Qwen38PromptStateSnapshot {
    let position: Int
    let ngramContext: [Int64]
    let runtimeState: Qwen38RuntimeStateSnapshot
}

private struct Qwen38FetchedMoE: @unchecked Sendable {
    let tokenIndex: Int
    let views: [TensorView]
    let offsets: MoEExpertOffsets
    let cacheHits: Int
    let cacheMisses: Int
    let readDiagnostics: ExpertReadDiagnostics
}

private final class Qwen38RunnerScratch {
    let prefillBatchCapacity: Int
    let tokenIDs: MTLBuffer
    let hiddenStreams: MTLBuffer
    let alternateStreams: MTLBuffer
    let mixedInput: MTLBuffer
    let attentionOutput: MTLBuffer
    let afterAttention: MTLBuffer
    let mlpOutput: MTLBuffer
    let finalHidden: MTLBuffer
    let ngramEmbedding: MTLBuffer
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
        let batchCapacity = min(Qwen38MoE.prefillBatchCapacity, maxContext)
        let lowRank = 320
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        let deltaKeyWidth = config.linearNumKeyHeads * config.linearKeyHeadDim
        let deltaValueWidth = config.linearNumValueHeads * config.linearValueHeadDim
        let deltaQKVWidth = deltaKeyWidth * 2 + deltaValueWidth
        let qsa = Qwen38QSAGeometry.qwen
        let compressionRatio = Int(qsa.compressRatio)
        let blockCount = max(1, maxContext / compressionRatio)

        prefillBatchCapacity = batchCapacity
        tokenIDs = try makeBuffer(batchCapacity, stride: MemoryLayout<UInt32>.stride)
        hiddenStreams = try makeBuffer(batchCapacity * hyperWidth)
        alternateStreams = try makeBuffer(batchCapacity * hyperWidth)
        mixedInput = try makeBuffer(batchCapacity * hiddenSize)
        attentionOutput = try makeBuffer(batchCapacity * hiddenSize)
        afterAttention = try makeBuffer(batchCapacity * hyperWidth)
        mlpOutput = try makeBuffer(batchCapacity * hiddenSize)
        finalHidden = try makeBuffer(batchCapacity * hiddenSize)
        ngramEmbedding = try makeBuffer(batchCapacity * pleEmbeddingSize)
        projection = try makeBuffer(batchCapacity * qWidth * 2)
        query = try makeBuffer(batchCapacity * qWidth)
        queryGate = try makeBuffer(batchCapacity * qWidth)
        key = try makeBuffer(batchCapacity * kvWidth)
        value = try makeBuffer(batchCapacity * kvWidth)
        normalizedQuery = try makeBuffer(batchCapacity * qWidth)
        normalizedKey = try makeBuffer(batchCapacity * kvWidth)
        attentionOutputHeads = try makeBuffer(batchCapacity * qWidth)
        gatedAttention = try makeBuffer(batchCapacity * qWidth)
        delta = Qwen38DeltaNetScratch(
            qkv: try makeBuffer(batchCapacity * deltaQKVWidth),
            gate: try makeBuffer(batchCapacity * deltaValueWidth),
            betaInput: try makeBuffer(batchCapacity * config.linearNumValueHeads),
            decayInput: try makeBuffer(batchCapacity * config.linearNumValueHeads),
            convolution: try makeBuffer(batchCapacity * deltaQKVWidth),
            query: try makeBuffer(batchCapacity * deltaKeyWidth),
            key: try makeBuffer(batchCapacity * deltaKeyWidth),
            value: try makeBuffer(batchCapacity * deltaValueWidth),
            decay: try makeBuffer(batchCapacity * config.linearNumValueHeads,
                                  stride: MemoryLayout<Float>.stride),
            beta: try makeBuffer(batchCapacity * config.linearNumValueHeads,
                                 stride: MemoryLayout<Float>.stride),
            recurrent: try makeBuffer(batchCapacity * deltaValueWidth),
            normalized: try makeBuffer(batchCapacity * deltaValueWidth))
        attentionHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(batchCapacity * hyperWidth),
            lowRank: try makeBuffer(batchCapacity * lowRank),
            activatedLowRank: try makeBuffer(batchCapacity * lowRank),
            mixLogits: try makeBuffer(batchCapacity * hyperWidth),
            injectionLogits: try makeBuffer(batchCapacity * streamCount),
            injectionWeights: try makeBuffer(batchCapacity * streamCount))
        mlpHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(batchCapacity * hyperWidth),
            lowRank: try makeBuffer(batchCapacity * lowRank),
            activatedLowRank: try makeBuffer(batchCapacity * lowRank),
            mixLogits: try makeBuffer(batchCapacity * hyperWidth),
            injectionLogits: try makeBuffer(batchCapacity * streamCount),
            injectionWeights: try makeBuffer(batchCapacity * streamCount))
        finalHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(batchCapacity * hyperWidth),
            lowRank: try makeBuffer(batchCapacity * lowRank),
            activatedLowRank: try makeBuffer(batchCapacity * lowRank),
            mixLogits: try makeBuffer(batchCapacity * hyperWidth),
            injectionLogits: try makeBuffer(batchCapacity * streamCount),
            injectionWeights: try makeBuffer(batchCapacity * streamCount))
        sharedOutput = try makeBuffer(batchCapacity * hiddenSize)
        qsaProjectedRows = try makeBuffer(batchCapacity * Int(qsa.projectionWidth))
        qsaBlockScores = try makeBuffer(batchCapacity * blockCount,
                                        stride: MemoryLayout<Float>.stride)
        qsaSelectionState = try makeBuffer(batchCapacity * Qwen38QSASelector.stateBytesPerQuery,
                                           stride: 1)
        qsaTokenMask = try makeBuffer(maxContext * maxContext, stride: 1)
        queryPositions = try makeBuffer(batchCapacity, stride: MemoryLayout<UInt32>.stride)
        visibleTokenCounts = try makeBuffer(batchCapacity, stride: MemoryLayout<UInt32>.stride)
        sharedGateScratch = try makeBuffer(batchCapacity * config.intermediateSize)
        sharedUpScratch = try makeBuffer(batchCapacity * config.intermediateSize)
        sharedActScratch = try makeBuffer(batchCapacity * config.intermediateSize)
        ple = Qwen38PLEScratch(
            projectedKey: try makeBuffer(batchCapacity * hyperWidth),
            value: try makeBuffer(batchCapacity * hiddenSize),
            normalizedKey: try makeBuffer(batchCapacity * hyperWidth),
            normalizedQuery: try makeBuffer(batchCapacity * hyperWidth),
            gatedValue: try makeBuffer(batchCapacity * hyperWidth),
            normalizedGatedValue: try makeBuffer(batchCapacity * hyperWidth),
            convolution: try makeBuffer(batchCapacity * hyperWidth))
    }
}

public final class Qwen38ForwardRunner: ForwardRunner, ContinuableLogitProducer,
    PromptStateSnapshotting, ChunkedPrefillRunner, @unchecked Sendable {
    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let embed: EmbedLookupInt4
    private let prefillEmbed: PrefillEmbedLookupInt4
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
    public private(set) var lastDecodeTiming: Qwen38DecodeTimingSample?
    private var ngramContext: [Int64] = []
    private var promptStateSnapshot: Qwen38PromptStateSnapshot?
    private var pendingCommandBuffer: MTLCommandBuffer?

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
        let pleAddressing = try Qwen38PLEAddressing(
            model: model,
            layer: pleLayer,
            architecture: architecture)

        self.model = model
        self.context = context
        self.config = model.config
        self.maxContext = maxContext
        self.embed = try EmbedLookupInt4(
            context: context,
            groupSize: Quantization.qwen38GroupSize)
        self.prefillEmbed = try PrefillEmbedLookupInt4(
            context: context,
            groupSize: Quantization.qwen38GroupSize)
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
                hiddenSize: model.config.hiddenSize),
            groupSize: Quantization.qwen38GroupSize)
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

        lastDecodeTiming = nil
        var embeddingNanos: UInt64 = 0
        let pleNanos: UInt64 = 0
        var attentionRouterNanos: UInt64 = 0
        var expertFetchNanos: UInt64 = 0
        var expertCacheHits = 0
        var expertCacheMisses = 0
        var moeNanos: UInt64 = 0
        var gpuActiveNanos: UInt64 = 0
        var commandBufferCount = 0
        var nextNgramContext = ngramContext
        let addresses = pleAddressing.addresses(
            tokens: [Int64(token)], context: &nextNgramContext)
        try writeNgramEmbedding(addresses: addresses[0])

        let embeddingStart = DispatchTime.now().uptimeNanoseconds
        let embeddingGPUActiveNanos = try runSync { commandBuffer in
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
        embeddingNanos = DispatchTime.now().uptimeNanoseconds - embeddingStart
        gpuActiveNanos += embeddingGPUActiveNanos
        commandBufferCount += 1

        var inputStreams = scratch.hiddenStreams
        var outputStreams = scratch.alternateStreams
        let initialFrontStart = DispatchTime.now().uptimeNanoseconds
        let initialFrontGPUActiveNanos = try runSync { commandBuffer in
            try encodeLayerInput(
                commandBuffer: commandBuffer,
                layer: 0,
                position: position,
                inputStreams: inputStreams,
                outputStreams: outputStreams)
        }
        attentionRouterNanos += DispatchTime.now().uptimeNanoseconds - initialFrontStart
        gpuActiveNanos += initialFrontGPUActiveNanos
        commandBufferCount += 1
        var finalHeadNanos: UInt64 = 0
        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            if layer > 0 {
                gpuActiveNanos += try waitPending()
            }
            let expertFetchStart = DispatchTime.now().uptimeNanoseconds
            let fetchedResult = try await moe.fetchSelectedExperts(model: model, layer: layer)
            let fetched = Qwen38FetchedMoE(
                tokenIndex: 0,
                views: fetchedResult.views,
                offsets: fetchedResult.offsets,
                cacheHits: fetchedResult.cacheHits,
                cacheMisses: fetchedResult.cacheMisses,
                readDiagnostics: fetchedResult.readDiagnostics)
            expertCacheHits += fetched.cacheHits
            expertCacheMisses += fetched.cacheMisses
            expertFetchNanos += DispatchTime.now().uptimeNanoseconds - expertFetchStart

            let moeStart = DispatchTime.now().uptimeNanoseconds
            let isLastLayer = layer == config.numLayers - 1
            let finalHeadStart = isLastLayer
                ? DispatchTime.now().uptimeNanoseconds
                : 0
            try runAsync { commandBuffer in
                try encodeLayerMoE(
                    commandBuffer: commandBuffer,
                    layer: layer,
                    inputStreams: inputStreams,
                    outputStreams: outputStreams,
                    fetched: fetched)
                if isLastLayer {
                    decoderFinalPrepare(
                        commandBuffer: commandBuffer,
                        input: outputStreams)
                    let lmHead = model.lmHead
                    head.encode(
                        commandBuffer: commandBuffer,
                        weights: lmHead.buffer,
                        weightsOffset: Int(lmHead.offset),
                        scales: lmHead.buffer,
                        scalesOffset: Int(lmHead.scaleOffset),
                        biases: lmHead.buffer,
                        biasesOffset: Int(lmHead.biasOffset),
                        hidden: scratch.mixedInput,
                        logits: logits)
                } else {
                    let nextLayer = layer + 1
                    let nextFrontStart = DispatchTime.now().uptimeNanoseconds
                    try encodeLayerInput(
                        commandBuffer: commandBuffer,
                        layer: nextLayer,
                        position: position,
                        inputStreams: outputStreams,
                        outputStreams: inputStreams)
                    attentionRouterNanos += DispatchTime.now().uptimeNanoseconds - nextFrontStart
                }
            }
            moeNanos += DispatchTime.now().uptimeNanoseconds - moeStart
            commandBufferCount += 1
            if isLastLayer {
                gpuActiveNanos += try waitPending()
                finalHeadNanos = DispatchTime.now().uptimeNanoseconds - finalHeadStart
            } else {
                swap(&inputStreams, &outputStreams)
            }
        }

        lastDecodeTiming = Qwen38DecodeTimingSample(
            embeddingNanos: embeddingNanos,
            pleNanos: pleNanos,
            attentionRouterNanos: attentionRouterNanos,
            expertFetchNanos: expertFetchNanos,
            expertCacheHits: expertCacheHits,
            expertCacheMisses: expertCacheMisses,
            moeNanos: moeNanos,
            finalHeadNanos: finalHeadNanos,
            gpuActiveNanos: gpuActiveNanos,
            commandBufferCount: commandBufferCount)
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
        guard logits.length >= config.vocabSize * MemoryLayout<Float16>.stride else {
            throw ModelError.residentBufferWrapFailed
        }
        var offset = 0
        var workCounter = PrefillWorkCounter()
        while offset < tokens.count {
            let chunkStart = tokens.index(tokens.startIndex, offsetBy: offset)
            let chunkCount = min(scratch.prefillBatchCapacity, tokens.count - offset)
            let chunkEnd = tokens.index(chunkStart, offsetBy: chunkCount)
            let work = try await produceBatch(
                tokens: tokens[chunkStart..<chunkEnd],
                startPosition: startPosition + offset,
                into: logits)
            workCounter.merge(work)
            offset += chunkCount
            onProgress(offset)
        }
        return PrefillResult(newPosition: continuationPosition,
                             seed: .logitsWritten,
                             work: workCounter.diagnostics)
    }

    private func produceBatch(tokens: ArraySlice<Int32>,
                              startPosition: Int,
                              into logits: MTLBuffer) async throws
        -> PrefillWorkDiagnostics {
        let tokenCount = tokens.count
        guard tokenCount > 0 && tokenCount <= scratch.prefillBatchCapacity else {
            throw PrefillError.prefillCursorMismatch(
                "prefill batch size \(tokenCount) exceeds runner capacity")
        }
        guard startPosition == continuationPosition else {
            throw PrefillError.prefillCursorMismatch(
                "batch start \(startPosition) != current position \(continuationPosition)")
        }
        guard startPosition >= 0,
              startPosition + tokenCount <= maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "batch \(startPosition)..<\(startPosition + tokenCount) exceeds maxContext \(maxContext)")
        }

        var workCounter = PrefillWorkCounter()
        let embeddingStart = DispatchTime.now().uptimeNanoseconds
        var nextNgramContext = ngramContext
        var addressRows: [[Int64]] = []
        addressRows.reserveCapacity(tokenCount)
        for token in tokens {
            let tokenAddresses = pleAddressing.addresses(
                tokens: [Int64(token)], context: &nextNgramContext)
            addressRows.append(contentsOf: tokenAddresses)
        }
        try writeNgramEmbeddings(addressRows: addressRows)
        let tokenPointer = scratch.tokenIDs.contents()
            .assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() {
            tokenPointer[index] = UInt32(bitPattern: token)
        }
        let positionPointer = scratch.queryPositions.contents()
            .assumingMemoryBound(to: UInt32.self)
        let visiblePointer = scratch.visibleTokenCounts.contents()
            .assumingMemoryBound(to: UInt32.self)
        for index in 0..<tokenCount {
            positionPointer[index] = UInt32(startPosition + index)
            visiblePointer[index] = UInt32(startPosition + index + 1)
        }

        try runSync { commandBuffer in
            let embedding = model.embedding
            prefillEmbed.encode(
                commandBuffer: commandBuffer,
                table: embedding.buffer,
                tableOffset: Int(embedding.offset),
                scales: embedding.buffer,
                scalesOffset: Int(embedding.scaleOffset),
                biases: embedding.buffer,
                biasesOffset: Int(embedding.biasOffset),
                tokens: scratch.tokenIDs,
                out: scratch.finalHidden,
                t: UInt32(tokenCount),
                d: UInt32(config.hiddenSize),
                outScale: 1)
            streamOps.encodeRepeatStreams(
                commandBuffer: commandBuffer,
                input: scratch.finalHidden,
                output: scratch.hiddenStreams,
                tokenCount: UInt32(tokenCount),
                streamCount: 4,
                hiddenSize: UInt32(config.hiddenSize))
        }
        workCounter.recordStageTimings(
            embedding: DispatchTime.now().uptimeNanoseconds - embeddingStart)
        workCounter.recordChunkPass()
        workCounter.recordCommandBuffers(1)

        var inputStreams = scratch.hiddenStreams
        var outputStreams = scratch.alternateStreams
        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            if layer == pleLayer {
                let pleStart = DispatchTime.now().uptimeNanoseconds
                try runSync { commandBuffer in
                    plePipeline.encode(
                        commandBuffer: commandBuffer,
                        embedding: scratch.ngramEmbedding,
                        hiddenStates: inputStreams,
                        weights: pleWeights,
                        scratch: scratch.ple,
                        state: runtimeState.pleConvolution,
                        output: outputStreams,
                        tokenCount: UInt32(tokenCount),
                        streamCount: 4,
                        hiddenSize: UInt32(config.hiddenSize),
                        embeddingSize: UInt32(config.qwen38Architecture?.pleEmbeddingSize
                            ?? config.hiddenSize),
                        epsilon: 1e-6)
                }
                workCounter.recordStageTimings(
                    mixer: DispatchTime.now().uptimeNanoseconds - pleStart)
                workCounter.recordChunkPass()
                workCounter.recordCommandBuffers(1)
                swap(&inputStreams, &outputStreams)
            }
            let layerTiming = try await encodeLayerBatch(
                layer: layer,
                startPosition: startPosition,
                inputStreams: inputStreams,
                outputStreams: outputStreams,
                tokenCount: UInt32(tokenCount))
            workCounter.recordStageTimings(
                mixer: layerTiming.mixerNanos,
                expertFetch: layerTiming.expertFetchNanos,
                routedMoE: layerTiming.routedMoENanos)
            workCounter.recordExpertReads(
                cacheHits: layerTiming.cacheHits,
                cacheMisses: layerTiming.cacheMisses,
                estimatedBytes: 0,
                readCount: layerTiming.expertReadCount,
                readNanos: layerTiming.expertReadNanos,
                readMaxNanos: layerTiming.expertReadMaxNanos)
            workCounter.recordChunkPass()
            workCounter.recordCommandBuffers(2)
            swap(&inputStreams, &outputStreams)
        }
        let finalHeadStart = DispatchTime.now().uptimeNanoseconds
        try runSync { commandBuffer in
            decoderFinalPrepare(
                commandBuffer: commandBuffer,
                input: inputStreams,
                tokenCount: UInt32(tokenCount))
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            let rowBytes = config.hiddenSize * MemoryLayout<Float16>.stride
            blit.copy(
                from: scratch.mixedInput,
                sourceOffset: (tokenCount - 1) * rowBytes,
                to: scratch.finalHidden,
                destinationOffset: 0,
                size: rowBytes)
            blit.endEncoding()
            let lmHead = model.lmHead
            head.encode(
                commandBuffer: commandBuffer,
                weights: lmHead.buffer,
                weightsOffset: Int(lmHead.offset),
                scales: lmHead.buffer,
                scalesOffset: Int(lmHead.scaleOffset),
                biases: lmHead.buffer,
                biasesOffset: Int(lmHead.biasOffset),
                hidden: scratch.finalHidden,
                logits: logits)
        }
        workCounter.recordStageTimings(
            finalHead: DispatchTime.now().uptimeNanoseconds - finalHeadStart)
        workCounter.recordChunkPass()
        workCounter.recordCommandBuffers(1)
        ngramContext = nextNgramContext
        continuationPosition += tokenCount
        guard let diagnostics = workCounter.diagnostics else {
            throw PrefillError.chunkedUnsupported("Qwen3.8 prefill produced no work diagnostics")
        }
        return diagnostics
    }

    private func encodeLayerBatch(layer: Int,
                                  startPosition: Int,
                                  inputStreams: MTLBuffer,
                                  outputStreams: MTLBuffer,
                                  tokenCount: UInt32) async throws
        -> Qwen38LayerPrefillTiming {
        let layerScratch = Qwen38DecoderLayerScratch(
            attentionHyperConnection: scratch.attentionHyperConnection,
            attentionInput: scratch.mixedInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
        let mixerStart = DispatchTime.now().uptimeNanoseconds
        try runSync { commandBuffer in
            decoder.encodeAttentionPrepare(
                commandBuffer: commandBuffer,
                weights: layers[layer],
                hyperInput: inputStreams,
                scratch: layerScratch,
                tokenCount: tokenCount,
                epsilon: 1e-6)
            let state = try decoder.attentionState(
                layer: layer, runtimeState: runtimeState)
            try encodeAttentionBatch(
                commandBuffer: commandBuffer,
                layer: layer,
                state: state,
                input: scratch.mixedInput,
                output: scratch.attentionOutput,
                startPosition: startPosition,
                tokenCount: tokenCount)
            decoder.encodeAttentionInject(
                commandBuffer: commandBuffer,
                hyperInput: inputStreams,
                scratch: layerScratch,
                tokenCount: tokenCount)
            decoder.encodeMLPPrepare(
                commandBuffer: commandBuffer,
                weights: layers[layer],
                scratch: layerScratch,
                tokenCount: tokenCount,
                epsilon: 1e-6)
            for tokenIndex in 0..<Int(tokenCount) {
                moe.encodeRouter(
                    commandBuffer: commandBuffer,
                    weights: moeWeights[layer],
                    hidden: scratch.mixedInput,
                    hiddenSize: UInt32(config.hiddenSize),
                    tokenIndex: tokenIndex)
                moe.encodeSelection(
                    commandBuffer: commandBuffer,
                    weights: moeWeights[layer],
                    tokenIndex: tokenIndex)
            }
        }
        let mixerNanos = DispatchTime.now().uptimeNanoseconds - mixerStart
        var expertFetchNanos: UInt64 = 0
        var cacheHits = 0
        var cacheMisses = 0
        var expertReadCount = 0
        var expertReadNanos: UInt64 = 0
        var expertReadMaxNanos: UInt64 = 0
        var fetched: [Qwen38FetchedMoE] = []
        fetched.reserveCapacity(Int(tokenCount))
        let expertFetchStart = DispatchTime.now().uptimeNanoseconds
        var plans: [RoutedExpertFetchPlan] = []
        plans.reserveCapacity(Int(tokenCount))
        for tokenIndex in 0..<Int(tokenCount) {
            plans.append(try moe.planSelectedExperts(
                model: model,
                layer: layer,
                tokenIndex: tokenIndex))
        }
        let results = try await model.fetchRoutedExpertsWithDiagnostics(plans: plans)
        for tokenIndex in plans.indices {
            let plan = plans[tokenIndex]
            let result = results[tokenIndex]
            fetched.append(Qwen38FetchedMoE(
                tokenIndex: tokenIndex,
                views: result.views,
                offsets: model.routedExpertOffsets(layer: layer),
                cacheHits: plan.hits,
                cacheMisses: plan.misses.count,
                readDiagnostics: result.readDiagnostics))
            cacheHits += plan.hits
            cacheMisses += plan.misses.count
            expertReadCount += result.readDiagnostics.readCount
            expertReadNanos += result.readDiagnostics.totalNanos
            expertReadMaxNanos = max(expertReadMaxNanos, result.readDiagnostics.maxNanos)
        }
        expertFetchNanos = DispatchTime.now().uptimeNanoseconds - expertFetchStart

        let routedMoEStart = DispatchTime.now().uptimeNanoseconds
        try runSync { commandBuffer in
            try sharedExpert.encodeBlock(
                commandBuffer: commandBuffer,
                x: scratch.mixedInput,
                y: scratch.sharedOutput,
                gate: sharedProjection(moeWeights[layer].sharedExpertGate,
                                       rows: config.intermediateSize,
                                       cols: config.hiddenSize),
                up: sharedProjection(moeWeights[layer].sharedExpertUp,
                                     rows: config.intermediateSize,
                                     cols: config.hiddenSize),
                down: sharedProjection(moeWeights[layer].sharedExpertDown,
                                       rows: config.hiddenSize,
                                       cols: config.intermediateSize),
                scratchGate: scratch.sharedGateScratch,
                scratchUp: scratch.sharedUpScratch,
                scratchAct: scratch.sharedActScratch,
                queryCount: Int(tokenCount),
                d: config.hiddenSize,
                intermediate: config.moeIntermediateSize,
                xStrideElements: config.hiddenSize,
                yStrideElements: config.hiddenSize)
            for result in fetched {
                let routedArguments = try moe.makeRoutedArgumentBuffer(
                    layer: layer,
                    slot: result.tokenIndex,
                    experts: result.views)
                moe.encodeRouted(
                    commandBuffer: commandBuffer,
                    routedArgumentBuffer: routedArguments,
                    routedOffsets: result.offsets,
                    input: scratch.mixedInput,
                    residual: scratch.sharedOutput,
                    output: scratch.mlpOutput,
                    hiddenSize: UInt32(config.hiddenSize),
                    intermediateSize: UInt32(config.moeIntermediateSize),
                    sharedExpertGateWeight: moeWeights[layer].sharedExpertGateWeight,
                    tokenIndex: result.tokenIndex)
            }
            decoder.encodeMLPInject(
                commandBuffer: commandBuffer,
                scratch: layerScratch,
                output: outputStreams,
                tokenCount: tokenCount)
        }
        return Qwen38LayerPrefillTiming(
            mixerNanos: mixerNanos,
            expertFetchNanos: expertFetchNanos,
            routedMoENanos: DispatchTime.now().uptimeNanoseconds - routedMoEStart,
            cacheHits: cacheHits,
            cacheMisses: cacheMisses,
            expertReadCount: expertReadCount,
            expertReadNanos: expertReadNanos,
            expertReadMaxNanos: expertReadMaxNanos)
    }

    private func encodeLayerInput(commandBuffer: MTLCommandBuffer,
                                  layer: Int,
                                  position: Int,
                                  inputStreams: MTLBuffer,
                                  outputStreams: MTLBuffer) throws {
        if layer == pleLayer {
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
                embeddingSize: UInt32(config.qwen38Architecture?.pleEmbeddingSize
                    ?? config.hiddenSize),
                epsilon: 1e-6)
            try encodeLayerFront(
                commandBuffer: commandBuffer,
                layer: layer,
                position: position,
                inputStreams: outputStreams,
                outputStreams: inputStreams)
        } else {
            try encodeLayerFront(
                commandBuffer: commandBuffer,
                layer: layer,
                position: position,
                inputStreams: inputStreams,
                outputStreams: outputStreams)
        }
    }

    private func encodeLayerFront(commandBuffer: MTLCommandBuffer,
                                  layer: Int,
                                  position: Int,
                                  inputStreams: MTLBuffer,
                                  outputStreams: MTLBuffer) throws {
        let layerScratch = Qwen38DecoderLayerScratch(
            attentionHyperConnection: scratch.attentionHyperConnection,
            attentionInput: scratch.mixedInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
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

    private func encodeLayerMoE(commandBuffer: MTLCommandBuffer,
                                layer: Int,
                                inputStreams: MTLBuffer,
                                outputStreams: MTLBuffer,
                                fetched: Qwen38FetchedMoE) throws {
        let layerScratch = Qwen38DecoderLayerScratch(
            attentionHyperConnection: scratch.attentionHyperConnection,
            attentionInput: scratch.mixedInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
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
            layer: layer,
            slot: 0,
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

    private func encodeAttentionBatch(commandBuffer: MTLCommandBuffer,
                                      layer: Int,
                                      state: Qwen38DecoderAttentionState,
                                      input: MTLBuffer,
                                      output: MTLBuffer,
                                      startPosition: Int,
                                      tokenCount: UInt32) throws {
        switch state {
        case .linear:
            let weights = try Qwen38DeltaNetWeights(model: model, layer: layer)
            try deltaNet.encodeBatch(
                commandBuffer: commandBuffer,
                state: state,
                weights: weights,
                input: input,
                scratch: scratch.delta,
                output: output,
                tokenCount: tokenCount,
                epsilon: 1e-6)
        case .sparse(let qsaState, let cache):
            guard qsaState.rawKeyCache.count == startPosition else {
                throw PrefillError.prefillCursorMismatch(
                    "QSA cache \(qsaState.rawKeyCache.count) != prefill start \(startPosition)")
            }
            guard cache.count == startPosition else {
                throw PrefillError.prefillCursorMismatch(
                    "full-attention cache \(cache.count) != prefill start \(startPosition)")
            }
            let weights = try model.qwenFullAttentionWeights(layer: layer)
            qsa.encode(
                commandBuffer: commandBuffer,
                state: qsaState,
                hiddenStates: input,
                queryPositions: scratch.queryPositions,
                visibleTokenCounts: scratch.visibleTokenCounts,
                scratch: Qwen38QSALayerExecutionScratch(
                    projectedRows: scratch.qsaProjectedRows,
                    blockScores: scratch.qsaBlockScores,
                    selection: Qwen38QSASelectionScratch(
                        state: scratch.qsaSelectionState,
                        tokenMask: scratch.qsaTokenMask)),
                tokenCount: tokenCount,
                inputWidth: UInt32(config.hiddenSize),
                epsilon: 1e-6)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.q,
                input: input,
                output: scratch.projection,
                tokenCount: tokenCount,
                outputWidth: config.numHeads * config.fullHeadDim * 2)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.k,
                input: input,
                output: scratch.key,
                tokenCount: tokenCount,
                outputWidth: config.numFullKVHeads * config.fullHeadDim)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.v,
                input: input,
                output: scratch.value,
                tokenCount: tokenCount,
                outputWidth: config.numFullKVHeads * config.fullHeadDim)
            attention.encodeSplitQueryGateBatch(
                commandBuffer: commandBuffer,
                projection: scratch.projection,
                query: scratch.query,
                gate: scratch.queryGate,
                tokenCount: tokenCount)
            attention.encodeQueryKeyBatch(
                commandBuffer: commandBuffer,
                query: scratch.query,
                key: scratch.key,
                queryNorm: weights.qNorm.buffer,
                queryNormOffset: Int(weights.qNorm.offset),
                keyNorm: weights.kNorm.buffer,
                keyNormOffset: Int(weights.kNorm.offset),
                normalizedQuery: scratch.normalizedQuery,
                normalizedKey: scratch.normalizedKey,
                position: UInt32(startPosition),
                tokenCount: tokenCount,
                epsilon: 1e-6)
            cache.appendBatch(
                commandBuffer: commandBuffer,
                key: scratch.normalizedKey,
                value: scratch.value,
                tokenCount: Int(tokenCount))
            attention.encodeBatch(
                commandBuffer: commandBuffer,
                query: scratch.normalizedQuery,
                cache: cache,
                output: scratch.attentionOutputHeads,
                startPosition: UInt32(startPosition),
                tokenCount: tokenCount,
                tokenMask: scratch.qsaTokenMask)
            attention.encodeOutputGateBatch(
                commandBuffer: commandBuffer,
                attention: scratch.attentionOutputHeads,
                gate: scratch.queryGate,
                output: scratch.gatedAttention,
                tokenCount: tokenCount)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.o,
                input: scratch.gatedAttention,
                output: output,
                tokenCount: tokenCount,
                outputWidth: config.hiddenSize,
                inputWidth: config.numHeads * config.fullHeadDim)
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
            attention.encodeBatch(
                commandBuffer: commandBuffer,
                query: scratch.normalizedQuery,
                cache: cache,
                output: scratch.attentionOutputHeads,
                startPosition: UInt32(position),
                tokenCount: 1,
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

    private func writeNgramEmbeddings(addressRows: [[Int64]]) throws {
        guard !addressRows.isEmpty else {
            throw ModelError.indexCorrupt(detail: "Qwen3.8 PLE produced no n-gram addresses")
        }
        let embeddingSize = config.qwen38Architecture?.pleEmbeddingSize ?? 0
        let output = scratch.ngramEmbedding.contents().assumingMemoryBound(to: UInt16.self)
        for (tokenIndex, addresses) in addressRows.enumerated() {
            guard !addresses.isEmpty else {
                throw ModelError.indexCorrupt(
                    detail: "Qwen3.8 PLE produced no n-gram addresses")
            }
            let rows = try ngramStreamer.read(addresses: addresses)
            guard rows.count == addresses.count,
                  embeddingSize.isMultiple(of: rows.count),
                  rows.allSatisfy({ $0.count == embeddingSize / rows.count }) else {
                throw ModelError.archMismatch(
                    field: "packedNgrams.rowWidth",
                    expected: "pleEmbeddingSize / addressCount",
                    actual: "\(rows.first?.count ?? 0)")
            }
            let headWidth = embeddingSize / rows.count
            let tokenOffset = tokenIndex * embeddingSize
            for (head, row) in rows.enumerated() {
                for (index, value) in row.enumerated() {
                    output[tokenOffset + head * headWidth + index] =
                        Float16(value).bitPattern
                }
            }
        }
    }

    private func decoderFinalPrepare(commandBuffer: MTLCommandBuffer,
                                     input: MTLBuffer,
                                     tokenCount: UInt32 = 1) {
        finalMixer.encodePrepare(
            commandBuffer: commandBuffer,
            hyperInput: input,
            weights: finalHyperConnection,
            scratch: scratch.finalHyperConnection,
            mixedInput: scratch.mixedInput,
            tokenCount: tokenCount,
            epsilon: 1e-6)
    }

    private func encodeProjection(commandBuffer: MTLCommandBuffer,
                                  weights: TensorView,
                                  input: MTLBuffer,
                                  output: MTLBuffer,
                                  tokenCount: UInt32 = 1,
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
            tokenCount: tokenCount,
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

    private func runAsync(_ body: (MTLCommandBuffer) throws -> Void) throws {
        guard pendingCommandBuffer == nil else {
            throw ModelError.residentBufferWrapFailed
        }
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try body(commandBuffer)
        commandBuffer.commit()
        pendingCommandBuffer = commandBuffer
    }

    private func waitPending() throws -> UInt64 {
        guard let pending = pendingCommandBuffer else { return 0 }
        pendingCommandBuffer = nil
        pending.waitUntilCompleted()
        try checkCompleted(pending)
        return gpuDurationNanos(pending)
    }

    @discardableResult
    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws -> UInt64 {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try body(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var gpuActiveNanos: UInt64 = 0
        if let pending = pendingCommandBuffer {
            pendingCommandBuffer = nil
            try checkCompleted(pending)
            gpuActiveNanos += gpuDurationNanos(pending)
        }
        try checkCompleted(commandBuffer)
        gpuActiveNanos += gpuDurationNanos(commandBuffer)
        return gpuActiveNanos
    }

    private func gpuDurationNanos(_ commandBuffer: MTLCommandBuffer) -> UInt64 {
        let duration = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        guard duration.isFinite, duration > 0 else { return 0 }
        return UInt64(duration * 1_000_000_000)
    }

    private func checkCompleted(_ commandBuffer: MTLCommandBuffer) throws {
        if let error = commandBuffer.error {
            throw error
        }
        guard commandBuffer.status == .completed else {
            throw ModelError.residentBufferWrapFailed
        }
    }
}
