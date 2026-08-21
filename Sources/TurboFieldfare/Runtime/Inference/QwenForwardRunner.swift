import Foundation
import Metal

/// Decode runner for the text-only Qwen3.6 MoE model.
///
/// The runner intentionally owns the two different forms of causal state:
/// Gated DeltaNet state for linear-attention layers and an FP16 KV cache for
/// every fourth full-attention layer. It does not share Gemma's fused layer
/// tail because Qwen's residual block and untied output head are different.
public final class QwenForwardRunner: ChunkedPrefillRunner, ContinuableLogitProducer,
    ContextWindowReporting, ForwardRunner, @unchecked Sendable {
    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let embed: EmbedLookupInt4
    private let rms: RMSNorm
    private let gemv: DequantInt4GEMV
    private let deltaNet: QwenGatedDeltaNet
    private let deltaElementwise: QwenElementwise
    private let attention: QwenFullAttention
    private let moe: QwenMoE
    private let head: QwenUntiedLMHead
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let deltaStates: QwenGatedDeltaNetStateManager
    private let fullCaches: [QwenFullAttentionKVCache?]
    private let prefillScratchCache = QwenPrefillScratchCache()

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let mixerOutput: MTLBuffer
    private let projection: MTLBuffer
    private let query: MTLBuffer
    private let queryGate: MTLBuffer
    private let key: MTLBuffer
    private let value: MTLBuffer
    private let normalizedQuery: MTLBuffer
    private let normalizedKey: MTLBuffer
    private let attentionOutput: MTLBuffer
    private let gatedAttention: MTLBuffer
    private let qkv: MTLBuffer
    private let deltaZ: MTLBuffer
    private let deltaA: MTLBuffer
    private let deltaB: MTLBuffer
    private let deltaQuery: MTLBuffer
    private let deltaKey: MTLBuffer
    private let deltaValue: MTLBuffer
    private let deltaConvolution: MTLBuffer
    private let deltaDecay: MTLBuffer
    private let deltaBeta: MTLBuffer
    private let deltaOutput: MTLBuffer
    private let sharedOutput: MTLBuffer
    private let routedOutput: MTLBuffer
    private let combinedOutput: MTLBuffer
    private let moeActs: MTLBuffer
    private let sharedGateScratch: MTLBuffer
    private let sharedUpScratch: MTLBuffer
    private let sharedActScratch: MTLBuffer
    private let routeIndices: MTLBuffer
    private let routeWeights: MTLBuffer

    public let maxContext: Int
    private var position = 0
    private var commandBufferSubmissionCount = 0

    public init(model: Model,
                context: MetalContext,
                maxContext: Int) throws {
        guard model.config.modelFamily == .qwen36MoeText else {
            throw ModelError.archMismatch(field: "modelFamily",
                                           expected: "qwen36MoeText",
                                           actual: "\(model.config.modelFamily)")
        }
        guard maxContext > 0 else {
            throw ModelError.archMismatch(field: "maxContext",
                                           expected: "positive",
                                           actual: "\(maxContext)")
        }
        self.model = model
        self.context = context
        self.config = model.config
        self.maxContext = maxContext
        self.embed = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.gemv = try DequantInt4GEMV(context: context)
        self.deltaNet = try QwenGatedDeltaNet(context: context)
        self.deltaElementwise = try QwenElementwise(context: context)
        self.attention = try QwenFullAttention(context: context)
        self.moe = try QwenMoE(context: context)
        self.head = try QwenUntiedLMHead(
            context: context,
            geometry: QwenLMHeadGeometry(vocabularySize: config.vocabSize,
                                          hiddenSize: config.hiddenSize))
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.deltaStates = try QwenGatedDeltaNetStateManager(
            context: context,
            layerCount: config.numLayers,
            geometry: QwenGatedDeltaNetGeometry(
                keyHeads: config.linearNumKeyHeads,
                valueHeads: config.linearNumValueHeads,
                keyHeadDim: config.linearKeyHeadDim,
                valueHeadDim: config.linearValueHeadDim,
                convolutionKernel: config.linearConvKernelDim),
            convolutionChannels: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                + config.linearNumValueHeads * config.linearValueHeadDim)

        let device = context.device
        func makeBuffer(_ elements: Int,
                        stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }

        let hiddenSize = config.hiddenSize
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        let deltaKeyWidth = config.linearNumKeyHeads * config.linearKeyHeadDim
        let deltaValueWidth = config.linearNumValueHeads * config.linearValueHeadDim
        let deltaQKVWidth = deltaKeyWidth * 2 + deltaValueWidth
        let sharedWidth = config.intermediateSize

        self.hidden = try makeBuffer(hiddenSize)
        self.normed = try makeBuffer(hiddenSize)
        self.mixerOutput = try makeBuffer(hiddenSize)
        self.projection = try makeBuffer(max(qWidth * 2, deltaQKVWidth))
        self.query = try makeBuffer(max(qWidth, deltaKeyWidth))
        self.queryGate = try makeBuffer(qWidth)
        self.key = try makeBuffer(max(kvWidth, deltaKeyWidth))
        self.value = try makeBuffer(max(kvWidth, deltaValueWidth))
        self.normalizedQuery = try makeBuffer(qWidth)
        self.normalizedKey = try makeBuffer(kvWidth)
        self.attentionOutput = try makeBuffer(qWidth)
        self.gatedAttention = try makeBuffer(qWidth)
        self.qkv = try makeBuffer(deltaQKVWidth)
        self.deltaZ = try makeBuffer(deltaValueWidth)
        self.deltaA = try makeBuffer(config.linearNumValueHeads,
                                     stride: MemoryLayout<Float16>.stride)
        self.deltaB = try makeBuffer(config.linearNumValueHeads,
                                     stride: MemoryLayout<Float16>.stride)
        self.deltaQuery = try makeBuffer(deltaKeyWidth)
        self.deltaKey = try makeBuffer(deltaKeyWidth)
        self.deltaValue = try makeBuffer(deltaValueWidth)
        self.deltaConvolution = try makeBuffer(deltaQKVWidth)
        self.deltaDecay = try makeBuffer(config.linearNumValueHeads,
                                         stride: MemoryLayout<Float>.stride)
        self.deltaBeta = try makeBuffer(config.linearNumValueHeads,
                                        stride: MemoryLayout<Float>.stride)
        self.deltaOutput = try makeBuffer(deltaValueWidth)
        self.sharedOutput = try makeBuffer(hiddenSize)
        self.routedOutput = try makeBuffer(hiddenSize)
        self.combinedOutput = try makeBuffer(hiddenSize)
        self.zeroBuffer = try makeBuffer(hiddenSize)
        memset(self.zeroBuffer.contents(), 0, self.zeroBuffer.length)
        self.moeActs = try makeBuffer(config.topKExperts * config.moeIntermediateSize)
        self.sharedGateScratch = try makeBuffer(sharedWidth)
        self.sharedUpScratch = try makeBuffer(sharedWidth)
        self.sharedActScratch = try makeBuffer(sharedWidth)
        self.routeIndices = try makeBuffer(config.topKExperts,
                                           stride: MemoryLayout<UInt32>.stride)
        self.routeWeights = try makeBuffer(config.topKExperts)

        var caches = Array<QwenFullAttentionKVCache?>(
            repeating: nil,
            count: config.numLayers)
        for layer in 0..<config.numLayers where config.fullAttentionLayerMask[layer] != 0 {
            caches[layer] = try QwenFullAttentionKVCache(
                device: device,
                capacity: maxContext,
                geometry: QwenFullAttentionGeometry(
                    queryHeads: config.numHeads,
                    keyValueHeads: config.numFullKVHeads,
                    headDimension: config.fullHeadDim,
                    rotaryDimension: Int(Double(config.fullHeadDim)
                        * config.partialRotaryFactor),
                    ropeTheta: Float(config.fullRopeTheta)))
        }
        self.fullCaches = caches

        for layer in 0..<config.numLayers {
            _ = try model.inputNorm(layer: layer)
            _ = try model.postAttnNorm(layer: layer)
            _ = try model.qwenMoEWeights(layer: layer)
            if config.fullAttentionLayerMask[layer] != 0 {
                _ = try model.qwenFullAttentionWeights(layer: layer)
            } else {
                _ = try model.qwenDeltaNetWeights(layer: layer)
            }
        }
        _ = model.finalNorm
        _ = model.lmHead
    }

    public func reset() {
        position = 0
        deltaStates.reset()
        for cache in fullCaches {
            cache?.reset()
        }
    }

    public var continuationPosition: Int { position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, expectedPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected position \(expectedPosition), current \(position)")
        }
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode _: PrefillOutputMode,
                               config runtimeConfig: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard startPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "prefill start \(startPosition) != current position \(position)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: position, seed: .logitsWritten)
        }
        let scratchLayout = QwenPrefillScratchLayout(config: config, runtime: runtimeConfig)
        guard tokens.count <= scratchLayout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "Qwen prefill token count \(tokens.count) exceeds chunk size \(scratchLayout.chunkTokens)")
        }
        let scratch = try prefillScratchCache.buffers(
            device: context.device,
            layout: scratchLayout)
        let tokenIDs = scratch.tokenIDs.contents().assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() {
            tokenIDs[index] = UInt32(bitPattern: token)
        }

        let commandBufferStart = commandBufferSubmissionCount
        var workCounter = PrefillWorkCounter()
        let embedding = model.embedding
        try runSync { commandBuffer in
            prefillEmbed.encode(commandBuffer: commandBuffer,
                                table: embedding.buffer,
                                tableOffset: Int(embedding.offset),
                                scales: embedding.buffer,
                                scalesOffset: Int(embedding.scaleOffset),
                                biases: embedding.buffer,
                                biasesOffset: Int(embedding.biasOffset),
                                tokens: scratch.tokenIDs,
                                out: scratch.hidden,
                                t: UInt32(tokens.count),
                                d: UInt32(config.hiddenSize),
                                outScale: 1)
        }
        workCounter.recordChunkPass()

        for index in tokens.indices {
            try Task.checkCancellation()
            let row = tokens.distance(from: tokens.startIndex, to: index)
            try runSync { commandBuffer in
                copy(commandBuffer: commandBuffer,
                     source: scratch.hidden,
                     sourceOffset: row * config.hiddenSize * MemoryLayout<Float16>.stride,
                     destination: hidden,
                     size: config.hiddenSize * MemoryLayout<Float16>.stride)
            }
            try await finishCurrentToken(into: logits, emitHead: row == tokens.count - 1)
            workCounter.recordScalarForward()
            onProgress(position)
        }
        workCounter.recordCommandBuffers(commandBufferSubmissionCount - commandBufferStart)
        return PrefillResult(newPosition: position,
                             seed: .logitsWritten,
                             work: workCounter.diagnostics)
    }

    public func produce(token: Int32,
                        position requestedPosition: Int,
                        into logits: MTLBuffer) async throws {
        try Task.checkCancellation()
        guard requestedPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(position) != position \(requestedPosition)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }

        let embedding = model.embedding
        try runSync { commandBuffer in
            embed.encode(commandBuffer: commandBuffer,
                         table: embedding.buffer,
                         tableOffset: Int(embedding.offset),
                         scales: embedding.buffer,
                         scalesOffset: Int(embedding.scaleOffset),
                         biases: embedding.buffer,
                         biasesOffset: Int(embedding.biasOffset),
                         out: hidden,
                         tokenId: UInt32(bitPattern: token),
                         d: UInt32(config.hiddenSize),
                         outScale: 1)
        }

        try await finishCurrentToken(into: logits, emitHead: true)
    }

    private func finishCurrentToken(into logits: MTLBuffer,
                                    emitHead: Bool) async throws {

        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            try await encodeLayer(layer: layer)
        }

        if emitHead {
            let finalNorm = model.finalNorm
            let lmHead = model.lmHead
            try runSync { commandBuffer in
                rms.encodeBF16W(commandBuffer: commandBuffer,
                                x: hidden,
                                weight: finalNorm.buffer,
                                weightOffset: Int(finalNorm.offset),
                                out: normed,
                                d: UInt32(config.hiddenSize),
                                eps: 1e-6)
                head.encode(commandBuffer: commandBuffer,
                            weights: lmHead.buffer,
                            weightsOffset: Int(lmHead.offset),
                            scales: lmHead.buffer,
                            scalesOffset: Int(lmHead.scaleOffset),
                            biases: lmHead.buffer,
                            biasesOffset: Int(lmHead.biasOffset),
                            hidden: normed,
                            logits: logits)
            }
            let values = logits.contents().assumingMemoryBound(to: Float16.self)
            var bestIndex = 0
            var bestValue = values[0]
            for index in 1..<config.vocabSize where values[index] > bestValue {
                bestIndex = index
                bestValue = values[index]
            }
            lastGreedyToken = UInt32(bestIndex)
        }
        position += 1
    }

    public let usesFusedGreedyHead = false
    public private(set) var lastGreedyToken: UInt32 = 0

    private func encodeLayer(layer: Int) async throws {
        let inputNorm = try model.inputNorm(layer: layer)
        let postAttentionNorm = try model.postAttnNorm(layer: layer)
        let isFull = config.fullAttentionLayerMask[layer] != 0

        try runSync { commandBuffer in
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            weight: inputNorm.buffer,
                            weightOffset: Int(inputNorm.offset),
                            out: normed,
                            d: UInt32(config.hiddenSize),
                            eps: 1e-6)
            if isFull {
                try encodeFullAttention(commandBuffer: commandBuffer, layer: layer)
            } else {
                try encodeDeltaNet(commandBuffer: commandBuffer, layer: layer)
            }
            deltaElementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                               lhs: hidden,
                                               rhs: mixerOutput,
                                               output: hidden,
                                               count: UInt32(config.hiddenSize))
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            weight: postAttentionNorm.buffer,
                            weightOffset: Int(postAttentionNorm.offset),
                            out: normed,
                            d: UInt32(config.hiddenSize),
                            eps: 1e-6)
        }

        let moeWeights = try model.qwenMoEWeights(layer: layer)
        let router = moeWeights.router
        try runSync { commandBuffer in
            let sharedGate = sharedProjection(moeWeights.sharedExpertGate,
                                               rows: config.intermediateSize,
                                               cols: config.hiddenSize)
            let sharedUp = sharedProjection(moeWeights.sharedExpertUp,
                                             rows: config.intermediateSize,
                                             cols: config.hiddenSize)
            let sharedDown = sharedProjection(moeWeights.sharedExpertDown,
                                               rows: config.hiddenSize,
                                               cols: config.intermediateSize)
            try moe.encodeSharedExpert(commandBuffer: commandBuffer,
                                       x: normed,
                                       gate: sharedGate,
                                       up: sharedUp,
                                       down: sharedDown,
                                       y: sharedOutput,
                                       scratchGate: sharedGateScratch,
                                       scratchUp: sharedUpScratch,
                                       scratchAct: sharedActScratch)
            moe.encodeRouter(commandBuffer: commandBuffer,
                             weights: router.buffer,
                             weightsOffset: Int(router.offset),
                             scales: router.buffer,
                             scalesOffset: Int(router.scaleOffset),
                             biases: router.buffer,
                             biasesOffset: Int(router.biasOffset),
                             hidden: normed,
                             outIndices: routeIndices,
                             outWeights: routeWeights,
                             numExperts: UInt32(config.numExperts),
                             d: UInt32(config.hiddenSize))
        }

        let indices = routeIndices.contents().assumingMemoryBound(to: UInt32.self)
        let experts = (0..<config.topKExperts).map { Int(indices[$0]) }
        guard let plan = try model.planRoutedExperts(layer: layer, experts: experts) else {
            throw ModelError.indexCorrupt(detail: "Qwen layer \(layer) has no expert plan")
        }
        let expertViews = try await model.fetchRoutedExperts(plan: plan)
        guard let argumentBuffer = moe.makeRoutedArgumentBuffer(
            routedBlobs: expertViews.map(\.buffer)) else {
            throw ModelError.residentBufferWrapFailed
        }
        let expertBuffers = expertViews.map(\.buffer)
        let offsets = model.routedExpertOffsets(layer: layer)
        let sharedGate = sharedProjection(moeWeights.sharedRouterGate,
                                           rows: 1,
                                           cols: config.hiddenSize)
        try runSync { commandBuffer in
            moe.encodeRoutedPhase1(commandBuffer: commandBuffer,
                                   routedArgBuffer: argumentBuffer,
                                   routedBlobs: expertBuffers,
                                   routedOffsets: offsets,
                                   x: normed,
                                   acts: moeActs,
                                   d: UInt32(config.hiddenSize),
                                   f: UInt32(config.moeIntermediateSize))
            moe.encodeRoutedPhase2(commandBuffer: commandBuffer,
                                   routedArgBuffer: argumentBuffer,
                                   routedBlobs: expertBuffers,
                                   routedOffsets: offsets,
                                   acts: moeActs,
                                   routingWeights: routeWeights,
                                   residual: zeroBuffer,
                                   y: routedOutput,
                                   d: UInt32(config.hiddenSize),
                                   f: UInt32(config.moeIntermediateSize))
            moe.encodeSharedGateAndCombine(commandBuffer: commandBuffer,
                                           x: normed,
                                           gate: sharedGate,
                                           sharedOutput: sharedOutput,
                                           routedOutput: routedOutput,
                                           y: combinedOutput,
                                           d: UInt32(config.hiddenSize))
            deltaElementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                               lhs: hidden,
                                               rhs: combinedOutput,
                                               output: hidden,
                                               count: UInt32(config.hiddenSize))
        }
    }

    private let zeroBuffer: MTLBuffer

    private func encodeDeltaNet(commandBuffer: MTLCommandBuffer,
                                layer: Int) throws {
        let weights = try model.qwenDeltaNetWeights(layer: layer)
        let deltaWidth = config.linearNumKeyHeads * config.linearKeyHeadDim * 2
            + config.linearNumValueHeads * config.linearValueHeadDim
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.qkv.buffer,
                    weightsOffset: Int(weights.qkv.offset),
                    scales: weights.qkv.buffer,
                    scalesOffset: Int(weights.qkv.scaleOffset),
                    biases: weights.qkv.buffer,
                    biasesOffset: Int(weights.qkv.biasOffset),
                    x: normed,
                    y: qkv,
                    m: UInt32(deltaWidth),
                    n: UInt32(config.hiddenSize))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.z.buffer,
                    weightsOffset: Int(weights.z.offset),
                    scales: weights.z.buffer,
                    scalesOffset: Int(weights.z.scaleOffset),
                    biases: weights.z.buffer,
                    biasesOffset: Int(weights.z.biasOffset),
                    x: normed,
                    y: deltaZ,
                    m: UInt32(config.linearNumValueHeads * config.linearValueHeadDim),
                    n: UInt32(config.hiddenSize))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.b.buffer,
                    weightsOffset: Int(weights.b.offset),
                    scales: weights.b.buffer,
                    scalesOffset: Int(weights.b.scaleOffset),
                    biases: weights.b.buffer,
                    biasesOffset: Int(weights.b.biasOffset),
                    x: normed,
                    y: deltaB,
                    m: UInt32(config.linearNumValueHeads),
                    n: UInt32(config.hiddenSize))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.a.buffer,
                    weightsOffset: Int(weights.a.offset),
                    scales: weights.a.buffer,
                    scalesOffset: Int(weights.a.scaleOffset),
                    biases: weights.a.buffer,
                    biasesOffset: Int(weights.a.biasOffset),
                    x: normed,
                    y: deltaA,
                    m: UInt32(config.linearNumValueHeads),
                    n: UInt32(config.hiddenSize))
        deltaNet.encodeCausalConvolution(commandBuffer: commandBuffer,
                                         input: qkv,
                                         weights: weights.convolution.buffer,
                                         weightsOffset: Int(weights.convolution.offset),
                                         output: deltaConvolution,
                                         state: deltaStates.state(layer: layer))
        copy(commandBuffer: commandBuffer,
             source: deltaConvolution,
             sourceOffset: 0,
             destination: deltaQuery,
             size: config.linearNumKeyHeads * config.linearKeyHeadDim
                * MemoryLayout<Float16>.stride)
        copy(commandBuffer: commandBuffer,
             source: deltaConvolution,
             sourceOffset: config.linearNumKeyHeads * config.linearKeyHeadDim
                * MemoryLayout<Float16>.stride,
             destination: deltaKey,
             size: config.linearNumKeyHeads * config.linearKeyHeadDim
                * MemoryLayout<Float16>.stride)
        copy(commandBuffer: commandBuffer,
             source: deltaConvolution,
             sourceOffset: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                * MemoryLayout<Float16>.stride,
             destination: deltaValue,
             size: config.linearNumValueHeads * config.linearValueHeadDim
                * MemoryLayout<Float16>.stride)
        deltaElementwise.encodeDeltaParameters(commandBuffer: commandBuffer,
                                               a: deltaA,
                                               betaInput: deltaB,
                                               aLog: weights.aLog.buffer,
                                               aLogOffset: Int(weights.aLog.offset),
                                               dtBias: weights.dtBias.buffer,
                                               dtBiasOffset: Int(weights.dtBias.offset),
                                               decay: deltaDecay,
                                               beta: deltaBeta,
                                               count: UInt32(config.linearNumValueHeads))
        deltaNet.encodeRecurrent(commandBuffer: commandBuffer,
                                 query: deltaQuery,
                                 key: deltaKey,
                                 value: deltaValue,
                                 decay: deltaDecay,
                                 beta: deltaBeta,
                                 output: deltaOutput,
                                 state: deltaStates.state(layer: layer))
        deltaElementwise.encodeGatedNorm(commandBuffer: commandBuffer,
                                         input: deltaOutput,
                                         gate: deltaZ,
                                         weight: weights.norm.buffer,
                                         weightOffset: Int(weights.norm.offset),
                                         output: normed,
                                         headCount: UInt32(config.linearNumValueHeads),
                                         headDimension: UInt32(config.linearValueHeadDim),
                                         epsilon: 1e-6)
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.out.buffer,
                    weightsOffset: Int(weights.out.offset),
                    scales: weights.out.buffer,
                    scalesOffset: Int(weights.out.scaleOffset),
                    biases: weights.out.buffer,
                    biasesOffset: Int(weights.out.biasOffset),
                    x: normed,
                    y: mixerOutput,
                    m: UInt32(config.hiddenSize),
                    n: UInt32(config.linearNumValueHeads * config.linearValueHeadDim))
    }

    private func encodeFullAttention(commandBuffer: MTLCommandBuffer,
                                     layer: Int) throws {
        let weights = try model.qwenFullAttentionWeights(layer: layer)
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.q.buffer,
                    weightsOffset: Int(weights.q.offset),
                    scales: weights.q.buffer,
                    scalesOffset: Int(weights.q.scaleOffset),
                    biases: weights.q.buffer,
                    biasesOffset: Int(weights.q.biasOffset),
                    x: normed,
                    y: projection,
                    m: UInt32(qWidth * 2),
                    n: UInt32(config.hiddenSize))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.k.buffer,
                    weightsOffset: Int(weights.k.offset),
                    scales: weights.k.buffer,
                    scalesOffset: Int(weights.k.scaleOffset),
                    biases: weights.k.buffer,
                    biasesOffset: Int(weights.k.biasOffset),
                    x: normed,
                    y: key,
                    m: UInt32(kvWidth),
                    n: UInt32(config.hiddenSize))
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.v.buffer,
                    weightsOffset: Int(weights.v.offset),
                    scales: weights.v.buffer,
                    scalesOffset: Int(weights.v.scaleOffset),
                    biases: weights.v.buffer,
                    biasesOffset: Int(weights.v.biasOffset),
                    x: normed,
                    y: value,
                    m: UInt32(kvWidth),
                    n: UInt32(config.hiddenSize))
        attention.encodeSplitQueryGate(commandBuffer: commandBuffer,
                                       projection: projection,
                                       query: query,
                                       gate: queryGate)
        guard let cache = fullCaches[layer] else {
            throw ModelError.indexCorrupt(detail: "missing full-attention cache for layer \(layer)")
        }
        attention.encodeQueryKey(commandBuffer: commandBuffer,
                                 query: query,
                                 key: key,
                                 queryNorm: (try model.qwenFullAttentionWeights(layer: layer)).qNorm.buffer,
                                 queryNormOffset: Int((try model.qwenFullAttentionWeights(layer: layer)).qNorm.offset),
                                 keyNorm: (try model.qwenFullAttentionWeights(layer: layer)).kNorm.buffer,
                                 keyNormOffset: Int((try model.qwenFullAttentionWeights(layer: layer)).kNorm.offset),
                                 normalizedQuery: normalizedQuery,
                                 normalizedKey: normalizedKey,
                                 position: UInt32(position),
                                 epsilon: 1e-6)
        cache.append(commandBuffer: commandBuffer,
                     key: normalizedKey,
                     value: value)
        attention.encode(commandBuffer: commandBuffer,
                         query: normalizedQuery,
                         keyValueCache: cache,
                         output: attentionOutput)
        attention.encodeOutputGate(commandBuffer: commandBuffer,
                                   attention: attentionOutput,
                                   gate: queryGate,
                                   output: gatedAttention)
        gemv.encode(commandBuffer: commandBuffer,
                    weights: weights.o.buffer,
                    weightsOffset: Int(weights.o.offset),
                    scales: weights.o.buffer,
                    scalesOffset: Int(weights.o.scaleOffset),
                    biases: weights.o.buffer,
                    biasesOffset: Int(weights.o.biasOffset),
                    x: gatedAttention,
                    y: mixerOutput,
                    m: UInt32(config.hiddenSize),
                    n: UInt32(qWidth))
    }

    private func sharedProjection(_ view: TensorView,
                                  rows: Int,
                                  cols: Int) -> SharedExpertProjection {
        SharedExpertProjection(weights: view.buffer,
                               scales: view.buffer,
                               biases: view.buffer,
                               weightsOffset: Int(view.offset),
                               scalesOffset: Int(view.scaleOffset),
                               biasesOffset: Int(view.biasOffset),
                               rows: UInt32(rows),
                               cols: UInt32(cols))
    }

    private func copy(commandBuffer: MTLCommandBuffer,
                      source: MTLBuffer,
                      sourceOffset: Int,
                      destination: MTLBuffer,
                      size: Int) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: source,
                  sourceOffset: sourceOffset,
                  to: destination,
                  destinationOffset: 0,
                  size: size)
        blit.endEncoding()
    }

    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try body(commandBuffer)
        commandBuffer.commit()
        commandBufferSubmissionCount += 1
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        guard commandBuffer.status == .completed else {
            throw ModelError.residentBufferWrapFailed
        }
    }
}