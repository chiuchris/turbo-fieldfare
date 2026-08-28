import Metal

struct Qwen38QSALayerWeights {
    let projection: TensorView
    let queryNorm: TensorView
    let keyNorm: TensorView
}

final class Qwen38QSALayerState {
    let layer: Int
    let weights: Qwen38QSALayerWeights
    let rawKeyCache: Qwen38QSARawKeyCache

    init(layer: Int,
         weights: Qwen38QSALayerWeights,
         rawKeyCache: Qwen38QSARawKeyCache) {
        self.layer = layer
        self.weights = weights
        self.rawKeyCache = rawKeyCache
    }
}

struct Qwen38QSALayerExecutionScratch {
    let projectedRows: MTLBuffer
    let blockScores: MTLBuffer
    let selection: Qwen38QSASelectionScratch
}

final class Qwen38QSALayerExecutor {
    private let projection: Qwen38QSAProjection
    private let scorer: Qwen38QSABlockScorer
    private let selector: Qwen38QSASelector
    let geometry: Qwen38QSAGeometry

    init(context: MetalContext, geometry: Qwen38QSAGeometry = .qwen) throws {
        self.geometry = geometry
        self.projection = try Qwen38QSAProjection(
            context: context, geometry: geometry)
        self.scorer = try Qwen38QSABlockScorer(
            context: context, geometry: geometry)
        self.selector = try Qwen38QSASelector(
            context: context, geometry: geometry)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                state: Qwen38QSALayerState,
                hiddenStates: MTLBuffer,
                queryPositions: MTLBuffer,
                visibleTokenCounts: MTLBuffer,
                scratch: Qwen38QSALayerExecutionScratch,
                tokenCount: UInt32,
                inputWidth: UInt32,
                epsilon: Float,
                validTokenCount: UInt32? = nil) {
        precondition(state.rawKeyCache.geometry == geometry,
                     "QSA executor and layer state geometry must match")
        precondition(tokenCount > 0 && inputWidth > 0)
        let validTokens = validTokenCount ?? tokenCount
        precondition(validTokens <= tokenCount,
                     "QSA valid-token count cannot exceed allocated rows")
        let keyCount = state.rawKeyCache.count + Int(validTokens)
        precondition(keyCount <= state.rawKeyCache.capacity,
                     "QSA layer execution exceeds cache capacity")
        let projectedBytes = Int(validTokens * geometry.projectionWidth)
            * MemoryLayout<Float16>.stride
        let blockCount = keyCount / Int(geometry.compressRatio)
        let scoreBytes = Int(validTokens) * blockCount
            * MemoryLayout<Float>.stride
        let maskBytes = Int(tokenCount) * keyCount
        precondition(scratch.projectedRows.length >= projectedBytes,
                     "QSA projected-row scratch is too small")
        precondition(scratch.blockScores.length >= scoreBytes,
                     "QSA block-score scratch is too small")
        precondition(scratch.selection.state.length >=
            Int(validTokens) * Qwen38QSASelector.stateBytesPerQuery,
            "QSA selection-state scratch is too small")
        precondition(scratch.selection.tokenMask.length >= maskBytes,
                     "QSA token-mask scratch is too small")
        precondition(queryPositions.length >=
            Int(validTokens) * MemoryLayout<UInt32>.stride,
            "QSA query positions are too small")
        precondition(visibleTokenCounts.length >=
            Int(validTokens) * MemoryLayout<UInt32>.stride,
            "QSA visible-token counts are too small")

        if maskBytes > 0,
           let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: scratch.selection.tokenMask,
                      range: 0..<maskBytes,
                      value: 0)
            blit.endEncoding()
        }
        guard validTokens > 0 else { return }

        let weights = state.weights.projection
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.buffer,
            weightsOffset: Int(weights.offset),
            scales: weights.buffer,
            scalesOffset: Int(weights.scaleOffset),
            biases: weights.buffer,
            biasesOffset: Int(weights.biasOffset),
            hiddenStates: hiddenStates,
            projectedRows: scratch.projectedRows,
            positions: queryPositions,
            rawKeyCache: state.rawKeyCache,
            tokenCount: validTokens,
            inputWidth: inputWidth)

        if blockCount > 0 {
            scorer.encode(
                commandBuffer: commandBuffer,
                projectedQueries: scratch.projectedRows,
                rawKeyCache: state.rawKeyCache,
                queryNorm: state.weights.queryNorm.buffer,
                queryNormOffset: Int(state.weights.queryNorm.offset),
                keyNorm: state.weights.keyNorm.buffer,
                keyNormOffset: Int(state.weights.keyNorm.offset),
                queryPositions: queryPositions,
                outputScores: scratch.blockScores,
                queryCount: validTokens,
                epsilon: epsilon)
        }
        selector.encode(
            commandBuffer: commandBuffer,
            scores: scratch.blockScores,
            visibleTokenCounts: visibleTokenCounts,
            scratch: scratch.selection,
            queryCount: validTokens,
            keyCount: UInt32(keyCount))
    }
}

struct Qwen38QSASnapshot {
    let layers: [Qwen38QSARawKeySnapshot?]
}

final class Qwen38QSAStateManager {
    let geometry: Qwen38QSAGeometry
    let capacity: Int
    private let layers: [Qwen38QSALayerState?]

    init(model: Model, capacity: Int) throws {
        guard model.config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(model.config.modelFamily)")
        }
        guard capacity > 0 else {
            throw ModelError.archMismatch(
                field: "maxContext",
                expected: "positive",
                actual: "\(capacity)")
        }
        guard let architecture = model.config.qwen38Architecture else {
            throw ModelError.archMismatch(
                field: "qwen38Architecture",
                expected: "present",
                actual: "nil")
        }
        let geometry = Qwen38QSAGeometry(
            queryHeads: UInt32(architecture.indexerHeads),
            keyValueHeads: UInt32(architecture.indexerKeyValueHeads),
            headDimension: UInt32(architecture.indexerHeadDim),
            compressRatio: UInt32(architecture.indexerCompressRatio),
            tokenBudget: UInt32(architecture.indexerBudget),
            rotaryDimension: UInt32(
                Double(architecture.indexerHeadDim)
                    * model.config.partialRotaryFactor),
            ropeTheta: Float(model.config.ropeTheta))
        guard geometry == .qwen else {
            throw ModelError.archMismatch(
                field: "qwen38QSA",
                expected: "\(Qwen38QSAGeometry.qwen)",
                actual: "\(geometry)")
        }

        var layers = Array<Qwen38QSALayerState?>(
            repeating: nil,
            count: model.config.numLayers)
        for layer in 0..<model.config.numLayers where Qwen38TensorNames.hasQSA(layer: layer) {
            let weights = Qwen38QSALayerWeights(
                projection: try model.qwen38QSA(
                    layer: layer, tensor: .queryKeyProjection),
                queryNorm: try model.qwen38QSA(layer: layer, tensor: .queryNorm),
                keyNorm: try model.qwen38QSA(layer: layer, tensor: .keyNorm))
            let cache = try Qwen38QSARawKeyCache(
                device: model.device,
                capacity: capacity,
                geometry: geometry)
            layers[layer] = Qwen38QSALayerState(
                layer: layer,
                weights: weights,
                rawKeyCache: cache)
        }
        self.geometry = geometry
        self.capacity = capacity
        self.layers = layers
    }

    var layerCount: Int {
        layers.compactMap { $0 }.count
    }

    func state(layer: Int) -> Qwen38QSALayerState? {
        precondition(layer >= 0 && layer < layers.count,
                     "QSA layer index is out of bounds")
        return layers[layer]
    }

    func snapshot() -> Qwen38QSASnapshot {
        Qwen38QSASnapshot(
            layers: layers.map { $0?.rawKeyCache.snapshot() })
    }

    func restore(_ snapshot: Qwen38QSASnapshot) {
        precondition(snapshot.layers.count == layers.count,
                     "QSA snapshot layer count does not match state")
        for (index, layer) in layers.enumerated() {
            switch (layer, snapshot.layers[index]) {
            case let (.some(layer), .some(cache)):
                layer.rawKeyCache.restore(cache)
            case (.none, .none):
                break
            default:
                preconditionFailure("QSA snapshot sparse-layer layout does not match state")
            }
        }
    }

    func reset() {
        for layer in layers {
            layer?.rawKeyCache.reset()
        }
    }
}
