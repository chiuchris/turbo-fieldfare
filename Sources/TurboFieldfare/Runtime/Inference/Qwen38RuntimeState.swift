import Metal

struct Qwen38RuntimeStateSnapshot {
    let pleConvolution: Qwen38PLEConvolutionSnapshot
    let qsa: Qwen38QSASnapshot
    let deltaStates: [QwenGatedDeltaNetSnapshot?]
    let fullCaches: [QwenFullAttentionKVSnapshot?]
}

final class Qwen38RuntimeState {
    let maxContext: Int
    let deltaGeometry: QwenGatedDeltaNetGeometry
    let fullAttentionGeometry: QwenFullAttentionGeometry
    let pleConvolution: Qwen38PLEConvolutionState
    let qsa: Qwen38QSAStateManager

    private let deltaStates: [QwenGatedDeltaNetState?]
    private let fullCaches: [QwenFullAttentionKVCache?]

    init(model: Model, maxContext: Int) throws {
        let config = model.config
        guard config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(config.modelFamily)")
        }
        guard maxContext > 0 else {
            throw ModelError.archMismatch(
                field: "maxContext",
                expected: "positive",
                actual: "\(maxContext)")
        }
        guard config.fullAttentionLayerMask.count == config.numLayers else {
            throw ModelError.archMismatch(
                field: "fullAttentionLayerMask.count",
                expected: "\(config.numLayers)",
                actual: "\(config.fullAttentionLayerMask.count)")
        }
        guard let architecture = config.qwen38Architecture else {
            throw ModelError.archMismatch(
                field: "qwen38Architecture",
                expected: "present",
                actual: "nil")
        }
        guard architecture.pleLayerIDs == [2] else {
            throw ModelError.archMismatch(
                field: "pleLayerIDs",
                expected: "[2]",
                actual: "\(architecture.pleLayerIDs)")
        }

        let deltaGeometry = QwenGatedDeltaNetGeometry(
            keyHeads: config.linearNumKeyHeads,
            valueHeads: config.linearNumValueHeads,
            keyHeadDim: config.linearKeyHeadDim,
            valueHeadDim: config.linearValueHeadDim,
            convolutionKernel: config.linearConvKernelDim)
        let fullAttentionGeometry = QwenFullAttentionGeometry(
            queryHeads: config.numHeads,
            keyValueHeads: config.numFullKVHeads,
            headDimension: config.fullHeadDim,
            rotaryDimension: Int(
                Double(config.fullHeadDim) * config.partialRotaryFactor),
            ropeTheta: Float(config.fullRopeTheta))
        let convolutionChannels = deltaGeometry.qkvDimension

        var deltaStates = Array<QwenGatedDeltaNetState?>(
            repeating: nil,
            count: config.numLayers)
        var fullCaches = Array<QwenFullAttentionKVCache?>(
            repeating: nil,
            count: config.numLayers)
        for layer in 0..<config.numLayers {
            if config.fullAttentionLayerMask[layer] != 0 {
                guard Qwen38TensorNames.hasQSA(layer: layer) else {
                    throw ModelError.archMismatch(
                        field: "fullAttentionLayerMask[\(layer)]",
                        expected: "QSA layer",
                        actual: "non-QSA layer")
                }
                fullCaches[layer] = try QwenFullAttentionKVCache(
                    device: model.device,
                    capacity: maxContext,
                    geometry: fullAttentionGeometry)
            } else {
                deltaStates[layer] = try QwenGatedDeltaNetState(
                    device: model.device,
                    geometry: deltaGeometry,
                    convolutionChannels: convolutionChannels)
            }
        }

        self.maxContext = maxContext
        self.deltaGeometry = deltaGeometry
        self.fullAttentionGeometry = fullAttentionGeometry
        self.pleConvolution = try Qwen38PLEConvolutionState(
            device: model.device,
            channels: architecture.pleEmbeddingSize,
            kernelSize: architecture.pleConvolutionKernel)
        self.qsa = try Qwen38QSAStateManager(
            model: model,
            capacity: maxContext)
        self.deltaStates = deltaStates
        self.fullCaches = fullCaches
    }

    var linearLayerCount: Int {
        deltaStates.compactMap { $0 }.count
    }

    var sparseLayerCount: Int {
        fullCaches.compactMap { $0 }.count
    }

    func deltaState(layer: Int) -> QwenGatedDeltaNetState? {
        precondition(layer >= 0 && layer < deltaStates.count,
                     "Qwen3.8 layer index is out of bounds")
        return deltaStates[layer]
    }

    func fullCache(layer: Int) -> QwenFullAttentionKVCache? {
        precondition(layer >= 0 && layer < fullCaches.count,
                     "Qwen3.8 layer index is out of bounds")
        return fullCaches[layer]
    }

    func snapshot() -> Qwen38RuntimeStateSnapshot {
        Qwen38RuntimeStateSnapshot(
            pleConvolution: pleConvolution.snapshot(),
            qsa: qsa.snapshot(),
            deltaStates: deltaStates.map { $0?.snapshot() },
            fullCaches: fullCaches.map { $0?.snapshot() })
    }

    func restore(_ snapshot: Qwen38RuntimeStateSnapshot) {
        precondition(snapshot.deltaStates.count == deltaStates.count,
                     "DeltaNet snapshot layer count does not match state")
        precondition(snapshot.fullCaches.count == fullCaches.count,
                     "full-attention snapshot layer count does not match state")
        for (index, state) in deltaStates.enumerated() {
            switch (state, snapshot.deltaStates[index]) {
            case let (.some(state), .some(savedState)):
                state.restore(savedState)
            case (.none, .none):
                break
            default:
                preconditionFailure("DeltaNet snapshot layer layout does not match state")
            }
        }
        for (index, cache) in fullCaches.enumerated() {
            switch (cache, snapshot.fullCaches[index]) {
            case let (.some(cache), .some(savedCache)):
                cache.restore(savedCache)
            case (.none, .none):
                break
            default:
                preconditionFailure("full-attention snapshot layer layout does not match state")
            }
        }
        pleConvolution.restore(snapshot.pleConvolution)
        qsa.restore(snapshot.qsa)
    }

    func reset() {
        pleConvolution.reset()
        qsa.reset()
        for state in deltaStates {
            state?.reset()
        }
        for cache in fullCaches {
            cache?.reset()
        }
    }
}
