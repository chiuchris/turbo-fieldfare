import Metal

struct Qwen38MTPStateSnapshot {
    let position: Int
    let qsa: Qwen38QSARawKeySnapshot
    let fullAttention: QwenFullAttentionKVSnapshot
}

/// Mutable state used by the MTP draft path. It is deliberately separate from
/// Qwen38RuntimeState so speculative tokens cannot mutate target caches.
final class Qwen38MTPState {
    let maxContext: Int
    let qsa: Qwen38QSALayerState
    let fullAttention: QwenFullAttentionKVCache
    private(set) var position = 0

    init(model: Model, maxContext: Int) throws {
        guard model.config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(model.config.modelFamily)")
        }
        guard model.hasMTP else {
            throw ModelError.archMismatch(
                field: "mtp",
                expected: "present",
                actual: "missing")
        }
        guard maxContext > 0 else {
            throw ModelError.archMismatch(
                field: "maxContext",
                expected: "positive",
                actual: "\(maxContext)")
        }
        guard let architecture = model.config.qwen38Architecture else {
            throw ModelError.archMismatch(
                field: "qwen38Architecture",
                expected: "present",
                actual: "nil")
        }
        let mtpWeights = try Qwen38MTPWeights(model: model)
        let qsaGeometry = Qwen38QSAGeometry(
            queryHeads: UInt32(architecture.indexerHeads),
            keyValueHeads: UInt32(architecture.indexerKeyValueHeads),
            headDimension: UInt32(architecture.indexerHeadDim),
            compressRatio: UInt32(architecture.indexerCompressRatio),
            tokenBudget: UInt32(architecture.indexerBudget),
            rotaryDimension: UInt32(
                Double(architecture.indexerHeadDim)
                    * model.config.partialRotaryFactor),
            ropeTheta: Float(model.config.ropeTheta))
        let mtpGeometry = Qwen38MTPExecutionGeometry.qwen
        let fullGeometry = QwenFullAttentionGeometry(
            queryHeads: mtpGeometry.fullAttentionQueryHeads,
            keyValueHeads: mtpGeometry.fullAttentionKeyValueHeads,
            headDimension: mtpGeometry.fullAttentionHeadDimension,
            rotaryDimension: Int(
                Double(mtpGeometry.fullAttentionHeadDimension)
                    * model.config.partialRotaryFactor),
            ropeTheta: Float(model.config.fullRopeTheta))

        self.maxContext = maxContext
        let qsaCache = try Qwen38QSARawKeyCache(
            device: model.device,
            capacity: maxContext,
            geometry: qsaGeometry)
        self.qsa = Qwen38QSALayerState(
            layer: 0,
            weights: Qwen38QSALayerWeights(
                projection: try mtpWeights.tensor(role: .indexerProjection),
                queryNorm: try mtpWeights.tensor(role: .indexerQueryNorm),
                keyNorm: try mtpWeights.tensor(role: .indexerKeyNorm)),
            rawKeyCache: qsaCache)
        self.fullAttention = try QwenFullAttentionKVCache(
            device: model.device,
            capacity: maxContext,
            geometry: fullGeometry)
    }

    func advance(by tokenCount: Int) {
        precondition(tokenCount >= 0 && position + tokenCount <= maxContext,
                     "MTP state position exceeds context capacity")
        position += tokenCount
    }

    func snapshot() -> Qwen38MTPStateSnapshot {
        Qwen38MTPStateSnapshot(
            position: position,
            qsa: qsa.rawKeyCache.snapshot(),
            fullAttention: fullAttention.snapshot())
    }

    func restore(_ snapshot: Qwen38MTPStateSnapshot) {
        precondition(snapshot.position >= 0 && snapshot.position <= maxContext,
                     "MTP state snapshot position exceeds context capacity")
        qsa.rawKeyCache.restore(snapshot.qsa)
        fullAttention.restore(snapshot.fullAttention)
        position = snapshot.position
    }

    func reset() {
        qsa.rawKeyCache.reset()
        fullAttention.reset()
        position = 0
    }
}
