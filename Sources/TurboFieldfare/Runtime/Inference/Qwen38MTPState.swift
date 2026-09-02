import Metal

struct Qwen38MTPStateSnapshot {
    let position: Int
    let qsa: Qwen38QSARawKeySnapshot
    let fullAttention: QwenFullAttentionKVSnapshot
    let feedback: [UInt16]?
    let qsaSelectionMask: [UInt8]?
    let qsaSelectionKeyCount: Int
}

/// Mutable state used by the MTP draft path. It is deliberately separate from
/// Qwen38RuntimeState so speculative tokens cannot mutate target caches.
final class Qwen38MTPState {
    private static let feedbackElementCount =
        Qwen38MTPExecutionGeometry.qwen.streamCount
            * Qwen38MTPExecutionGeometry.qwen.hiddenSize

    let maxContext: Int
    let qsa: Qwen38QSALayerState
    let fullAttention: QwenFullAttentionKVCache
    let qsaSelectionMask: MTLBuffer
    private let feedbackStorage: MTLBuffer
    private(set) var hasFeedback = false
    private(set) var qsaSelectionKeyCount = 0
    private(set) var position = 0

    var feedback: MTLBuffer? {
        hasFeedback ? feedbackStorage : nil
    }

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
        guard let qsaSelectionMask = model.device.makeBuffer(
            length: maxContext,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.qsaSelectionMask = qsaSelectionMask
        let feedbackBytes = Self.feedbackElementCount
            * MemoryLayout<UInt16>.stride
        guard let feedbackStorage = model.device.makeBuffer(
            length: feedbackBytes,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.feedbackStorage = feedbackStorage
    }

    func storeFeedback(from source: MTLBuffer) {
        let feedbackBytes = Self.feedbackElementCount
            * MemoryLayout<UInt16>.stride
        precondition(source.length >= feedbackBytes,
                     "MTP feedback buffer is smaller than expected")
        feedbackStorage.contents().copyMemory(
            from: source.contents(),
            byteCount: feedbackBytes)
        hasFeedback = true
    }

    func recordQSASelection(keyCount: Int) {
        precondition(qsaSelectionKeyCount == 0,
                     "MTP QSA selection row is already recorded")
        precondition(keyCount > 0 && keyCount <= maxContext,
                     "MTP QSA selection key count exceeds context capacity")
        qsaSelectionKeyCount = keyCount
    }

    func advance(by tokenCount: Int) {
        precondition(tokenCount >= 0 && position + tokenCount <= maxContext,
                     "MTP state position exceeds context capacity")
        position += tokenCount
    }

    func snapshot() -> Qwen38MTPStateSnapshot {
        let feedback = hasFeedback
            ? Array(UnsafeBufferPointer(
                start: feedbackStorage.contents()
                    .assumingMemoryBound(to: UInt16.self),
                count: Self.feedbackElementCount))
            : nil
        let selectionMask = qsaSelectionKeyCount > 0
            ? Array(UnsafeBufferPointer(
                start: qsaSelectionMask.contents()
                    .assumingMemoryBound(to: UInt8.self),
                count: qsaSelectionKeyCount))
            : nil
        return Qwen38MTPStateSnapshot(
            position: position,
            qsa: qsa.rawKeyCache.snapshot(),
            fullAttention: fullAttention.snapshot(),
            feedback: feedback,
            qsaSelectionMask: selectionMask,
            qsaSelectionKeyCount: qsaSelectionKeyCount)
    }

    func restore(_ snapshot: Qwen38MTPStateSnapshot) {
        precondition(snapshot.position >= 0 && snapshot.position <= maxContext,
                     "MTP state snapshot position exceeds context capacity")
        qsa.rawKeyCache.restore(snapshot.qsa)
        fullAttention.restore(snapshot.fullAttention)
        if let feedback = snapshot.feedback {
            precondition(feedback.count == Self.feedbackElementCount,
                         "MTP feedback snapshot has an unexpected size")
            feedback.withUnsafeBufferPointer { source in
                feedbackStorage.contents().copyMemory(
                    from: source.baseAddress!,
                    byteCount: feedback.count * MemoryLayout<UInt16>.stride)
            }
            hasFeedback = true
        } else {
            hasFeedback = false
        }
        if let selectionMask = snapshot.qsaSelectionMask {
            precondition(snapshot.qsaSelectionKeyCount > 0,
                         "MTP selection snapshot has an unexpected key count")
            precondition(selectionMask.count == snapshot.qsaSelectionKeyCount,
                         "MTP selection snapshot mask does not match its key count")
            precondition(selectionMask.count <= maxContext,
                         "MTP selection snapshot exceeds context capacity")
            selectionMask.withUnsafeBufferPointer { source in
                qsaSelectionMask.contents().copyMemory(
                    from: source.baseAddress!,
                    byteCount: selectionMask.count)
            }
        } else {
            precondition(snapshot.qsaSelectionKeyCount == 0,
                         "MTP selection snapshot is missing its mask")
        }
        qsaSelectionKeyCount = snapshot.qsaSelectionKeyCount
        position = snapshot.position
    }

    func reset() {
        qsa.rawKeyCache.reset()
        fullAttention.reset()
        memset(qsaSelectionMask.contents(), 0, qsaSelectionMask.length)
        hasFeedback = false
        qsaSelectionKeyCount = 0
        position = 0
    }
}
