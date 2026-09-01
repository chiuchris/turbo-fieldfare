public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public enum QwenGPUExecutionMode: String, Codable, Sendable {
    case ordered
    case parallelDeltaProjections = "parallel-delta-projections"
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let allowedExpertCacheSlots = [8, 16, 24, 32]
    public static let defaultExpertCacheSlots = 16
    public static let allowedNgramReadConcurrency = [1, 4, 8, 16]
    public static let defaultNgramReadConcurrency = 8
    public static let defaultNgramRowCacheBytes = 8 * 1024 * 1024
    public static let maxNgramRowCacheBytes = 256 * 1024 * 1024
    public static let defaultNgramRowCacheMaxUniqueRows = 64
    public static let maxNgramRowCacheMaxUniqueRows = 4_096
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedChunkTokens
    public static let minimumExpertCacheSlotsForChunkedPrefill = 16

    public let expertCacheSlots: Int
    public let ngramReadConcurrency: Int
    public let ngramRowCacheBytes: Int
    public let ngramRowCacheMaxUniqueRows: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    public let qwenGPUStageTimingEnabled: Bool
    public let qwenGPUExecutionMode: QwenGPUExecutionMode

    public init(expertCacheSlots: Int = RuntimeConfiguration.defaultExpertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                qwenGPUStageTimingEnabled: Bool = false,
                qwenGPUExecutionMode: QwenGPUExecutionMode = .ordered,
                ngramReadConcurrency: Int = RuntimeConfiguration.defaultNgramReadConcurrency,
                ngramRowCacheBytes: Int = RuntimeConfiguration.defaultNgramRowCacheBytes,
                ngramRowCacheMaxUniqueRows: Int = RuntimeConfiguration.defaultNgramRowCacheMaxUniqueRows) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedNgramReadConcurrency.contains(ngramReadConcurrency),
                     "unsupported n-gram read concurrency")
        precondition(ngramRowCacheBytes >= 0,
                     "n-gram row cache bytes must be non-negative")
        precondition(ngramRowCacheBytes <= Self.maxNgramRowCacheBytes,
                     "n-gram row cache bytes exceed maximum")
        precondition(ngramRowCacheMaxUniqueRows > 0,
                     "n-gram row cache row count must be positive")
        precondition(ngramRowCacheMaxUniqueRows <= Self.maxNgramRowCacheMaxUniqueRows,
                     "n-gram row cache row count exceeds maximum")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.ngramReadConcurrency = ngramReadConcurrency
        self.ngramRowCacheBytes = ngramRowCacheBytes
        self.ngramRowCacheMaxUniqueRows = ngramRowCacheMaxUniqueRows
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.qwenGPUStageTimingEnabled = qwenGPUStageTimingEnabled
        self.qwenGPUExecutionMode = qwenGPUExecutionMode
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
