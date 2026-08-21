import Foundation

public protocol QwenDecodeDiagnosticsProviding: AnyObject {
    var lastQwenDecodeDiagnostics: QwenDecodeDiagnostics? { get }
}

public struct QwenDecodeLayerDiagnostics: Sendable, Equatable {
    public let layer: Int
    public let isFullAttention: Bool
    public let elapsedNanos: UInt64
    public let expertFetchNanos: UInt64

    public init(layer: Int,
                isFullAttention: Bool,
                elapsedNanos: UInt64,
                expertFetchNanos: UInt64) {
        self.layer = layer
        self.isFullAttention = isFullAttention
        self.elapsedNanos = elapsedNanos
        self.expertFetchNanos = expertFetchNanos
    }
}

public struct QwenDecodeDiagnostics: Sendable, Equatable {
    public let wallNanos: UInt64
    public let embeddingNanos: UInt64
    public let layerNanos: UInt64
    public let logitsNanos: UInt64
    public let expertFetchNanos: UInt64
    public let layerCount: Int
    public let fullAttentionLayerCount: Int
    public let deltaNetLayerCount: Int
    public let commandBufferCount: Int
    public let routerEvaluationCount: Int
    public let routedExpertCount: Int
    public let routedExpertCacheHitCount: Int
    public let routedExpertCacheMissCount: Int
    public let routedExpertEstimatedBytes: UInt64
    public let layers: [QwenDecodeLayerDiagnostics]

    public init(wallNanos: UInt64,
                embeddingNanos: UInt64,
                layerNanos: UInt64,
                logitsNanos: UInt64,
                expertFetchNanos: UInt64,
                layerCount: Int,
                fullAttentionLayerCount: Int,
                deltaNetLayerCount: Int,
                commandBufferCount: Int,
                routerEvaluationCount: Int,
                routedExpertCount: Int,
                routedExpertCacheHitCount: Int,
                routedExpertCacheMissCount: Int,
                routedExpertEstimatedBytes: UInt64,
                layers: [QwenDecodeLayerDiagnostics] = []) {
        self.wallNanos = wallNanos
        self.embeddingNanos = embeddingNanos
        self.layerNanos = layerNanos
        self.logitsNanos = logitsNanos
        self.expertFetchNanos = expertFetchNanos
        self.layerCount = layerCount
        self.fullAttentionLayerCount = fullAttentionLayerCount
        self.deltaNetLayerCount = deltaNetLayerCount
        self.commandBufferCount = commandBufferCount
        self.routerEvaluationCount = routerEvaluationCount
        self.routedExpertCount = routedExpertCount
        self.routedExpertCacheHitCount = routedExpertCacheHitCount
        self.routedExpertCacheMissCount = routedExpertCacheMissCount
        self.routedExpertEstimatedBytes = routedExpertEstimatedBytes
        self.layers = layers
    }
}