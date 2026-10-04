import Foundation

public struct QwenVisionConfig: Sendable, Equatable {
    public let hiddenSize = 1_152
    public let intermediateSize = 4_304
    public let numLayers = 27
    public let numHeads = 16
    public let headDimension = 72
    public let inputChannels = 3
    public let patchSize = 16
    public let temporalPatchSize = 2
    public let spatialMergeSize = 2
    public let positionEmbeddingCount = 2_304
    public let positionGridSize = 48
    public let outputHiddenSize = 2_048
    public let layerNormEpsilon: Float = 1e-6
    public let ropeTheta: Float = 10_000

    public var mergerInputSize: Int {
        hiddenSize * spatialMergeSize * spatialMergeSize
    }

    public init() {}
}
