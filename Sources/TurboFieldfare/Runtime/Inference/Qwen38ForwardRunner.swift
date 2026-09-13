import Foundation
import Metal

public struct Qwen38LogitDiagnostics: Codable, Sendable, Equatable {
    public let finiteCount: Int
    public let nanCount: Int
    public let positiveInfinityCount: Int
    public let negativeInfinityCount: Int
    public let zeroCount: Int
    public let minimum: Float
    public let maximum: Float
    public let greedyToken: Int32

    init(values: UnsafeBufferPointer<Float16>) {
        var finiteCount = 0
        var nanCount = 0
        var positiveInfinityCount = 0
        var negativeInfinityCount = 0
        var zeroCount = 0
        var minimum = Float.infinity
        var maximum = -Float.infinity
        var greedyToken: Int32 = 0
        var greedyValue = -Float.infinity
        for (index, value) in values.enumerated() {
            let scalar = Float(value)
            if scalar.isNaN {
                nanCount += 1
            } else if scalar == .infinity {
                positiveInfinityCount += 1
            } else if scalar == -.infinity {
                negativeInfinityCount += 1
            } else {
                finiteCount += 1
                minimum = min(minimum, scalar)
                maximum = max(maximum, scalar)
                if scalar == 0 { zeroCount += 1 }
                if scalar > greedyValue {
                    greedyValue = scalar
                    greedyToken = Int32(index)
                }
            }
        }
        self.finiteCount = finiteCount
        self.nanCount = nanCount
        self.positiveInfinityCount = positiveInfinityCount
        self.negativeInfinityCount = negativeInfinityCount
        self.zeroCount = zeroCount
        self.minimum = finiteCount > 0 ? minimum : 0
        self.maximum = finiteCount > 0 ? maximum : 0
        self.greedyToken = Int32(greedyToken)
    }
}

public struct Qwen38StageCapture: Codable, Sendable, Equatable {
    public let layerIndex: Int
    public let stage: String
    public let tokenPosition: Int
    public let inputToken: Int32
    public let shape: [Int]
    public let dtype: String
    public let values: [Float16]
    public let float32Values: [Float]?

    public init(layerIndex: Int,
                stage: String,
                tokenPosition: Int,
                inputToken: Int32,
                shape: [Int],
                dtype: String = "float16",
                values: [Float16],
                float32Values: [Float]? = nil) {
        self.layerIndex = layerIndex
        self.stage = stage
        self.tokenPosition = tokenPosition
        self.inputToken = inputToken
        self.shape = shape
        self.dtype = dtype
        self.values = values
        self.float32Values = float32Values
    }
}

public struct Qwen38RouterDiagnostics: Codable, Sendable, Equatable {
    public let layerIndex: Int
    public let tokenIndex: Int
    public let routerLogits: [Float]
    public let selectedExperts: [Int]
    public let routeWeightBits: [UInt16]
    public let sharedGateValue: Float

    public init(layerIndex: Int,
                tokenIndex: Int,
                routerLogits: [Float],
                selectedExperts: [Int],
                routeWeightBits: [UInt16],
                sharedGateValue: Float) {
        self.layerIndex = layerIndex
        self.tokenIndex = tokenIndex
        self.routerLogits = routerLogits
        self.selectedExperts = selectedExperts
        self.routeWeightBits = routeWeightBits
        self.sharedGateValue = sharedGateValue
    }
}

public struct Qwen38TargetBoundarySnapshot: Sendable {
    public let targetPosition: Int
    public let inputToken: Int32
    public let streamCount: Int
    public let hiddenSize: Int
    public let targetHiddenStreams: [Float16]
    public let targetHiddenFloat32Streams: [Float]?
    public let rawTargetHiddenStreams: [Float16]?
    public let stageCaptures: [Qwen38StageCapture]
    public let routerDiagnostics: Qwen38RouterDiagnostics?

    public init(targetPosition: Int,
                inputToken: Int32,
                streamCount: Int,
                hiddenSize: Int,
                targetHiddenStreams: [Float16],
                targetHiddenFloat32Streams: [Float]? = nil,
                rawTargetHiddenStreams: [Float16]?,
                stageCaptures: [Qwen38StageCapture] = [],
                routerDiagnostics: Qwen38RouterDiagnostics? = nil) {
        self.targetPosition = targetPosition
        self.inputToken = inputToken
        self.streamCount = streamCount
        self.hiddenSize = hiddenSize
        self.targetHiddenStreams = targetHiddenStreams
        self.targetHiddenFloat32Streams = targetHiddenFloat32Streams
        self.rawTargetHiddenStreams = rawTargetHiddenStreams
        self.stageCaptures = stageCaptures
        self.routerDiagnostics = routerDiagnostics
    }
}

public enum Qwen38TargetBoundarySnapshotError: Error, CustomStringConvertible {
    case diagnosticsDisabled
    case targetBoundaryUnavailable
    case payloadExceedsLimit(byteCount: Int, limit: Int)

    public var description: String {
        switch self {
        case .diagnosticsDisabled:
            return "target boundary snapshot requires MTP diagnostics"
        case .targetBoundaryUnavailable:
            return "no completed target boundary is available"
        case .payloadExceedsLimit(let byteCount, let limit):
            return "target boundary snapshot payload \(byteCount) exceeds limit \(limit)"
        }
    }
}

public struct Qwen38DecodeTimingSample: Codable, Sendable, Equatable {
    public let embeddingNanos: UInt64
    public let pleNanos: UInt64
    public let attentionRouterNanos: UInt64
    public let deltaNetNanos: UInt64
    public let expertFetchNanos: UInt64
    public let expertCacheHits: Int
    public let expertCacheMisses: Int
    public let moeNanos: UInt64
    public let finalHeadNanos: UInt64
    public let gpuActiveNanos: UInt64
    public let commandBufferCount: Int
    public let commandBufferEncodeNanos: UInt64
    public let commandBufferWaitNanos: UInt64

    public static let zero = Qwen38DecodeTimingSample(
        embeddingNanos: 0,
        pleNanos: 0,
        attentionRouterNanos: 0,
        deltaNetNanos: 0,
        expertFetchNanos: 0,
        moeNanos: 0,
        finalHeadNanos: 0,
        gpuActiveNanos: 0,
        commandBufferCount: 0,
        commandBufferEncodeNanos: 0,
        commandBufferWaitNanos: 0)

    public init(embeddingNanos: UInt64,
                pleNanos: UInt64,
                attentionRouterNanos: UInt64,
                deltaNetNanos: UInt64 = 0,
                expertFetchNanos: UInt64,
                expertCacheHits: Int = 0,
                expertCacheMisses: Int = 0,
                moeNanos: UInt64,
                finalHeadNanos: UInt64,
                gpuActiveNanos: UInt64,
                commandBufferCount: Int,
                commandBufferEncodeNanos: UInt64 = 0,
                commandBufferWaitNanos: UInt64 = 0) {
        self.embeddingNanos = embeddingNanos
        self.pleNanos = pleNanos
        self.attentionRouterNanos = attentionRouterNanos
        self.deltaNetNanos = deltaNetNanos
        self.expertFetchNanos = expertFetchNanos
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.moeNanos = moeNanos
        self.finalHeadNanos = finalHeadNanos
        self.gpuActiveNanos = gpuActiveNanos
        self.commandBufferCount = commandBufferCount
        self.commandBufferEncodeNanos = commandBufferEncodeNanos
        self.commandBufferWaitNanos = commandBufferWaitNanos
    }
}

public struct Qwen38SpeculativeReplaySample: Codable, Sendable, Equatable {
    public let replayedTokenCount: Int
    public let replayNanos: UInt64

    public static let zero = Qwen38SpeculativeReplaySample(
        replayedTokenCount: 0,
        replayNanos: 0)

    public init(replayedTokenCount: Int, replayNanos: UInt64) {
        self.replayedTokenCount = replayedTokenCount
        self.replayNanos = replayNanos
    }
}

public struct Qwen38DraftingDiagnostics: Codable, Sendable, Equatable {
    public let strategy: Qwen38DraftingStrategy
    public let proposedToken: Int32?
    public let targetToken: Int32?
    public let matchesTarget: Bool?
    public let fallbackReason: String?
    public let inputToken: Int32?
    public let proposalPosition: Int?
    public let targetPosition: Int?

    public init(strategy: Qwen38DraftingStrategy,
                proposedToken: Int32? = nil,
                targetToken: Int32? = nil,
                matchesTarget: Bool? = nil,
                fallbackReason: String? = nil,
                inputToken: Int32? = nil,
                proposalPosition: Int? = nil,
                targetPosition: Int? = nil) {
        self.strategy = strategy
        self.proposedToken = proposedToken
        self.targetToken = targetToken
        self.matchesTarget = matchesTarget
        self.fallbackReason = fallbackReason
        self.inputToken = inputToken
        self.proposalPosition = proposalPosition
        self.targetPosition = targetPosition
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
    let commandBufferEncodeNanos: UInt64
    let commandBufferWaitNanos: UInt64
    let routedMoENanos: UInt64
    let cacheHits: Int
    let cacheMisses: Int
    let expertReadCount: Int
    let expertReadNanos: UInt64
    let expertReadMaxNanos: UInt64
}

struct Qwen38BatchLogitLayout: Sendable, Equatable {
    let tokenCount: Int
    let vocabularySize: Int
    let rowByteStride: Int
    let byteCount: Int

    init(tokenCount: Int, vocabularySize: Int) throws {
        guard tokenCount > 0, vocabularySize > 0 else {
            throw PrefillError.chunkedUnsupported(
                "batch logits require positive token and vocabulary counts")
        }
        let (rowByteStride, rowOverflow) = vocabularySize.multipliedReportingOverflow(
            by: MemoryLayout<Float16>.stride)
        let (byteCount, byteOverflow) = tokenCount.multipliedReportingOverflow(by: rowByteStride)
        guard !rowOverflow && !byteOverflow else {
            throw PrefillError.chunkedUnsupported(
                "batch logits exceed integer capacity")
        }
        self.tokenCount = tokenCount
        self.vocabularySize = vocabularySize
        self.rowByteStride = rowByteStride
        self.byteCount = byteCount
    }

    func offset(for tokenIndex: Int) throws -> Int {
        guard tokenIndex >= 0 && tokenIndex < tokenCount else {
            throw PrefillError.chunkedUnsupported(
                "batch logits token index \(tokenIndex) is outside \(tokenCount) rows")
        }
        return tokenIndex * rowByteStride
    }
}

private struct Qwen38PromptStateSnapshot {
    let position: Int
    let ngramContext: [Int64]
    let runtimeState: Qwen38RuntimeStateSnapshot
    let mtpState: Qwen38MTPStateSnapshot?
}

private struct Qwen38SpeculativeStateCheckpoint {
    let position: Int
    let ngramContext: [Int64]
    let runtimeState: Qwen38RuntimeStateSnapshot
    let mtpState: Qwen38MTPStateSnapshot?
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
    let rawTargetHiddenStreams: MTLBuffer
    let layerZeroOutputCapture: MTLBuffer
    let layerZeroHyperInputCapture: MTLBuffer
    let layerZeroHyperNormalizedCapture: MTLBuffer
    let layerZeroHyperLowRankCapture: MTLBuffer
    let layerZeroHyperActivatedLowRankCapture: MTLBuffer
    let layerZeroHyperMixLogitsCapture: MTLBuffer
    let layerZeroMLPHyperNormalizedCapture: MTLBuffer
    let layerZeroMLPHyperMixLogitsCapture: MTLBuffer
    let layerOneOutputCapture: MTLBuffer
    let layerOneAttentionInputCapture: MTLBuffer
    let layerOneHyperNormalizedCapture: MTLBuffer
    let layerOneHyperLowRankCapture: MTLBuffer
    let layerOneHyperActivatedLowRankCapture: MTLBuffer
    let layerOneHyperMixLogitsCapture: MTLBuffer
    let layerOneMLPHyperNormalizedCapture: MTLBuffer
    let layerOneMLPHyperMixLogitsCapture: MTLBuffer
    let layerOneAttentionOutputCapture: MTLBuffer
    let layerOneAfterAttentionCapture: MTLBuffer
    let layerOneMLPInputCapture: MTLBuffer
    let layerOneMLPOutputCapture: MTLBuffer
    let layerOneDeltaQKVCapture: MTLBuffer
    let layerOneDeltaRecurrentCapture: MTLBuffer
    let layerOneDeltaNormalizedCapture: MTLBuffer
    let layerZeroAttentionOutputCapture: MTLBuffer
    let layerZeroAttentionInputCapture: MTLBuffer
    let attentionInput: MTLBuffer
    let layerZeroAfterAttentionCapture: MTLBuffer
    let layerZeroMLPInputCapture: MTLBuffer
    let layerZeroMLPOutputCapture: MTLBuffer
    let layerZeroDeltaQKVCapture: MTLBuffer
    let layerZeroDeltaRecurrentCapture: MTLBuffer
    let layerZeroDeltaNormalizedCapture: MTLBuffer
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
    let mtpLogits: MTLBuffer
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
        rawTargetHiddenStreams = try makeBuffer(batchCapacity * hyperWidth)
        layerZeroOutputCapture = try makeBuffer(hyperWidth)
        layerZeroHyperInputCapture = try makeBuffer(hyperWidth)
        layerZeroHyperNormalizedCapture = try makeBuffer(
            hyperWidth, stride: MemoryLayout<Float>.stride)
        layerZeroHyperLowRankCapture = try makeBuffer(lowRank)
        layerZeroHyperActivatedLowRankCapture = try makeBuffer(lowRank)
        layerZeroHyperMixLogitsCapture = try makeBuffer(hyperWidth)
        layerZeroMLPHyperNormalizedCapture = try makeBuffer(
            hyperWidth, stride: MemoryLayout<Float>.stride)
        layerZeroMLPHyperMixLogitsCapture = try makeBuffer(hyperWidth)
        layerOneOutputCapture = try makeBuffer(hyperWidth)
        layerOneAttentionInputCapture = try makeBuffer(
            hiddenSize, stride: MemoryLayout<Float>.stride)
        layerOneHyperNormalizedCapture = try makeBuffer(
            hyperWidth, stride: MemoryLayout<Float>.stride)
        layerOneHyperLowRankCapture = try makeBuffer(lowRank)
        layerOneHyperActivatedLowRankCapture = try makeBuffer(lowRank)
        layerOneHyperMixLogitsCapture = try makeBuffer(hyperWidth)
        layerOneMLPHyperNormalizedCapture = try makeBuffer(
            hyperWidth, stride: MemoryLayout<Float>.stride)
        layerOneMLPHyperMixLogitsCapture = try makeBuffer(hyperWidth)
        layerOneAttentionOutputCapture = try makeBuffer(hiddenSize)
        layerOneAfterAttentionCapture = try makeBuffer(hyperWidth)
        layerOneMLPInputCapture = try makeBuffer(hiddenSize)
        layerOneMLPOutputCapture = try makeBuffer(hiddenSize)
        layerOneDeltaQKVCapture = try makeBuffer(
            deltaQKVWidth, stride: MemoryLayout<Float>.stride)
        layerOneDeltaRecurrentCapture = try makeBuffer(
            deltaValueWidth, stride: MemoryLayout<Float>.stride)
        layerOneDeltaNormalizedCapture = try makeBuffer(
            deltaValueWidth, stride: MemoryLayout<Float>.stride)
        layerZeroAttentionOutputCapture = try makeBuffer(hiddenSize)
        layerZeroAttentionInputCapture = try makeBuffer(
            hiddenSize, stride: MemoryLayout<Float>.stride)
        layerZeroAfterAttentionCapture = try makeBuffer(hyperWidth)
        layerZeroMLPInputCapture = try makeBuffer(hiddenSize)
        layerZeroMLPOutputCapture = try makeBuffer(hiddenSize)
        layerZeroDeltaQKVCapture = try makeBuffer(
            deltaQKVWidth, stride: MemoryLayout<Float>.stride)
        attentionInput = try makeBuffer(
            batchCapacity * hiddenSize, stride: MemoryLayout<Float>.stride)
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
            qkv: try makeBuffer(batchCapacity * deltaQKVWidth,
                                stride: MemoryLayout<Float>.stride),
            gate: try makeBuffer(batchCapacity * deltaValueWidth,
                                 stride: MemoryLayout<Float>.stride),
            betaInput: try makeBuffer(
                batchCapacity * config.linearNumValueHeads,
                stride: MemoryLayout<Float>.stride),
            decayInput: try makeBuffer(
                batchCapacity * config.linearNumValueHeads,
                stride: MemoryLayout<Float>.stride),
            convolution: try makeBuffer(batchCapacity * deltaQKVWidth,
                                        stride: MemoryLayout<Float>.stride),
            query: try makeBuffer(batchCapacity * deltaKeyWidth,
                                  stride: MemoryLayout<Float>.stride),
            key: try makeBuffer(batchCapacity * deltaKeyWidth,
                                stride: MemoryLayout<Float>.stride),
            value: try makeBuffer(batchCapacity * deltaValueWidth,
                                  stride: MemoryLayout<Float>.stride),
            decay: try makeBuffer(batchCapacity * config.linearNumValueHeads,
                                  stride: MemoryLayout<Float>.stride),
            beta: try makeBuffer(batchCapacity * config.linearNumValueHeads,
                                 stride: MemoryLayout<Float>.stride),
            recurrent: try makeBuffer(batchCapacity * deltaValueWidth,
                                      stride: MemoryLayout<Float>.stride),
            normalized: try makeBuffer(batchCapacity * deltaValueWidth,
                                       stride: MemoryLayout<Float>.stride))
        layerZeroDeltaRecurrentCapture = try makeBuffer(
            deltaValueWidth, stride: MemoryLayout<Float>.stride)
        layerZeroDeltaNormalizedCapture = try makeBuffer(
            deltaValueWidth, stride: MemoryLayout<Float>.stride)
        attentionHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(
                batchCapacity * hyperWidth, stride: MemoryLayout<Float>.stride),
            lowRank: try makeBuffer(batchCapacity * lowRank),
            activatedLowRank: try makeBuffer(batchCapacity * lowRank),
            mixLogits: try makeBuffer(batchCapacity * hyperWidth),
            injectionLogits: try makeBuffer(batchCapacity * streamCount),
            injectionWeights: try makeBuffer(batchCapacity * streamCount))
        mlpHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(
                batchCapacity * hyperWidth, stride: MemoryLayout<Float>.stride),
            lowRank: try makeBuffer(batchCapacity * lowRank),
            activatedLowRank: try makeBuffer(batchCapacity * lowRank),
            mixLogits: try makeBuffer(batchCapacity * hyperWidth),
            injectionLogits: try makeBuffer(batchCapacity * streamCount),
            injectionWeights: try makeBuffer(batchCapacity * streamCount))
        finalHyperConnection = Qwen38HyperConnectionScratch(
            normalized: try makeBuffer(
                batchCapacity * hyperWidth, stride: MemoryLayout<Float>.stride),
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
        mtpLogits = try makeBuffer(config.vocabSize)
        ple = Qwen38PLEScratch(
            projectedKey: try makeBuffer(batchCapacity * hyperWidth),
            value: try makeBuffer(batchCapacity * hiddenSize),
            normalizedKey: try makeBuffer(
                batchCapacity * hyperWidth, stride: MemoryLayout<Float>.stride),
            normalizedQuery: try makeBuffer(
                batchCapacity * hyperWidth, stride: MemoryLayout<Float>.stride),
            gatedValue: try makeBuffer(batchCapacity * hyperWidth),
            normalizedGatedValue: try makeBuffer(
                batchCapacity * hyperWidth, stride: MemoryLayout<Float>.stride),
            convolution: try makeBuffer(batchCapacity * hyperWidth))
    }
}

public enum Qwen38SemanticValidity: String, Codable, Sendable, Equatable {
    case valid
    case emptyTokenIDs = "invalid-empty-token-ids"
    case allZeroTokenIDs = "invalid-all-zero-token-ids"

    public static func from(tokenIDs: [Int32]) -> Self {
        guard !tokenIDs.isEmpty else {
            return .emptyTokenIDs
        }
        return tokenIDs.allSatisfy { $0 == 0 } ? .allZeroTokenIDs : .valid
    }
}

public struct Qwen38MemoryDiagnostics: Codable, Sendable, Equatable {
    public let residentPayloadBytes: UInt64
    public let residentMappedBytes: UInt64
    public let residentMappingOverheadBytes: UInt64
    public let expertSlotBytes: UInt64
    public let expertSlotCapacityBytes: UInt64
    public let denseCacheCurrentBytes: UInt64
    public let denseCachePeakBytes: UInt64
    public let denseCacheCapacityBytes: UInt64
    public let denseCacheHits: UInt64
    public let denseCacheMisses: UInt64
    public let denseCacheEvictions: UInt64
    public let denseCacheBypasses: UInt64
    public let ngramCurrentBytes: Int
    public let ngramPeakBytes: Int
    public let ngramCacheCapacityBytes: Int
    public let ngramPinnedBytes: Int
    public let ngramPinnedRowCount: Int
    public let ngramPinnedRowByteBudget: Int
    public let managedCurrentBytes: UInt64
    public let managedCapacityBytes: UInt64

    init(model: Model, ngram: NgramCacheDiagnostics) {
        let modelMemory = model.memoryDiagnostics
        self.residentPayloadBytes = modelMemory.residentPayloadBytes
        self.residentMappedBytes = modelMemory.residentMappedBytes
        self.residentMappingOverheadBytes = modelMemory.residentMappedBytes
            >= modelMemory.residentPayloadBytes
            ? modelMemory.residentMappedBytes - modelMemory.residentPayloadBytes
            : 0
        self.expertSlotBytes = modelMemory.expertSlotBytes
        self.expertSlotCapacityBytes = modelMemory.expertSlotCapacityBytes
        self.denseCacheCurrentBytes = modelMemory.denseCacheCurrentBytes
        self.denseCachePeakBytes = modelMemory.denseCachePeakBytes
        self.denseCacheCapacityBytes = modelMemory.denseCacheCapacityBytes
        self.denseCacheHits = modelMemory.denseCacheHits
        self.denseCacheMisses = modelMemory.denseCacheMisses
        self.denseCacheEvictions = modelMemory.denseCacheEvictions
        self.denseCacheBypasses = modelMemory.denseCacheBypasses
        self.ngramCurrentBytes = ngram.currentBytes
        self.ngramPeakBytes = ngram.peakBytes
        self.ngramCacheCapacityBytes = ngram.cacheCapacityBytes
        self.ngramPinnedBytes = ngram.pinnedBytes
        self.ngramPinnedRowCount = ngram.pinnedRowCount
        self.ngramPinnedRowByteBudget = ngram.pinnedRowByteBudget
        let resident = modelMemory.residentMappedBytes
        let currentExperts = modelMemory.expertSlotBytes
        let capacityExperts = modelMemory.expertSlotCapacityBytes
        self.managedCurrentBytes = resident + currentExperts
            + modelMemory.denseCacheCurrentBytes + UInt64(ngram.currentBytes)
        self.managedCapacityBytes = resident + capacityExperts
            + modelMemory.denseCacheCapacityBytes
            + UInt64(ngram.cacheCapacityBytes) + UInt64(ngram.pinnedRowByteBudget)
    }
}

public final class Qwen38ForwardRunner: ForwardRunner, ContinuableLogitProducer,
    PromptStateSnapshotting, ChunkedPrefillRunner, GreedyBlockVerifyingLogitProducer,
    DraftingLogitProducer, @unchecked Sendable {
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
    private let mtp: Qwen38MTP?
    private let mtpInputFusion: Qwen38MTPInputFusion?
    private let mtpInputFusionWeights: Qwen38MTPInputFusionWeights?
    private let mtpInputFusionScratch: Qwen38MTPInputFusionScratch?
    private let mtpState: Qwen38MTPState?
    private let mtpDraftExecutor: Qwen38MTPDraftExecutor?
    private let layers: [Qwen38DecoderLayerWeights]
    private let moeWeights: [Qwen38MoEWeights]
    private let scratch: Qwen38RunnerScratch
    private let finalHyperConnection: Qwen38HyperConnectionWeights
    private let pleLayer: Int
    private let pleWeights: Qwen38PLEWeights
    private let pleAddressing: Qwen38PLEAddressing
    private let ngramStreamer: PreadNgramStreamer
    private let ngramReadConcurrency: Int
    private let deltaNetGPUStageTimer: QwenGPUStageTimer?
    private let qwenGPUExecutionMode: QwenGPUExecutionMode
    private let enableMTPDiagnostics: Bool
    private var deltaNetProjectionQueues: (MTLCommandQueue, MTLCommandQueue)?

    public let maxContext: Int
    public let targetLayerCount: Int
    public let draftingStrategy: Qwen38DraftingStrategy
    public private(set) var continuationPosition = 0
    public private(set) var lastLogitDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastFinalHiddenDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastPreFinalMixerDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastEmbeddingDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastFirstLayerDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerZeroAttentionOutputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerZeroAfterAttentionDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerZeroMLPInputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerZeroMLPOutputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneOutputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneNgramEmbeddingDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOnePLEProjectedKeyDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOnePLEValueDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOnePLEOutputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneAttentionInputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneQKVDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneRecurrentDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneNormalizedDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneAttentionOutputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneAfterAttentionDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneMLPInputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastLayerOneMLPOutputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastFinalLayerInputDiagnostics: Qwen38LogitDiagnostics?
    public private(set) var lastRouterDiagnostics: Qwen38RouterDiagnostics?
    public private(set) var lastDecodeTiming: Qwen38DecodeTimingSample?
    public private(set) var lastSpeculativeReplay = Qwen38SpeculativeReplaySample.zero
    public private(set) var lastNativeDraftToken: Int32?
    public private(set) var lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
        strategy: .disabled)

    public var ngramCacheDiagnostics: NgramCacheDiagnostics {
        ngramStreamer.cacheDiagnostics
    }

    public var ngramRowProfile: [NgramRowProfileEntry] {
        ngramStreamer.rowProfileSnapshot
    }

    public var memoryDiagnostics: Qwen38MemoryDiagnostics {
        Qwen38MemoryDiagnostics(model: model, ngram: ngramStreamer.cacheDiagnostics)
    }

    public var mtpExecutionCapability: Qwen38MTPExecutionCapability {
        mtp?.executionCapability ?? .unavailable
    }

    public var usesTargetOnlyFallback: Bool {
        draftingStrategy == .disabled || !mtpExecutionCapability.supportsNativeDraftGeneration
    }

    public var draftingDiagnostics: DraftingDiagnosticsAggregate {
        DraftingDiagnosticsAggregate(
            strategy: draftingStrategy.rawValue,
            draftAttempts: draftAttempts,
            proposedTokens: proposedTokens,
            acceptedTokens: acceptedTokens,
            rejectedTokens: rejectedTokens,
            fallbackCount: fallbackCount,
            fallbackReason: fallbackReason,
            lastProposedToken: lastDraftingDiagnostics.proposedToken,
            lastTargetToken: lastDraftingDiagnostics.targetToken,
            lastMatchesTarget: lastDraftingDiagnostics.matchesTarget,
            lastInputToken: lastDraftingDiagnostics.inputToken,
            lastProposalPosition: lastDraftingDiagnostics.proposalPosition,
            lastTargetPosition: lastDraftingDiagnostics.targetPosition)
    }

    public var mtpStatePosition: Int? {
        mtpState?.position
    }

    public func targetBoundarySnapshot(
        maxPayloadBytes: Int = 1_048_576) throws -> Qwen38TargetBoundarySnapshot {
        guard enableMTPDiagnostics else {
            throw Qwen38TargetBoundarySnapshotError.diagnosticsDisabled
        }
        guard let hiddenStreams = lastTargetHiddenStreams,
              let inputToken = lastTargetInputToken,
              continuationPosition > 0 else {
            throw Qwen38TargetBoundarySnapshotError.targetBoundaryUnavailable
        }
        let streamCount = 4
        let elementCount = streamCount * config.hiddenSize
        let streamByteCount = elementCount * MemoryLayout<Float16>.stride
        let normalizedStreamByteCount = elementCount * MemoryLayout<Float>.stride
        let rawStreams = lastTargetRawHiddenStreams
        let stagePayloadByteCount = lastStageCaptures.reduce(0) {
            $0 + $1.values.count * MemoryLayout<Float16>.stride
        }
        let payloadByteCount = streamByteCount
            + (rawStreams == nil ? 0 : streamByteCount)
            + stagePayloadByteCount
          guard hiddenStreams.length >= normalizedStreamByteCount,
              rawStreams.map({ $0.length >= streamByteCount }) ?? true else {
            throw Qwen38TargetBoundarySnapshotError.targetBoundaryUnavailable
        }
        guard payloadByteCount <= maxPayloadBytes else {
            throw Qwen38TargetBoundarySnapshotError.payloadExceedsLimit(
                byteCount: payloadByteCount,
                limit: maxPayloadBytes)
        }
        let rawValues = rawStreams.map { buffer in
            Array(UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: Float16.self),
                count: elementCount))
        }
        let normalizedValues = Array(UnsafeBufferPointer(
            start: hiddenStreams.contents().assumingMemoryBound(to: Float.self),
            count: elementCount))
        return Qwen38TargetBoundarySnapshot(
            targetPosition: continuationPosition - 1,
            inputToken: inputToken,
            streamCount: streamCount,
            hiddenSize: config.hiddenSize,
            targetHiddenStreams: normalizedValues.map(Float16.init),
            targetHiddenFloat32Streams: normalizedValues,
            rawTargetHiddenStreams: rawValues,
            stageCaptures: lastStageCaptures,
            routerDiagnostics: lastRouterDiagnostics)
    }

    private var ngramContext: [Int64] = []
    private var promptStateSnapshot: Qwen38PromptStateSnapshot?
    private var pendingCommandBuffer: MTLCommandBuffer?
    private var pendingDeltaNetMarkerEncoded = false
    private var completedDeltaNetNanos: UInt64 = 0
    private var commandBufferEncodeNanos: UInt64 = 0
    private var commandBufferWaitNanos: UInt64 = 0
    private var draftCandidateConsumed = true
    private var draftAttempts = 0
    private var proposedTokens = 0
    private var acceptedTokens = 0
    private var rejectedTokens = 0
    private var fallbackCount = 0
    private var fallbackReason: String?
    private var suppressDraftingDiagnostics = false
    private var prefillMTPEmbeddingToken: Int32?
    private var forceFreshMTPInput = true
    private var lastTargetHiddenStreams: MTLBuffer?
    private var lastTargetRawHiddenStreams: MTLBuffer?
    private var lastStageCaptures: [Qwen38StageCapture] = []
    private var lastTargetInputToken: Int32?
    private var lastMTPPrimeSnapshot: Qwen38MTPStateSnapshot?
    private var lastMTPPrimeHiddenStreams: MTLBuffer?
    private var lastMTPPrimeInputToken: Int32?
    private var lastMTPPrimeOutputToken: Int32?

    static func resolveTargetLayerCount(requested: Int?,
                                        modelLayerCount: Int,
                                        pleLayer: Int) -> Int? {
        let resolved = requested ?? modelLayerCount
        guard resolved >= pleLayer + 1,
              resolved <= modelLayerCount else {
            return nil
        }
        return resolved
    }

    static func routerSelectionRequiresWait(for mode: QwenGPUExecutionMode) -> Bool {
        mode == .parallelDeltaProjections
    }

    public init(model: Model,
                context: MetalContext,
                maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production,
                enableMTPDiagnostics: Bool = false,
                mtpFCOrientation: Qwen38MTPFCOrientation = .normal,
                draftingStrategy: Qwen38DraftingStrategy = .disabled,
                targetLayerCount: Int? = nil) throws {
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
        guard let targetLayerCount = Self.resolveTargetLayerCount(
            requested: targetLayerCount,
            modelLayerCount: model.config.numLayers,
            pleLayer: pleLayer) else {
            throw ModelError.archMismatch(
                field: "targetLayerCount",
                expected: "between \(pleLayer + 1) and \(model.config.numLayers)",
                actual: "\(targetLayerCount ?? -1)")
        }
        let pleWeights = try Qwen38PLEWeights(model: model, layer: pleLayer)
        let ngramStreamer = try model.qwen38NgramStreamer(
            rowCacheBytes: runtimeConfiguration.ngramRowCacheBytes,
            rowCacheMaxUniqueRows: runtimeConfiguration.ngramRowCacheMaxUniqueRows,
            rowProfileMaxRows: runtimeConfiguration.ngramRowProfileMaxRows,
            pinnedRows: runtimeConfiguration.ngramPinnedRows,
            pinnedRowByteBudget: runtimeConfiguration.ngramPinnedRowBytes)
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
        let embeddingWeightBits = model.embedding.quantization?.bits ?? 4
        let lmHeadWeightBits = model.lmHead.quantization?.bits ?? 4

        self.model = model
        self.context = context
        self.config = model.config
        self.maxContext = maxContext
        self.targetLayerCount = targetLayerCount
        self.draftingStrategy = draftingStrategy
        self.ngramReadConcurrency = runtimeConfiguration.ngramReadConcurrency
        self.qwenGPUExecutionMode = runtimeConfiguration.qwenGPUExecutionMode
        self.enableMTPDiagnostics = enableMTPDiagnostics
        self.deltaNetProjectionQueues = nil
        self.embed = try EmbedLookupInt4(
            context: context,
            groupSize: Quantization.qwen38GroupSize,
            weightBits: embeddingWeightBits)
        self.prefillEmbed = try PrefillEmbedLookupInt4(
            context: context,
            groupSize: Quantization.qwen38GroupSize,
            weightBits: embeddingWeightBits)
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
            groupSize: Quantization.qwen38GroupSize,
            weightBits: lmHeadWeightBits)
        self.runtimeState = try Qwen38RuntimeState(model: model, maxContext: maxContext)
        let mtp = model.hasMTP ? try Qwen38MTP(model: model) : nil
        self.mtp = mtp
        if let mtp {
            self.mtpInputFusion = try Qwen38MTPInputFusion(
                context: context,
                fcOrientation: mtpFCOrientation)
            self.mtpInputFusionWeights = try Qwen38MTPInputFusionWeights(mtp: mtp)
            self.mtpInputFusionScratch = try Qwen38MTPInputFusionScratch(
                device: context.device)
        } else {
            self.mtpInputFusion = nil
            self.mtpInputFusionWeights = nil
            self.mtpInputFusionScratch = nil
        }
        self.mtpState = model.hasMTP
            ? try Qwen38MTPState(model: model, maxContext: maxContext)
            : nil
        self.mtpDraftExecutor = model.hasMTP
            ? try Qwen38MTPDraftExecutor(
                model: model,
                context: context,
                maxContext: maxContext,
                diagnosticMode: enableMTPDiagnostics ? .on : .off,
                fcOrientation: mtpFCOrientation)
            : nil
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
        self.deltaNetGPUStageTimer = QwenGPUStageTimer(device: context.device)
        self.lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
            strategy: draftingStrategy)

        _ = model.embedding
        _ = model.lmHead
    }

    public func reset() {
        continuationPosition = 0
        lastLogitDiagnostics = nil
        lastFinalHiddenDiagnostics = nil
        lastPreFinalMixerDiagnostics = nil
        lastEmbeddingDiagnostics = nil
        lastFirstLayerDiagnostics = nil
        lastLayerZeroAttentionOutputDiagnostics = nil
        lastLayerZeroAfterAttentionDiagnostics = nil
        lastLayerZeroMLPInputDiagnostics = nil
        lastLayerZeroMLPOutputDiagnostics = nil
        lastLayerOneOutputDiagnostics = nil
        lastLayerOneNgramEmbeddingDiagnostics = nil
        lastLayerOnePLEProjectedKeyDiagnostics = nil
        lastLayerOnePLEValueDiagnostics = nil
        lastLayerOnePLEOutputDiagnostics = nil
        lastLayerOneAttentionInputDiagnostics = nil
        lastLayerOneQKVDiagnostics = nil
        lastLayerOneRecurrentDiagnostics = nil
        lastLayerOneNormalizedDiagnostics = nil
        lastLayerOneAttentionOutputDiagnostics = nil
        lastLayerOneAfterAttentionDiagnostics = nil
        lastLayerOneMLPInputDiagnostics = nil
        lastLayerOneMLPOutputDiagnostics = nil
        lastFinalLayerInputDiagnostics = nil
        lastRouterDiagnostics = nil
        ngramContext.removeAll(keepingCapacity: true)
        promptStateSnapshot = nil
        pendingCommandBuffer = nil
        pendingDeltaNetMarkerEncoded = false
        completedDeltaNetNanos = 0
        commandBufferEncodeNanos = 0
        commandBufferWaitNanos = 0
        draftCandidateConsumed = true
        draftAttempts = 0
        proposedTokens = 0
        acceptedTokens = 0
        rejectedTokens = 0
        fallbackCount = 0
        fallbackReason = nil
        suppressDraftingDiagnostics = false
        forceFreshMTPInput = true
        lastTargetHiddenStreams = nil
        lastTargetRawHiddenStreams = nil
        lastStageCaptures.removeAll(keepingCapacity: true)
        lastTargetInputToken = nil
        lastMTPPrimeSnapshot = nil
        lastMTPPrimeHiddenStreams = nil
        lastMTPPrimeInputToken = nil
        lastMTPPrimeOutputToken = nil
        lastNativeDraftToken = nil
        lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
            strategy: draftingStrategy)
        runtimeState.reset()
        mtpState?.reset()
    }

    public func takeDraftCandidate() -> Int32? {
        guard draftingStrategy.isEnabled, !draftCandidateConsumed else {
            return nil
        }
        draftCandidateConsumed = true
        guard let proposedToken = lastDraftingDiagnostics.proposedToken,
              lastDraftingDiagnostics.matchesTarget == true else {
            return nil
        }
        acceptedTokens += 1
        return proposedToken
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
            runtimeState: runtimeState.snapshot(),
            mtpState: mtpState?.snapshot())
    }

    public func primeNativeMTPState(token: Int32,
                                     freshTargetHiddenStreams: Bool = false) throws -> Int32 {
        guard token >= 0 && token < Int32(config.vocabSize) else {
            throw PrefillError.chunkedUnsupported(
                "MTP priming token must be a valid vocabulary ID")
        }
          guard let mtpDraftExecutor,
              let mtpState,
              let hiddenStreams = lastTargetHiddenStreams else {
            throw ModelError.archMismatch(
                field: "mtp.validationPrime",
                expected: "a completed target decode with MTP resources",
                actual: "missing")
        }
        guard continuationPosition > 0,
              mtpState.position == continuationPosition - 1 else {
            throw PrefillError.prefillCursorMismatch(
                "MTP priming state \(mtpState.position) is not aligned with "
                    + "target position \(continuationPosition - 1)")
        }
        let primeInputStreams = freshTargetHiddenStreams
            ? hiddenStreams
            : (mtpState.feedback ?? hiddenStreams)
        let primeSnapshot = enableMTPDiagnostics ? mtpState.snapshot() : nil
        if enableMTPDiagnostics {
            let byteCount = 4 * config.hiddenSize * MemoryLayout<Float16>.stride
            let primeBuffer: MTLBuffer
            if let existingBuffer = lastMTPPrimeHiddenStreams {
                primeBuffer = existingBuffer
            } else {
                guard let allocatedBuffer = context.device.makeBuffer(
                    length: byteCount,
                    options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                primeBuffer = allocatedBuffer
            }
            primeBuffer.contents().copyMemory(
                from: primeInputStreams.contents(),
                byteCount: byteCount)
            lastMTPPrimeSnapshot = primeSnapshot
            lastMTPPrimeHiddenStreams = primeBuffer
            lastMTPPrimeInputToken = token
            lastMTPPrimeOutputToken = nil
        }
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
        }
        let primeOutputToken = try mtpDraftExecutor.generate(
            embedding: scratch.finalHidden,
            hiddenStreams: primeInputStreams,
            targetFinalHidden: scratch.mixedInput,
            state: mtpState,
            logits: scratch.mtpLogits)
        if enableMTPDiagnostics {
            lastMTPPrimeOutputToken = primeOutputToken
        }
        return primeOutputToken
    }

    public func draftNativeMTPBlock(initialToken: Int32,
                                    tokenCount: Int) throws -> [Int32] {
        guard tokenCount > 0 && tokenCount <= Qwen38MTPDraftBlock.maxTokenCount else {
            throw PrefillError.chunkedUnsupported(
                "MTP draft block supports between 1 and \(Qwen38MTPDraftBlock.maxTokenCount) tokens")
        }
        guard mtpExecutionCapability.supportsNativeDraftGeneration,
              let mtpDraftExecutor,
              let mtpState,
              let hiddenStreams = lastTargetHiddenStreams else {
            throw ModelError.archMismatch(
                field: "mtp.blockDraft",
                expected: "native MTP resources and completed target hidden streams",
                actual: "missing")
        }
        guard mtpState.position == continuationPosition else {
            throw PrefillError.prefillCursorMismatch(
                "MTP block state \(mtpState.position) is not aligned with target position \(continuationPosition)")
        }
        let checkpoint = captureSpeculativeState()
        do {
            let block = try mtpDraftExecutor.generateBlock(
                initialToken: initialToken,
                hiddenStreams: hiddenStreams,
                state: mtpState,
                logits: scratch.mtpLogits,
                tokenCount: tokenCount) { token in
                    try self.runSync { commandBuffer in
                        let embedding = self.model.embedding
                        self.embed.encode(
                            commandBuffer: commandBuffer,
                            table: embedding.buffer,
                            tableOffset: Int(embedding.offset),
                            scales: embedding.buffer,
                            scalesOffset: Int(embedding.scaleOffset),
                            biases: embedding.buffer,
                            biasesOffset: Int(embedding.biasOffset),
                            out: self.scratch.finalHidden,
                            tokenId: UInt32(bitPattern: token),
                            d: UInt32(self.config.hiddenSize),
                            outScale: 1)
                    }
                    return self.scratch.finalHidden
                }
            restoreSpeculativeState(checkpoint)
            return block.tokens
        } catch {
            restoreSpeculativeState(checkpoint)
            throw error
        }
    }

    public func validateNativeMTP(boundaryToken: Int32,
                                  alternateEmbeddingToken: Int32,
                                  into logits: MTLBuffer) async throws
        -> (draftToken: Int32,
            freshTargetDraftToken: Int32,
            rawTargetDraftToken: Int32?,
            alternateDraftToken: Int32,
            targetToken: Int32,
            streamOrderDrafts: [([Int], Int32)]) {
        guard boundaryToken >= 0 && boundaryToken < Int32(config.vocabSize),
              alternateEmbeddingToken >= 0,
              alternateEmbeddingToken < Int32(config.vocabSize) else {
            throw PrefillError.chunkedUnsupported(
                "MTP validation tokens must be valid vocabulary IDs")
        }
        guard let mtpDraftExecutor,
              let mtpState,
              let hiddenStreams = lastTargetHiddenStreams else {
            throw ModelError.archMismatch(
                field: "mtp.validationDraft",
                expected: "a completed target decode with MTP resources",
                actual: "missing")
        }
        guard continuationPosition > 0,
              mtpState.position == continuationPosition - 1 else {
            throw PrefillError.prefillCursorMismatch(
                "MTP validation state \(mtpState.position) is not aligned with "
                    + "target position \(continuationPosition - 1)")
        }
        if enableMTPDiagnostics {
            FileHandle.standardError.write(
                Data("mtp validation_entry\n".utf8))
        }
        let checkpoint = captureSpeculativeState()
        do {
            let generateDraft: (Int32, MTLBuffer, String?) throws -> Int32 = {
                embeddingToken, inputStreams, diagnosticLabel in
                try self.runSync { commandBuffer in
                    let embedding = self.model.embedding
                    self.embed.encode(
                        commandBuffer: commandBuffer,
                        table: embedding.buffer,
                        tableOffset: Int(embedding.offset),
                        scales: embedding.buffer,
                        scalesOffset: Int(embedding.scaleOffset),
                        biases: embedding.buffer,
                        biasesOffset: Int(embedding.biasOffset),
                        out: self.scratch.finalHidden,
                        tokenId: UInt32(bitPattern: embeddingToken),
                        d: UInt32(self.config.hiddenSize),
                        outScale: 1)
                }
                FileHandle.standardError.write(
                    Data("mtp validation_label=\(diagnosticLabel ?? "nil")\n".utf8))
                return try mtpDraftExecutor.generate(
                    embedding: self.scratch.finalHidden,
                    hiddenStreams: inputStreams,
                    targetFinalHidden: self.scratch.mixedInput,
                    state: mtpState,
                    logits: logits,
                    diagnosticLabel: diagnosticLabel)
            }
            let carriedFeedback = mtpState.feedback
            let draftToken = try generateDraft(
                boundaryToken,
                carriedFeedback ?? hiddenStreams,
                "carried")
            restoreSpeculativeState(checkpoint)
            let freshTargetDraftToken = try generateDraft(
                boundaryToken,
                hiddenStreams,
                "fresh_target")
            restoreSpeculativeState(checkpoint)
            let rawTargetDraftToken: Int32?
            if let rawTargetHiddenStreams = lastTargetRawHiddenStreams {
                rawTargetDraftToken = try generateDraft(
                    boundaryToken,
                    rawTargetHiddenStreams,
                    "raw_target")
                restoreSpeculativeState(checkpoint)
            } else {
                rawTargetDraftToken = nil
            }
            if enableMTPDiagnostics {
                let rawTargetDescription = rawTargetDraftToken.map(String.init)
                    ?? "unavailable"
                let diagnostic = "mtp boundary_input target_position=\(continuationPosition - 1) "
                    + "state_position=\(mtpState.position) "
                    + "feedback_present=\(carriedFeedback != nil) "
                    + "carried=\(draftToken) fresh_target=\(freshTargetDraftToken) "
                    + "raw_target=\(rawTargetDescription)"
                FileHandle.standardError.write(Data((diagnostic + "\n").utf8))
            }
            if enableMTPDiagnostics,
               let primeSnapshot = lastMTPPrimeSnapshot,
               let primeHiddenStreams = lastMTPPrimeHiddenStreams,
               let primeInputToken = lastMTPPrimeInputToken,
               let primeOutputToken = lastMTPPrimeOutputToken {
                restoreSpeculativeState(checkpoint)
                mtpState.restore(primeSnapshot)
                let replayedPrimeToken = try generateDraft(
                    primeInputToken,
                    primeHiddenStreams,
                    "prime_replay")
                restoreSpeculativeState(checkpoint)
                let diagnostic = "mtp prime_replay snapshot_position=\(primeSnapshot.position) "
                    + "current_position=\(mtpState.position) "
                    + "expected=\(primeOutputToken) "
                    + "replayed=\(replayedPrimeToken) "
                    + "matches=\(replayedPrimeToken == primeOutputToken)"
                FileHandle.standardError.write(Data((diagnostic + "\n").utf8))
            }
            let alternateDraftToken = try generateDraft(
                alternateEmbeddingToken,
                mtpState.feedback ?? hiddenStreams,
                "alternate_embedding")
            restoreSpeculativeState(checkpoint)

            func permutations(_ values: [Int]) -> [[Int]] {
                guard let first = values.first else { return [[]] }
                return permutations(Array(values.dropFirst())).flatMap { suffix in
                    (0...suffix.count).map { insertionIndex in
                        var result = suffix
                        result.insert(first, at: insertionIndex)
                        return result
                    }
                }
            }
            let streamOrders = permutations([0, 1, 2, 3])
            let permutationDestination = hiddenStreams === scratch.hiddenStreams
                ? scratch.alternateStreams
                : scratch.hiddenStreams
            let streamBytes = config.hiddenSize * MemoryLayout<Float16>.stride
            let streamOrderDrafts = try streamOrders.map { order in
                restoreSpeculativeState(checkpoint)
                let source = hiddenStreams.contents()
                let destination = permutationDestination.contents()
                for (destinationStream, sourceStream) in order.enumerated() {
                    destination
                        .advanced(by: destinationStream * streamBytes)
                        .copyMemory(
                            from: source.advanced(by: sourceStream * streamBytes),
                            byteCount: streamBytes)
                }
                let candidateToken = try generateDraft(
                    boundaryToken,
                    permutationDestination,
                    nil)
                return (order, candidateToken)
            }
            let feedbackStreamOrderDrafts: [([Int], Int32)]?
            if let feedback = mtpState.feedback {
                let feedbackDestination = feedback === scratch.hiddenStreams
                    ? scratch.alternateStreams
                    : scratch.hiddenStreams
                feedbackStreamOrderDrafts = try streamOrders.map { order in
                    restoreSpeculativeState(checkpoint)
                    let source = feedback.contents()
                    let destination = feedbackDestination.contents()
                    for (destinationStream, sourceStream) in order.enumerated() {
                        destination
                            .advanced(by: destinationStream * streamBytes)
                            .copyMemory(
                                from: source.advanced(by: sourceStream * streamBytes),
                                byteCount: streamBytes)
                    }
                    let candidateToken = try generateDraft(
                        boundaryToken,
                        feedbackDestination,
                        nil)
                    return (order, candidateToken)
                }
            } else {
                feedbackStreamOrderDrafts = nil
            }
            let feedbackOrderReceipt = feedbackStreamOrderDrafts?.map { result in
                "\(result.0.map(String.init).joined())=\(result.1)"
            }.joined(separator: ",") ?? "unavailable"
            let diagnostic = "mtp feedback_stream_order_drafts=\(feedbackOrderReceipt)"
            FileHandle.standardError.write(Data((diagnostic + "\n").utf8))
            restoreSpeculativeState(checkpoint)
            try await produce(
                token: boundaryToken,
                position: continuationPosition,
                into: logits)
            let targetToken = greedyToken(from: logits)
            lastNativeDraftToken = draftToken
            restoreSpeculativeState(checkpoint)
            return (
                draftToken,
                freshTargetDraftToken,
                rawTargetDraftToken,
                alternateDraftToken,
                targetToken,
                streamOrderDrafts)
        } catch {
            restoreSpeculativeState(checkpoint)
            throw error
        }
    }

    public func restorePromptState(expectedPosition: Int) throws {
        guard let snapshot = promptStateSnapshot,
              snapshot.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "prompt replay expected position \(expectedPosition) has no matching snapshot")
        }
        runtimeState.restore(snapshot.runtimeState)
        if let mtpSnapshot = snapshot.mtpState {
            mtpState?.restore(mtpSnapshot)
        }
        ngramContext = snapshot.ngramContext
        continuationPosition = snapshot.position
    }

    public func verifyGreedyBlock(boundaryToken: Int32,
                                  proposedTokens: ArraySlice<Int32>,
                                  startPosition: Int,
                                  config runtimeConfig: PrefillRuntimeConfig) async throws
        -> GreedyBlockVerification {
        try await verifyGreedyBlockImpl(
            boundaryToken: boundaryToken,
            proposedTokens: proposedTokens,
            startPosition: startPosition,
            config: runtimeConfig,
            logitsOutput: nil)
    }

    public func verifyGreedyBlock(boundaryToken: Int32,
                                  proposedTokens: ArraySlice<Int32>,
                                  startPosition: Int,
                                  config runtimeConfig: PrefillRuntimeConfig,
                                  into logitsOutput: MTLBuffer) async throws
        -> GreedyBlockVerification {
        try await verifyGreedyBlockImpl(
            boundaryToken: boundaryToken,
            proposedTokens: proposedTokens,
            startPosition: startPosition,
            config: runtimeConfig,
            logitsOutput: logitsOutput)
    }

    private func verifyGreedyBlockImpl(boundaryToken: Int32,
                                       proposedTokens: ArraySlice<Int32>,
                                       startPosition: Int,
                                       config runtimeConfig: PrefillRuntimeConfig,
                                       logitsOutput: MTLBuffer?) async throws
        -> GreedyBlockVerification {
        lastSpeculativeReplay = .zero
        guard runtimeConfig.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "Qwen3.8 MTP verification requires chunked prefill")
        }
        guard startPosition == continuationPosition else {
            throw PrefillError.prefillCursorMismatch(
                "verification start \(startPosition) != current position \(continuationPosition)")
        }
        guard !proposedTokens.isEmpty else {
            throw PrefillError.chunkedUnsupported(
                "Qwen3.8 MTP verification requires at least one proposal")
        }
        guard proposedTokens.count <= runtimeConfig.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "proposal block \(proposedTokens.count) exceeds chunk size \(runtimeConfig.chunkTokens)")
        }
        guard proposedTokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "proposal block exceeds the remaining context window")
        }
        guard boundaryToken >= 0,
              boundaryToken < Int32(config.vocabSize),
              proposedTokens.allSatisfy({ $0 >= 0 && $0 < Int32(config.vocabSize) }) else {
            throw PrefillError.chunkedUnsupported(
                "MTP verification tokens must be valid vocabulary IDs")
        }

        guard let logits = context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        if let logitsOutput {
            guard logitsOutput.length >= logits.length else {
                throw ModelError.residentBufferWrapFailed
            }
        }
        let proposals = Array(proposedTokens)
        let inputTokens = [boundaryToken] + Array(proposals.dropLast())
        let batchLogitLayout = try Qwen38BatchLogitLayout(
            tokenCount: inputTokens.count,
            vocabularySize: config.vocabSize)
        guard let logitsRows = context.device.makeBuffer(
            length: batchLogitLayout.byteCount,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let checkpoint = captureSpeculativeState()
        do {
            _ = try await produceBatch(
                tokens: inputTokens[...],
                startPosition: startPosition,
                into: logits,
                logitsRows: logitsRows)
            var targetTokens: [Int32] = []
            targetTokens.reserveCapacity(inputTokens.count)
            for tokenIndex in inputTokens.indices {
                targetTokens.append(greedyToken(
                    from: logitsRows,
                    byteOffset: tokenIndex * batchLogitLayout.rowByteStride))
            }
            let verification = GreedyBlockVerification(
                targetTokens: targetTokens,
                proposedTokens: proposals,
                startPosition: startPosition)
            if verification.acceptedTokenCount < proposals.count {
                let replayStart = DispatchTime.now().uptimeNanoseconds
                restoreSpeculativeState(checkpoint)
                let replayCount = verification.acceptedTokenCount + 1
                for token in inputTokens.prefix(replayCount) {
                    try await produce(
                        token: token,
                        position: continuationPosition,
                        into: logits)
                }
                lastSpeculativeReplay = Qwen38SpeculativeReplaySample(
                    replayedTokenCount: replayCount,
                    replayNanos: DispatchTime.now().uptimeNanoseconds - replayStart)
            }
            guard continuationPosition == verification.statePosition else {
                throw PrefillError.prefillCursorMismatch(
                    "verification resolved position \(continuationPosition) != expected \(verification.statePosition)")
            }
            if let logitsOutput {
                logitsOutput.contents().copyMemory(
                    from: logits.contents(),
                    byteCount: logits.length)
            }
            return verification
        } catch {
            restoreSpeculativeState(checkpoint)
            throw error
        }
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

        if draftingStrategy.isEnabled,
           lastNativeDraftToken != nil,
           lastDraftingDiagnostics.matchesTarget == false {
            forceFreshMTPInput = true
        }

        lastDecodeTiming = nil
        lastRouterDiagnostics = nil
        lastStageCaptures.removeAll(keepingCapacity: true)
        commandBufferEncodeNanos = 0
        commandBufferWaitNanos = 0
        var embeddingNanos: UInt64 = 0
        let pleNanos: UInt64 = 0
        var attentionRouterNanos: UInt64 = 0
        var expertFetchNanos: UInt64 = 0
        var expertCacheHits = 0
        var expertCacheMisses = 0
        var moeNanos: UInt64 = 0
        var deltaNetNanos: UInt64 = 0
        var gpuActiveNanos: UInt64 = 0
        var commandBufferCount = 0
        var nextNgramContext = ngramContext
        let addresses = pleAddressing.addresses(
            tokens: [Int64(token)], context: &nextNgramContext)
        let ngramTask = Task {
            try await ngramStreamer.readAsync(
                addresses: addresses[0], maxConcurrentReads: ngramReadConcurrency)
        }

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
        if enableMTPDiagnostics {
            lastEmbeddingDiagnostics = diagnostics(
                buffer: scratch.finalHidden,
                offset: 0,
                count: config.hiddenSize)
            try captureOwnedStage(
                layerIndex: -1,
                stage: "embedding",
                streamCount: 1,
                buffer: scratch.finalHidden,
                position: position,
                inputToken: token)
        }
        gpuActiveNanos += embeddingGPUActiveNanos
        commandBufferCount += 1

        var inputStreams = scratch.hiddenStreams
        var outputStreams = scratch.alternateStreams
        let initialFrontStart = DispatchTime.now().uptimeNanoseconds
        if Self.routerSelectionRequiresWait(for: qwenGPUExecutionMode) {
            let timing = try runParallelLayerInput(
                layer: 0,
                position: position,
                inputStreams: inputStreams,
                outputStreams: outputStreams)
            gpuActiveNanos += timing.gpuActiveNanos
            deltaNetNanos += timing.deltaNetGPUActiveNanos
            commandBufferCount += timing.commandBufferCount
        } else {
            let initialFrontGPUActiveNanos = try runSync { commandBuffer in
                try encodeLayerInput(
                    commandBuffer: commandBuffer,
                    layer: 0,
                    position: position,
                    inputStreams: inputStreams,
                    outputStreams: outputStreams)
            }
            gpuActiveNanos += initialFrontGPUActiveNanos
            deltaNetNanos += consumeCompletedDeltaNetNanos()
            commandBufferCount += 1
        }
        attentionRouterNanos += DispatchTime.now().uptimeNanoseconds - initialFrontStart
        var finalHeadNanos: UInt64 = 0
        for layer in 0..<targetLayerCount {
            try Task.checkCancellation()
            if layer > 0 {
                gpuActiveNanos += try waitPending()
                deltaNetNanos += consumeCompletedDeltaNetNanos()
                if enableMTPDiagnostics && layer == pleLayer {
                    let pleEmbeddingSize = config.qwen38Architecture?.pleEmbeddingSize
                        ?? config.hiddenSize
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-ngram-embedding",
                        streamCount: 1,
                        elementCount: pleEmbeddingSize,
                        shape: [1, pleEmbeddingSize],
                        buffer: scratch.ngramEmbedding,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-projected-key",
                        buffer: scratch.ple.projectedKey,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-value",
                        streamCount: 1,
                        buffer: scratch.ple.value,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-normalized-key",
                        buffer: scratch.ple.normalizedKey,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-normalized-query",
                        buffer: scratch.ple.normalizedQuery,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-gated-value",
                        buffer: scratch.ple.gatedValue,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-normalized-gated-value",
                        buffer: scratch.ple.normalizedGatedValue,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-convolution",
                        buffer: scratch.ple.convolution,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple_layer_1",
                        streamCount: 4,
                        buffer: outputStreams,
                        position: position,
                        inputToken: token)
                }
                if enableMTPDiagnostics && layer == 1 {
                    captureRouterDiagnostics(layerIndex: 0)
                    try captureOwnedStage(
                        layerIndex: 0,
                        buffer: scratch.layerZeroOutputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-hyper-normalized",
                        buffer: scratch.layerZeroHyperNormalizedCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-hyper-low-rank",
                        streamCount: 1,
                        elementCount: 320,
                        shape: [1, 320],
                        buffer: scratch.layerZeroHyperLowRankCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-hyper-activated-low-rank",
                        streamCount: 1,
                        elementCount: 320,
                        shape: [1, 320],
                        buffer: scratch.layerZeroHyperActivatedLowRankCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-hyper-mix-logits",
                        buffer: scratch.layerZeroHyperMixLogitsCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-hyper-input",
                        buffer: scratch.layerZeroHyperInputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-attention-input",
                        streamCount: 1,
                        buffer: scratch.layerZeroAttentionInputCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-attention-output",
                        streamCount: 1,
                        buffer: scratch.layerZeroAttentionOutputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-after-attention",
                        buffer: scratch.layerZeroAfterAttentionCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-mlp-input",
                        streamCount: 1,
                        buffer: scratch.layerZeroMLPInputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-routed-phase1-activation",
                        elementCount: Qwen38MoE.topK * config.moeIntermediateSize,
                        shape: [Qwen38MoE.topK, config.moeIntermediateSize],
                        buffer: moe.diagnosticActivationBuffer(slot: 0),
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-shared-output",
                        streamCount: 1,
                        buffer: scratch.sharedOutput,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-mlp-output",
                        streamCount: 1,
                        buffer: scratch.layerZeroMLPOutputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-mlp-hyper-normalized",
                        buffer: scratch.layerZeroMLPHyperNormalizedCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-mlp-hyper-mix-logits",
                        buffer: scratch.layerZeroMLPHyperMixLogitsCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-delta-recurrent",
                        streamCount: 1,
                        elementCount: Int(config.linearNumValueHeads
                            * config.linearValueHeadDim),
                        shape: [1, Int(config.linearNumValueHeads
                            * config.linearValueHeadDim)],
                        buffer: scratch.layerZeroDeltaRecurrentCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-delta-qkv",
                        streamCount: 1,
                        elementCount: Int(config.linearNumKeyHeads
                            * config.linearKeyHeadDim * 2
                            + config.linearNumValueHeads
                            * config.linearValueHeadDim),
                        shape: [1, Int(config.linearNumKeyHeads
                            * config.linearKeyHeadDim * 2
                            + config.linearNumValueHeads
                            * config.linearValueHeadDim)],
                        buffer: scratch.layerZeroDeltaQKVCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 0,
                        stage: "layer-0-delta-normalized",
                        streamCount: 1,
                        elementCount: Int(config.linearNumValueHeads
                            * config.linearValueHeadDim),
                        shape: [1, Int(config.linearNumValueHeads
                            * config.linearValueHeadDim)],
                        buffer: scratch.layerZeroDeltaNormalizedCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    lastLayerZeroAttentionOutputDiagnostics = diagnostics(
                        buffer: scratch.layerZeroAttentionOutputCapture,
                        offset: 0,
                        count: config.hiddenSize)
                    lastLayerZeroAfterAttentionDiagnostics = diagnostics(
                        buffer: scratch.layerZeroAfterAttentionCapture,
                        offset: 0,
                        count: config.hiddenSize * 4)
                    lastLayerZeroMLPInputDiagnostics = diagnostics(
                        buffer: scratch.layerZeroMLPInputCapture,
                        offset: 0,
                        count: config.hiddenSize)
                    lastLayerZeroMLPOutputDiagnostics = diagnostics(
                        buffer: scratch.layerZeroMLPOutputCapture,
                        offset: 0,
                        count: config.hiddenSize)
                } else if enableMTPDiagnostics && layer == 2 {
                    try captureOwnedStage(
                        layerIndex: 1,
                        buffer: scratch.layerOneOutputCapture,
                        position: position,
                        inputToken: token)
                }
                if enableMTPDiagnostics && layer == 1 {
                    lastFirstLayerDiagnostics = diagnostics(
                        buffer: inputStreams,
                        offset: 0,
                        count: config.hiddenSize * 4)
                }
                if enableMTPDiagnostics && layer == 2 {
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-attention-input",
                        streamCount: 1,
                        buffer: scratch.layerOneAttentionInputCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-attn-hyper-normalized",
                        buffer: scratch.layerOneHyperNormalizedCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-attn-hyper-low-rank",
                        streamCount: 1,
                        elementCount: 320,
                        shape: [1, 320],
                        buffer: scratch.layerOneHyperLowRankCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-attn-hyper-activated-low-rank",
                        streamCount: 1,
                        elementCount: 320,
                        shape: [1, 320],
                        buffer: scratch.layerOneHyperActivatedLowRankCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-attn-hyper-mix-logits",
                        buffer: scratch.layerOneHyperMixLogitsCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-delta-qkv",
                        streamCount: 1,
                        elementCount: Int(config.linearNumKeyHeads
                            * config.linearKeyHeadDim * 2
                            + config.linearNumValueHeads
                            * config.linearValueHeadDim),
                        shape: [1, Int(config.linearNumKeyHeads
                            * config.linearKeyHeadDim * 2
                            + config.linearNumValueHeads
                            * config.linearValueHeadDim)],
                        buffer: scratch.layerOneDeltaQKVCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-delta-recurrent",
                        streamCount: 1,
                        elementCount: Int(config.linearNumValueHeads
                            * config.linearValueHeadDim),
                        shape: [1, Int(config.linearNumValueHeads
                            * config.linearValueHeadDim)],
                        buffer: scratch.layerOneDeltaRecurrentCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-delta-normalized",
                        streamCount: 1,
                        elementCount: Int(config.linearNumValueHeads
                            * config.linearValueHeadDim),
                        shape: [1, Int(config.linearNumValueHeads
                            * config.linearValueHeadDim)],
                        buffer: scratch.layerOneDeltaNormalizedCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-attention-output",
                        streamCount: 1,
                        buffer: scratch.layerOneAttentionOutputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-after-attention",
                        buffer: scratch.layerOneAfterAttentionCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-mlp-hyper-normalized",
                        buffer: scratch.layerOneMLPHyperNormalizedCapture,
                        position: position,
                        inputToken: token,
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-mlp-hyper-mix-logits",
                        buffer: scratch.layerOneMLPHyperMixLogitsCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-mlp-input",
                        streamCount: 1,
                        buffer: scratch.layerOneMLPInputCapture,
                        position: position,
                        inputToken: token)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "layer-1-mlp-output",
                        streamCount: 1,
                        buffer: scratch.layerOneMLPOutputCapture,
                        position: position,
                        inputToken: token)
                    lastLayerOneQKVDiagnostics = diagnostics(
                        buffer: scratch.delta.qkv,
                        offset: 0,
                        count: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                            + config.linearNumValueHeads * config.linearValueHeadDim)
                    lastLayerOneRecurrentDiagnostics = diagnostics(
                        buffer: scratch.delta.recurrent,
                        offset: 0,
                        count: config.linearNumValueHeads * config.linearValueHeadDim)
                    lastLayerOneNormalizedDiagnostics = diagnostics(
                        buffer: scratch.delta.normalized,
                        offset: 0,
                        count: config.linearNumValueHeads * config.linearValueHeadDim)
                    lastLayerOneAttentionOutputDiagnostics = diagnostics(
                        buffer: scratch.attentionOutput,
                        offset: 0,
                        count: config.hiddenSize)
                    lastLayerOneAfterAttentionDiagnostics = diagnostics(
                        buffer: scratch.afterAttention,
                        offset: 0,
                        count: config.hiddenSize * 4)
                    lastLayerOneMLPInputDiagnostics = diagnostics(
                        buffer: scratch.mixedInput,
                        offset: 0,
                        count: config.hiddenSize)
                    lastLayerOneMLPOutputDiagnostics = diagnostics(
                        buffer: scratch.mlpOutput,
                        offset: 0,
                        count: config.hiddenSize)
                    lastLayerOneOutputDiagnostics = diagnostics(
                        buffer: inputStreams,
                        offset: 0,
                        count: config.hiddenSize * 4)
                }
                if enableMTPDiagnostics && layer == targetLayerCount - 1 {
                    lastFinalLayerInputDiagnostics = diagnostics(
                        buffer: inputStreams,
                        offset: 0,
                        count: config.hiddenSize * 4)
                }
                if Self.routerSelectionRequiresWait(for: qwenGPUExecutionMode) {
                    let timing = try runParallelLayerInput(
                        layer: layer,
                        position: position,
                        inputStreams: inputStreams,
                        outputStreams: outputStreams)
                    gpuActiveNanos += timing.gpuActiveNanos
                    deltaNetNanos += timing.deltaNetGPUActiveNanos
                    commandBufferCount += timing.commandBufferCount
                }
            }
            if layer + 1 == pleLayer {
                let ngramRows = try await ngramTask.value
                try writeNgramEmbedding(addresses: addresses[0], rows: ngramRows)
            }
            if Self.routerSelectionRequiresWait(for: qwenGPUExecutionMode) {
                gpuActiveNanos += try waitPending()
                deltaNetNanos += consumeCompletedDeltaNetNanos()
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
            let isLastLayer = layer == targetLayerCount - 1
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
                if enableMTPDiagnostics && layer < 2 {
                    guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let capture = layer == 0
                        ? scratch.layerZeroOutputCapture
                        : scratch.layerOneOutputCapture
                    blit.copy(
                        from: outputStreams,
                        sourceOffset: 0,
                        to: capture,
                        destinationOffset: 0,
                        size: config.hiddenSize * 4 * MemoryLayout<Float16>.stride)
                    if layer == 0 {
                        blit.copy(
                            from: scratch.attentionOutput,
                            sourceOffset: 0,
                            to: scratch.layerZeroAttentionOutputCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.afterAttention,
                            sourceOffset: 0,
                            to: scratch.layerZeroAfterAttentionCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4 * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.mixedInput,
                            sourceOffset: 0,
                            to: scratch.layerZeroMLPInputCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.mlpOutput,
                            sourceOffset: 0,
                            to: scratch.layerZeroMLPOutputCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * MemoryLayout<Float16>.stride)
                        let deltaKeyBytes = Int(config.linearNumKeyHeads
                            * config.linearKeyHeadDim)
                            * MemoryLayout<Float>.stride
                        let deltaValueBytes = Int(config.linearNumValueHeads
                            * config.linearValueHeadDim)
                            * MemoryLayout<Float>.stride
                        blit.copy(
                            from: scratch.delta.query,
                            sourceOffset: 0,
                            to: scratch.layerZeroDeltaQKVCapture,
                            destinationOffset: 0,
                            size: deltaKeyBytes)
                        blit.copy(
                            from: scratch.delta.key,
                            sourceOffset: 0,
                            to: scratch.layerZeroDeltaQKVCapture,
                            destinationOffset: deltaKeyBytes,
                            size: deltaKeyBytes)
                        blit.copy(
                            from: scratch.delta.value,
                            sourceOffset: 0,
                            to: scratch.layerZeroDeltaQKVCapture,
                            destinationOffset: deltaKeyBytes * 2,
                            size: deltaValueBytes)
                        blit.copy(
                            from: scratch.delta.recurrent,
                            sourceOffset: 0,
                            to: scratch.layerZeroDeltaRecurrentCapture,
                            destinationOffset: 0,
                            size: Int(config.linearNumValueHeads
                                * config.linearValueHeadDim)
                                * MemoryLayout<Float>.stride)
                        blit.copy(
                            from: scratch.delta.normalized,
                            sourceOffset: 0,
                            to: scratch.layerZeroDeltaNormalizedCapture,
                            destinationOffset: 0,
                            size: Int(config.linearNumValueHeads
                                * config.linearValueHeadDim)
                                * MemoryLayout<Float>.stride)
                    } else {
                        blit.copy(
                            from: scratch.attentionHyperConnection.normalized,
                            sourceOffset: 0,
                            to: scratch.layerOneHyperNormalizedCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4
                                * MemoryLayout<Float>.stride)
                        blit.copy(
                            from: scratch.attentionHyperConnection.lowRank,
                            sourceOffset: 0,
                            to: scratch.layerOneHyperLowRankCapture,
                            destinationOffset: 0,
                            size: 320 * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.attentionHyperConnection.activatedLowRank,
                            sourceOffset: 0,
                            to: scratch.layerOneHyperActivatedLowRankCapture,
                            destinationOffset: 0,
                            size: 320 * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.attentionHyperConnection.mixLogits,
                            sourceOffset: 0,
                            to: scratch.layerOneHyperMixLogitsCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4
                                * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.attentionOutput,
                            sourceOffset: 0,
                            to: scratch.layerOneAttentionOutputCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.afterAttention,
                            sourceOffset: 0,
                            to: scratch.layerOneAfterAttentionCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4
                                * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.mlpHyperConnection.normalized,
                            sourceOffset: 0,
                            to: scratch.layerOneMLPHyperNormalizedCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4
                                * MemoryLayout<Float>.stride)
                        blit.copy(
                            from: scratch.mlpHyperConnection.mixLogits,
                            sourceOffset: 0,
                            to: scratch.layerOneMLPHyperMixLogitsCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4
                                * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.mixedInput,
                            sourceOffset: 0,
                            to: scratch.layerOneMLPInputCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.mlpOutput,
                            sourceOffset: 0,
                            to: scratch.layerOneMLPOutputCapture,
                            destinationOffset: 0,
                            size: config.hiddenSize * MemoryLayout<Float16>.stride)
                        blit.copy(
                            from: scratch.delta.qkv,
                            sourceOffset: 0,
                            to: scratch.layerOneDeltaQKVCapture,
                            destinationOffset: 0,
                            size: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                                * MemoryLayout<Float>.stride
                                + config.linearNumValueHeads * config.linearValueHeadDim
                                * MemoryLayout<Float>.stride)
                        blit.copy(
                            from: scratch.delta.recurrent,
                            sourceOffset: 0,
                            to: scratch.layerOneDeltaRecurrentCapture,
                            destinationOffset: 0,
                            size: config.linearNumValueHeads * config.linearValueHeadDim
                                * MemoryLayout<Float>.stride)
                        blit.copy(
                            from: scratch.delta.normalized,
                            sourceOffset: 0,
                            to: scratch.layerOneDeltaNormalizedCapture,
                            destinationOffset: 0,
                            size: config.linearNumValueHeads * config.linearValueHeadDim
                                * MemoryLayout<Float>.stride)
                    }
                    blit.endEncoding()
                }
                if isLastLayer {
                    try encodeMTPInputFusion(
                        commandBuffer: commandBuffer,
                        embedding: scratch.finalHidden,
                        hiddenStreams: outputStreams)
                    if enableMTPDiagnostics,
                       let blit = commandBuffer.makeBlitCommandEncoder() {
                        blit.copy(
                            from: outputStreams,
                            sourceOffset: 0,
                            to: scratch.rawTargetHiddenStreams,
                            destinationOffset: 0,
                            size: config.hiddenSize * 4 * MemoryLayout<Float16>.stride)
                        blit.endEncoding()
                    }
                    decoderFinalPrepare(
                        commandBuffer: commandBuffer,
                        input: outputStreams)
                    guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
                    blit.copy(
                        from: scratch.mixedInput,
                        sourceOffset: 0,
                        to: scratch.finalHidden,
                        destinationOffset: 0,
                        size: config.hiddenSize * MemoryLayout<Float16>.stride)
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
                } else if qwenGPUExecutionMode == .ordered {
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
                deltaNetNanos += consumeCompletedDeltaNetNanos()
                if enableMTPDiagnostics {
                    if layer == 1 {
                        try captureOwnedStage(
                            layerIndex: 1,
                            stage: "layer-1-routed-phase1-activation",
                            elementCount: Qwen38MoE.topK * config.moeIntermediateSize,
                            shape: [Qwen38MoE.topK, config.moeIntermediateSize],
                            buffer: moe.diagnosticActivationBuffer(slot: 1),
                            position: position,
                            inputToken: token)
                        try captureOwnedStage(
                            layerIndex: 1,
                            stage: "layer-1-shared-output",
                            streamCount: 1,
                            buffer: scratch.sharedOutput,
                            position: position,
                            inputToken: token)
                        try captureOwnedStage(
                            layerIndex: 1,
                            stage: "layer-1-mlp-output",
                            streamCount: 1,
                            buffer: scratch.layerOneMLPOutputCapture,
                            position: position,
                            inputToken: token)
                        try captureOwnedStage(
                            layerIndex: 1,
                            stage: "layer_1",
                            buffer: scratch.layerOneOutputCapture,
                            position: position,
                            inputToken: token)
                        try captureOwnedStage(
                            layerIndex: 1,
                            stage: "final_hidden",
                            streamCount: 1,
                            buffer: scratch.finalHidden,
                            position: position,
                            inputToken: token)
                    }
                    lastPreFinalMixerDiagnostics = diagnostics(
                        buffer: outputStreams,
                        offset: 0,
                        count: config.hiddenSize * 4)
                    lastFinalHiddenDiagnostics = diagnostics(
                        buffer: scratch.mixedInput,
                        offset: 0,
                        count: config.hiddenSize)
                    lastLogitDiagnostics = diagnostics(
                        buffer: logits,
                        offset: 0,
                        count: config.vocabSize)
                }
                finalHeadNanos = DispatchTime.now().uptimeNanoseconds - finalHeadStart
            } else {
                swap(&inputStreams, &outputStreams)
            }
        }

        lastTargetHiddenStreams = scratch.finalHyperConnection.normalized
        lastTargetRawHiddenStreams = enableMTPDiagnostics
            ? scratch.rawTargetHiddenStreams
            : nil
        lastTargetInputToken = token
        let pendingDraftToken = lastNativeDraftToken
        lastNativeDraftToken = nil
        draftCandidateConsumed = true
        lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
            strategy: draftingStrategy)
        if draftingStrategy.isEnabled && suppressDraftingDiagnostics {
            guard let mtpDraftExecutor, let mtpState,
                  mtpState.position == position else {
                throw PrefillError.prefillCursorMismatch(
                    "MTP priming state is not aligned with target position")
            }
            let mtpCheckpoint = mtpState.snapshot()
            do {
                let embeddingToken = prefillMTPEmbeddingToken ?? greedyToken(from: logits)
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
                        tokenId: UInt32(bitPattern: embeddingToken),
                        d: UInt32(config.hiddenSize),
                        outScale: 1)
                }
                let mtpInputStreams = prefillMTPEmbeddingToken == nil
                    ? (lastTargetHiddenStreams ?? outputStreams)
                    : (mtpState.feedback ?? lastTargetHiddenStreams ?? outputStreams)
                let proposedToken = try mtpDraftExecutor.generate(
                    embedding: scratch.finalHidden,
                    hiddenStreams: mtpInputStreams,
                    targetFinalHidden: scratch.mixedInput,
                    state: mtpState,
                    logits: scratch.mtpLogits)
                lastNativeDraftToken = proposedToken
            } catch {
                mtpState.restore(mtpCheckpoint)
                throw error
            }
        } else if draftingStrategy.isEnabled {
            let targetToken = greedyToken(from: logits)
            if let pendingDraftToken {
                let matchesTarget = pendingDraftToken == targetToken
                if !matchesTarget {
                    rejectedTokens += 1
                    fallbackCount += 1
                    fallbackReason = fallbackReason ?? "proposal-mismatch"
                }
                lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
                    strategy: draftingStrategy,
                    proposedToken: pendingDraftToken,
                    targetToken: targetToken,
                    matchesTarget: matchesTarget,
                    inputToken: token,
                    proposalPosition: position + 1,
                    targetPosition: position + 1)
            } else {
                lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
                    strategy: draftingStrategy,
                    targetToken: targetToken)
            }
            let encodeTargetEmbedding: () throws -> Void = {
                _ = try self.runSync { commandBuffer in
                    let embedding = self.model.embedding
                    self.embed.encode(
                        commandBuffer: commandBuffer,
                        table: embedding.buffer,
                        tableOffset: Int(embedding.offset),
                        scales: embedding.buffer,
                        scalesOffset: Int(embedding.scaleOffset),
                        biases: embedding.buffer,
                        biasesOffset: Int(embedding.biasOffset),
                        out: self.scratch.finalHidden,
                        tokenId: UInt32(bitPattern: targetToken),
                        d: UInt32(self.config.hiddenSize),
                        outScale: 1)
                }
            }
            if let mtpDraftExecutor, let mtpState {
                if mtpState.position == position {
                    let mtpCheckpoint = mtpState.snapshot()
                    draftAttempts += 1
                    do {
                        try encodeTargetEmbedding()
                        let mtpInputStreams = forceFreshMTPInput
                            ? (lastTargetHiddenStreams ?? outputStreams)
                            : (mtpState.feedback ?? lastTargetHiddenStreams ?? outputStreams)
                        let proposedToken = try mtpDraftExecutor.generate(
                            embedding: scratch.finalHidden,
                            hiddenStreams: mtpInputStreams,
                            targetFinalHidden: scratch.mixedInput,
                            state: mtpState,
                            logits: scratch.mtpLogits)
                        proposedTokens += 1
                        lastNativeDraftToken = proposedToken
                        forceFreshMTPInput = false
                    } catch {
                        mtpState.restore(mtpCheckpoint)
                        fallbackCount += 1
                        fallbackReason = fallbackReason ?? String(describing: error)
                        lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
                            strategy: draftingStrategy,
                            targetToken: targetToken,
                            fallbackReason: String(describing: error))
                    }
                } else {
                    fallbackCount += 1
                    fallbackReason = fallbackReason ?? "MTP state is not aligned with target position"
                    lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
                        strategy: draftingStrategy,
                        targetToken: targetToken,
                        fallbackReason: "MTP state is not aligned with target position")
                }
            } else {
                fallbackCount += 1
                fallbackReason = fallbackReason ?? "native MTP resources unavailable"
                lastDraftingDiagnostics = Qwen38DraftingDiagnostics(
                    strategy: draftingStrategy,
                    targetToken: targetToken,
                    fallbackReason: "native MTP resources unavailable")
            }
            draftCandidateConsumed = false
        }

        lastDecodeTiming = Qwen38DecodeTimingSample(
            embeddingNanos: embeddingNanos,
            pleNanos: pleNanos,
            attentionRouterNanos: attentionRouterNanos,
            deltaNetNanos: deltaNetNanos,
            expertFetchNanos: expertFetchNanos,
            expertCacheHits: expertCacheHits,
            expertCacheMisses: expertCacheMisses,
            moeNanos: moeNanos,
            finalHeadNanos: finalHeadNanos,
            gpuActiveNanos: gpuActiveNanos,
            commandBufferCount: commandBufferCount,
            commandBufferEncodeNanos: commandBufferEncodeNanos,
            commandBufferWaitNanos: commandBufferWaitNanos)
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
        if draftingStrategy.isEnabled {
            suppressDraftingDiagnostics = true
            defer {
                suppressDraftingDiagnostics = false
                prefillMTPEmbeddingToken = nil
                forceFreshMTPInput = true
            }
            for (index, token) in tokens.enumerated() {
                let nextToken: Int32? = index + 1 < tokens.count
                    ? tokens[tokens.index(tokens.startIndex, offsetBy: index + 1)]
                    : nil
                prefillMTPEmbeddingToken = nextToken
                try await produce(
                    token: token,
                    position: continuationPosition,
                    into: logits)
                onProgress(continuationPosition)
            }
            return PrefillResult(newPosition: continuationPosition,
                                 seed: .logitsWritten)
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
                              into logits: MTLBuffer,
                              logitsRows: MTLBuffer? = nil) async throws
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
        let batchLogitLayout = try logitsRows.map { _ in
            try Qwen38BatchLogitLayout(
                tokenCount: tokenCount,
                vocabularySize: config.vocabSize)
        }
        if let logitsRows, let batchLogitLayout {
            guard logitsRows.length >= batchLogitLayout.byteCount else {
                throw ModelError.residentBufferWrapFailed
            }
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
        let ngramTask = Task {
            var rowsByToken: [[[Float]]] = []
            rowsByToken.reserveCapacity(addressRows.count)
            for addresses in addressRows {
                rowsByToken.append(
                    try await ngramStreamer.readAsync(
                        addresses: addresses, maxConcurrentReads: ngramReadConcurrency))
            }
            return rowsByToken
        }
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

        let embeddingDispatchStart = commandBufferTiming()
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
        if enableMTPDiagnostics {
            let rowOffset = (tokenCount - 1) * config.hiddenSize
                * MemoryLayout<Float16>.stride
            lastEmbeddingDiagnostics = diagnostics(
                buffer: scratch.finalHidden,
                offset: rowOffset,
                count: config.hiddenSize)
        }
        let embeddingDispatchEnd = commandBufferTiming()
        workCounter.recordCommandBufferTimings(
            encode: embeddingDispatchEnd.encode - embeddingDispatchStart.encode,
            wait: embeddingDispatchEnd.wait - embeddingDispatchStart.wait)
        workCounter.recordStageTimings(
            embedding: DispatchTime.now().uptimeNanoseconds - embeddingStart)
        workCounter.recordChunkPass()
        workCounter.recordCommandBuffers(1)

        var inputStreams = scratch.hiddenStreams
        var outputStreams = scratch.alternateStreams
        for layer in 0..<targetLayerCount {
            try Task.checkCancellation()
            if layer == pleLayer {
                let ngramRows = try await ngramTask.value
                try writeNgramEmbeddings(
                    addressRows: addressRows, rowsByToken: ngramRows)
                if enableMTPDiagnostics {
                    let embeddingSize = config.qwen38Architecture?.pleEmbeddingSize ?? 0
                    let embeddingOffset = (tokenCount - 1) * embeddingSize
                        * MemoryLayout<Float16>.stride
                    lastLayerOneNgramEmbeddingDiagnostics = diagnostics(
                        buffer: scratch.ngramEmbedding,
                        offset: embeddingOffset,
                        count: embeddingSize)
                }
                let pleStart = DispatchTime.now().uptimeNanoseconds
                let pleDispatchStart = commandBufferTiming()
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
                if enableMTPDiagnostics && layer == pleLayer {
                    let streamOffset = (tokenCount - 1) * config.hiddenSize * 4
                        * MemoryLayout<Float16>.stride
                    let embeddingOffset = (tokenCount - 1)
                        * (config.qwen38Architecture?.pleEmbeddingSize ?? 0)
                        * MemoryLayout<Float16>.stride
                    lastLayerOnePLEProjectedKeyDiagnostics = diagnostics(
                        buffer: scratch.ple.projectedKey,
                        offset: streamOffset,
                        count: config.hiddenSize * 4)
                    lastLayerOnePLEValueDiagnostics = diagnostics(
                        buffer: scratch.ple.value,
                        offset: embeddingOffset,
                        count: config.hiddenSize)
                    lastLayerOnePLEOutputDiagnostics = diagnostics(
                        buffer: outputStreams,
                        offset: streamOffset,
                        count: config.hiddenSize * 4)
                    let pleEmbeddingSize = config.qwen38Architecture?.pleEmbeddingSize
                        ?? config.hiddenSize
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-ngram-embedding",
                        streamCount: 1,
                        elementCount: pleEmbeddingSize,
                        shape: [1, pleEmbeddingSize],
                        buffer: scratch.ngramEmbedding,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)])
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-projected-key",
                        buffer: scratch.ple.projectedKey,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)])
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-value",
                        streamCount: 1,
                        buffer: scratch.ple.value,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)])
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-normalized-key",
                        buffer: scratch.ple.normalizedKey,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)],
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-normalized-query",
                        buffer: scratch.ple.normalizedQuery,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)],
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-gated-value",
                        buffer: scratch.ple.gatedValue,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)])
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-normalized-gated-value",
                        buffer: scratch.ple.normalizedGatedValue,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)],
                        float32: true)
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple-convolution",
                        buffer: scratch.ple.convolution,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)])
                    try captureOwnedStage(
                        layerIndex: 1,
                        stage: "ple_layer_1",
                        streamCount: 4,
                        buffer: outputStreams,
                        position: tokenCount - 1,
                        inputToken: tokens[tokens.index(
                            tokens.startIndex, offsetBy: tokenCount - 1)])
                }
                let pleDispatchEnd = commandBufferTiming()
                workCounter.recordCommandBufferTimings(
                    encode: pleDispatchEnd.encode - pleDispatchStart.encode,
                    wait: pleDispatchEnd.wait - pleDispatchStart.wait)
                workCounter.recordStageTimings(
                    mixer: DispatchTime.now().uptimeNanoseconds - pleStart)
                workCounter.recordChunkPass()
                workCounter.recordCommandBuffers(1)
                swap(&inputStreams, &outputStreams)
            }
            if enableMTPDiagnostics && layer == targetLayerCount - 1 {
                lastFinalLayerInputDiagnostics = diagnostics(
                    buffer: inputStreams,
                    offset: 0,
                    count: config.hiddenSize * 4)
            }
            let layerTiming = try await encodeLayerBatch(
                layer: layer,
                startPosition: startPosition,
                inputStreams: inputStreams,
                outputStreams: outputStreams,
                tokenCount: UInt32(tokenCount))
            if enableMTPDiagnostics && layer == 0 {
                captureRouterDiagnostics(layerIndex: 0)
            }
            if enableMTPDiagnostics && layer == 1 {
                let streamOffset = (tokenCount - 1) * config.hiddenSize * 4
                    * MemoryLayout<Float16>.stride
                let hiddenOffset = (tokenCount - 1) * config.hiddenSize
                    * MemoryLayout<Float16>.stride
                let qkvOffset = (tokenCount - 1)
                    * (config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                        + config.linearNumValueHeads * config.linearValueHeadDim)
                    * MemoryLayout<Float16>.stride
                let valueOffset = (tokenCount - 1)
                    * config.linearNumValueHeads * config.linearValueHeadDim
                    * MemoryLayout<Float16>.stride
                lastLayerOneQKVDiagnostics = diagnostics(
                    buffer: scratch.delta.qkv,
                    offset: qkvOffset,
                    count: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                        + config.linearNumValueHeads * config.linearValueHeadDim)
                lastLayerOneRecurrentDiagnostics = diagnostics(
                    buffer: scratch.delta.recurrent,
                    offset: valueOffset,
                    count: config.linearNumValueHeads * config.linearValueHeadDim)
                lastLayerOneNormalizedDiagnostics = diagnostics(
                    buffer: scratch.delta.normalized,
                    offset: valueOffset,
                    count: config.linearNumValueHeads * config.linearValueHeadDim)
                lastLayerOneAttentionOutputDiagnostics = diagnostics(
                    buffer: scratch.attentionOutput,
                    offset: hiddenOffset,
                    count: config.hiddenSize)
                lastLayerOneAfterAttentionDiagnostics = diagnostics(
                    buffer: scratch.afterAttention,
                    offset: streamOffset,
                    count: config.hiddenSize * 4)
                lastLayerOneMLPInputDiagnostics = diagnostics(
                    buffer: scratch.mixedInput,
                    offset: hiddenOffset,
                    count: config.hiddenSize)
                lastLayerOneMLPOutputDiagnostics = diagnostics(
                    buffer: scratch.mlpOutput,
                    offset: hiddenOffset,
                    count: config.hiddenSize)
                lastLayerOneOutputDiagnostics = diagnostics(
                    buffer: outputStreams,
                    offset: streamOffset,
                    count: config.hiddenSize * 4)
            }
            workCounter.recordCommandBufferTimings(
                encode: layerTiming.commandBufferEncodeNanos,
                wait: layerTiming.commandBufferWaitNanos)
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
            if enableMTPDiagnostics && layer == 0 {
                lastFirstLayerDiagnostics = diagnostics(
                    buffer: outputStreams,
                    offset: 0,
                    count: config.hiddenSize * 4)
            }
            swap(&inputStreams, &outputStreams)
        }
        let finalHeadStart = DispatchTime.now().uptimeNanoseconds
        let finalHeadDispatchStart = commandBufferTiming()
        try runSync { commandBuffer in
            decoderFinalPrepare(
                commandBuffer: commandBuffer,
                input: inputStreams,
                tokenCount: UInt32(tokenCount))
            let lmHead = model.lmHead
            let finalHeadInput = Self.finalHeadInput(
                logitsRows: logitsRows,
                finalHidden: scratch.finalHidden,
                mixedInput: scratch.mixedInput)
            if let logitsRows, let batchLogitLayout {
                let hiddenRowBytes = config.hiddenSize * MemoryLayout<Float16>.stride
                for tokenIndex in 0..<tokenCount {
                    head.encode(
                        commandBuffer: commandBuffer,
                        weights: lmHead.buffer,
                        weightsOffset: Int(lmHead.offset),
                        scales: lmHead.buffer,
                        scalesOffset: Int(lmHead.scaleOffset),
                        biases: lmHead.buffer,
                        biasesOffset: Int(lmHead.biasOffset),
                        hidden: finalHeadInput,
                        hiddenOffset: tokenIndex * hiddenRowBytes,
                        logits: logitsRows,
                        logitsOffset: tokenIndex * batchLogitLayout.rowByteStride)
                }
                guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
                blit.copy(
                    from: logitsRows,
                    sourceOffset: (tokenCount - 1) * batchLogitLayout.rowByteStride,
                    to: logits,
                    destinationOffset: 0,
                    size: batchLogitLayout.rowByteStride)
                blit.endEncoding()
            } else {
                guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
                let rowBytes = config.hiddenSize * MemoryLayout<Float16>.stride
                blit.copy(
                    from: scratch.mixedInput,
                    sourceOffset: (tokenCount - 1) * rowBytes,
                    to: scratch.finalHidden,
                    destinationOffset: 0,
                    size: rowBytes)
                blit.endEncoding()
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
        }
        if enableMTPDiagnostics {
            let rowOffset = (tokenCount - 1) * config.hiddenSize
                * MemoryLayout<Float16>.stride
            let streamOffset = (tokenCount - 1) * config.hiddenSize * 4
                * MemoryLayout<Float16>.stride
            lastPreFinalMixerDiagnostics = diagnostics(
                buffer: inputStreams,
                offset: streamOffset,
                count: config.hiddenSize * 4)
            lastFinalHiddenDiagnostics = diagnostics(
                buffer: scratch.mixedInput,
                offset: rowOffset,
                count: config.hiddenSize)
            lastLogitDiagnostics = diagnostics(
                buffer: logits,
                offset: 0,
                count: config.vocabSize)
        }
        let finalHeadDispatchEnd = commandBufferTiming()
        workCounter.recordCommandBufferTimings(
            encode: finalHeadDispatchEnd.encode - finalHeadDispatchStart.encode,
            wait: finalHeadDispatchEnd.wait - finalHeadDispatchStart.wait)
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
            attentionInput: scratch.attentionInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
        let mixerStart = DispatchTime.now().uptimeNanoseconds
        let layerDispatchStart = commandBufferTiming()
        try runSync { commandBuffer in
            decoder.encodeAttentionPrepare(
                commandBuffer: commandBuffer,
                weights: layers[layer],
                hyperInput: inputStreams,
                scratch: layerScratch,
                tokenCount: tokenCount,
                epsilon: 1e-6)
            if enableMTPDiagnostics && layer == 1 {
                guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                    throw ModelError.residentBufferWrapFailed
                }
                let rowBytes = config.hiddenSize * MemoryLayout<Float>.stride
                blit.copy(
                    from: scratch.attentionInput,
                    sourceOffset: (Int(tokenCount) - 1) * rowBytes,
                    to: scratch.layerOneAttentionInputCapture,
                    destinationOffset: 0,
                    size: rowBytes)
                blit.endEncoding()
            }
            let state = try decoder.attentionState(
                layer: layer, runtimeState: runtimeState)
            try encodeAttentionBatch(
                commandBuffer: commandBuffer,
                layer: layer,
                state: state,
                input: scratch.attentionInput,
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
            if enableMTPDiagnostics && layer == 0 {
                moe.encodeDiagnosticSnapshot(
                    commandBuffer: commandBuffer,
                    tokenIndex: Int(tokenCount) - 1)
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
                    routedResources: result.views.map { $0.buffer },
                    hiddenSize: UInt32(config.hiddenSize),
                    intermediateSize: UInt32(config.moeIntermediateSize),
                    sharedExpertGateWeight: moeWeights[layer].sharedExpertGateWeight,
                    tokenIndex: result.tokenIndex,
                    captureDiagnostics: enableMTPDiagnostics && layer == 0)
            }
            decoder.encodeMLPInject(
                commandBuffer: commandBuffer,
                scratch: layerScratch,
                output: outputStreams,
                tokenCount: tokenCount)
        }
        let layerDispatchEnd = commandBufferTiming()
        return Qwen38LayerPrefillTiming(
            mixerNanos: mixerNanos,
            expertFetchNanos: expertFetchNanos,
            commandBufferEncodeNanos: layerDispatchEnd.encode - layerDispatchStart.encode,
            commandBufferWaitNanos: layerDispatchEnd.wait - layerDispatchStart.wait,
            routedMoENanos: DispatchTime.now().uptimeNanoseconds - routedMoEStart,
            cacheHits: cacheHits,
            cacheMisses: cacheMisses,
            expertReadCount: expertReadCount,
            expertReadNanos: expertReadNanos,
            expertReadMaxNanos: expertReadMaxNanos)
    }

    private struct Qwen38ParallelLayerInputTiming {
        let gpuActiveNanos: UInt64
        let deltaNetGPUActiveNanos: UInt64
        let commandBufferCount: Int
    }

    private func runParallelLayerInput(layer: Int,
                                       position: Int,
                                       inputStreams: MTLBuffer,
                                       outputStreams: MTLBuffer)
        throws -> Qwen38ParallelLayerInputTiming {
        guard qwenGPUExecutionMode == .parallelDeltaProjections else {
            throw NSError(
                domain: "TurboFieldfareQwenGPUExecution",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "parallel DeltaNet path requires parallel-delta-projections mode"])
        }
        let queues: (MTLCommandQueue, MTLCommandQueue)
        if let existingQueues = deltaNetProjectionQueues {
            queues = existingQueues
        } else {
            guard let secondQueue = context.device.makeCommandQueue() else {
                throw MetalError.noQueue
            }
            let createdQueues = (context.queue, secondQueue)
            deltaNetProjectionQueues = createdQueues
            queues = createdQueues
        }
        let layerScratch = Qwen38DecoderLayerScratch(
            attentionHyperConnection: scratch.attentionHyperConnection,
            attentionInput: scratch.attentionInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
        let effectiveInput = layer == pleLayer ? outputStreams : inputStreams
        let state = try decoder.attentionState(
            layer: layer,
            runtimeState: runtimeState)
        guard case .linear = state else {
            let gpuActiveNanos = try runSync { commandBuffer in
                try encodeLayerInput(
                    commandBuffer: commandBuffer,
                    layer: layer,
                    position: position,
                    inputStreams: inputStreams,
                    outputStreams: outputStreams)
            }
            return Qwen38ParallelLayerInputTiming(
                gpuActiveNanos: gpuActiveNanos,
                deltaNetGPUActiveNanos: 0,
                commandBufferCount: 1)
        }
        let preparationGPUActiveNanos = try runSync { commandBuffer in
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
            }
            decoder.encodeAttentionPrepare(
                commandBuffer: commandBuffer,
                weights: layers[layer],
                hyperInput: effectiveInput,
                scratch: layerScratch,
                tokenCount: 1,
                epsilon: 1e-6)
        }
        if enableMTPDiagnostics && layer == 1 {
            lastLayerOnePLEOutputDiagnostics = diagnostics(
                buffer: outputStreams,
                offset: 0,
                count: config.hiddenSize * 4)
            lastLayerOneAttentionInputDiagnostics = diagnosticsFloat(
                buffer: scratch.attentionInput,
                offset: 0,
                count: config.hiddenSize)
        }
        let weights = try Qwen38DeltaNetWeights(model: model, layer: layer)
        guard let firstProjection = queues.0.makeCommandBuffer(),
              let secondProjection = queues.1.makeCommandBuffer() else {
            throw NSError(
                domain: "TurboFieldfareQwenGPUExecution",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "failed to allocate parallel DeltaNet projection command buffers"])
        }
        deltaNet.encodeProjectionGroup(
            commandBuffer: firstProjection,
            group: .qkvGate,
            weights: weights,
            input: scratch.attentionInput,
            scratch: scratch.delta,
            inputIsFloat: true)
        deltaNet.encodeProjectionGroup(
            commandBuffer: secondProjection,
            group: .betaDecay,
            weights: weights,
            input: scratch.attentionInput,
            scratch: scratch.delta,
            inputIsFloat: true)
        firstProjection.commit()
        secondProjection.commit()
        firstProjection.waitUntilCompleted()
        secondProjection.waitUntilCompleted()
        try checkCompleted(firstProjection)
        try checkCompleted(secondProjection)
        let firstProjectionNanos = gpuDurationNanos(firstProjection)
        let secondProjectionNanos = gpuDurationNanos(secondProjection)
        try runAsync { commandBuffer in
            try deltaNet.encodeAfterProjections(
                commandBuffer: commandBuffer,
                state: state,
                weights: weights,
                scratch: scratch.delta,
                output: scratch.attentionOutput,
                tokenCount: 1,
                epsilon: 1e-6,
                inputIsFloat: true)
            decoder.encodeAttentionInject(
                commandBuffer: commandBuffer,
                hyperInput: effectiveInput,
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
            if enableMTPDiagnostics && layer == 0 {
                moe.encodeDiagnosticSnapshot(commandBuffer: commandBuffer)
            }
        }
        return Qwen38ParallelLayerInputTiming(
            gpuActiveNanos: preparationGPUActiveNanos
                + firstProjectionNanos + secondProjectionNanos,
            deltaNetGPUActiveNanos: firstProjectionNanos + secondProjectionNanos,
            commandBufferCount: 4)
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
            attentionInput: scratch.attentionInput,
            attentionOutput: scratch.attentionOutput,
            afterAttention: scratch.afterAttention,
            mlpHyperConnection: scratch.mlpHyperConnection,
            mlpInput: scratch.mixedInput,
            mlpOutput: scratch.mlpOutput)
        if enableMTPDiagnostics && layer == 0 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(
                from: inputStreams,
                sourceOffset: 0,
                to: scratch.layerZeroHyperInputCapture,
                destinationOffset: 0,
                size: config.hiddenSize * 4 * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }
        decoder.encodeAttentionPrepare(
            commandBuffer: commandBuffer,
            weights: layers[layer],
            hyperInput: inputStreams,
            scratch: layerScratch,
            tokenCount: 1,
            epsilon: 1e-6)
        if enableMTPDiagnostics && layer == 1 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(
                from: scratch.attentionInput,
                sourceOffset: 0,
                to: scratch.layerOneAttentionInputCapture,
                destinationOffset: 0,
                size: config.hiddenSize * MemoryLayout<Float>.stride)
            blit.endEncoding()
        }
        if enableMTPDiagnostics && layer == 0 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(
                from: scratch.attentionInput,
                sourceOffset: 0,
                to: scratch.layerZeroAttentionInputCapture,
                destinationOffset: 0,
                size: config.hiddenSize * MemoryLayout<Float>.stride)
            blit.endEncoding()
        }
        if enableMTPDiagnostics && layer == 0 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(
                from: scratch.attentionHyperConnection.normalized,
                sourceOffset: 0,
                to: scratch.layerZeroHyperNormalizedCapture,
                destinationOffset: 0,
                size: config.hiddenSize * 4 * MemoryLayout<Float>.stride)
            blit.copy(
                from: scratch.attentionHyperConnection.lowRank,
                sourceOffset: 0,
                to: scratch.layerZeroHyperLowRankCapture,
                destinationOffset: 0,
                size: 320 * MemoryLayout<Float16>.stride)
            blit.copy(
                from: scratch.attentionHyperConnection.activatedLowRank,
                sourceOffset: 0,
                to: scratch.layerZeroHyperActivatedLowRankCapture,
                destinationOffset: 0,
                size: 320 * MemoryLayout<Float16>.stride)
            blit.copy(
                from: scratch.attentionHyperConnection.mixLogits,
                sourceOffset: 0,
                to: scratch.layerZeroHyperMixLogitsCapture,
                destinationOffset: 0,
                size: config.hiddenSize * 4 * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }
        let state = try decoder.attentionState(
            layer: layer,
            runtimeState: runtimeState)
        try encodeAttention(
            commandBuffer: commandBuffer,
            layer: layer,
            state: state,
            input: scratch.attentionInput,
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
        if enableMTPDiagnostics && layer == 0 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(
                from: scratch.mlpHyperConnection.normalized,
                sourceOffset: 0,
                to: scratch.layerZeroMLPHyperNormalizedCapture,
                destinationOffset: 0,
                size: config.hiddenSize * 4 * MemoryLayout<Float>.stride)
            blit.copy(
                from: scratch.mlpHyperConnection.mixLogits,
                sourceOffset: 0,
                to: scratch.layerZeroMLPHyperMixLogitsCapture,
                destinationOffset: 0,
                size: config.hiddenSize * 4 * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }
        moe.encodeRouter(
            commandBuffer: commandBuffer,
            weights: moeWeights[layer],
            hidden: scratch.mixedInput,
            hiddenSize: UInt32(config.hiddenSize))
        moe.encodeSelection(
            commandBuffer: commandBuffer,
            weights: moeWeights[layer])
        if enableMTPDiagnostics && layer < Qwen38MoE.diagnosticSlotCount {
            moe.encodeDiagnosticSnapshot(
                commandBuffer: commandBuffer,
                diagnosticSlot: layer)
        }
    }

    private func encodeLayerMoE(commandBuffer: MTLCommandBuffer,
                                layer: Int,
                                inputStreams: MTLBuffer,
                                outputStreams: MTLBuffer,
                                fetched: Qwen38FetchedMoE) throws {
        let layerScratch = Qwen38DecoderLayerScratch(
            attentionHyperConnection: scratch.attentionHyperConnection,
            attentionInput: scratch.attentionInput,
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
            routedResources: fetched.views.map { $0.buffer },
            hiddenSize: UInt32(config.hiddenSize),
            intermediateSize: UInt32(config.moeIntermediateSize),
            sharedExpertGateWeight: moeWeights[layer].sharedExpertGateWeight,
            captureDiagnostics: enableMTPDiagnostics
                && layer < Qwen38MoE.diagnosticSlotCount,
            diagnosticSlot: min(layer, Qwen38MoE.diagnosticSlotCount - 1))
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
                epsilon: 1e-6,
                inputIsFloat: true)
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
                epsilon: 1e-6,
                centeredWeights: true)
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
            if let deltaNetGPUStageTimer {
                pendingDeltaNetMarkerEncoded = deltaNetGPUStageTimer.encodeMarker(
                    QwenDeltaNetGPUStageMarker.beforeDeltaNet,
                    commandBuffer: commandBuffer)
            }
            try deltaNet.encode(
                commandBuffer: commandBuffer,
                state: state,
                weights: weights,
                input: input,
                scratch: scratch.delta,
                output: output,
                epsilon: 1e-6,
                inputIsFloat: true)
            if let deltaNetGPUStageTimer {
                pendingDeltaNetMarkerEncoded = deltaNetGPUStageTimer.encodeMarker(
                    QwenDeltaNetGPUStageMarker.afterDeltaNet,
                    commandBuffer: commandBuffer)
                    && pendingDeltaNetMarkerEncoded
            }
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
                epsilon: 1e-6,
                centeredWeights: true)
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

    private func writeNgramEmbedding(addresses: [Int64], rows: [[Float]]) throws {
        guard !addresses.isEmpty else {
            throw ModelError.indexCorrupt(detail: "Qwen3.8 PLE produced no n-gram addresses")
        }
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

    private func writeNgramEmbeddings(addressRows: [[Int64]],
                                      rowsByToken: [[[Float]]]) throws {
        guard !addressRows.isEmpty else {
            throw ModelError.indexCorrupt(detail: "Qwen3.8 PLE produced no n-gram addresses")
        }
        guard rowsByToken.count == addressRows.count else {
            throw ModelError.archMismatch(
                field: "packedNgrams.tokenRows",
                expected: "addressRows.count",
                actual: "\(rowsByToken.count)")
        }
        let embeddingSize = config.qwen38Architecture?.pleEmbeddingSize ?? 0
        let output = scratch.ngramEmbedding.contents().assumingMemoryBound(to: UInt16.self)
        for (tokenIndex, addresses) in addressRows.enumerated() {
            guard !addresses.isEmpty else {
                throw ModelError.indexCorrupt(
                    detail: "Qwen3.8 PLE produced no n-gram addresses")
            }
            let rows = rowsByToken[tokenIndex]
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

    private func encodeMTPInputFusion(commandBuffer: MTLCommandBuffer,
                                      embedding: MTLBuffer,
                                      hiddenStreams: MTLBuffer) throws {
        guard mtpExecutionCapability.supportsNativeDraftGeneration else { return }
        guard let mtpInputFusion,
              let mtpInputFusionWeights,
              let mtpInputFusionScratch else {
            throw ModelError.archMismatch(
                field: "mtp.inputFusion",
                expected: "initialized fusion resources",
                actual: "missing")
        }
        mtpInputFusion.encode(
            commandBuffer: commandBuffer,
            embedding: embedding,
            hidden: hiddenStreams,
            weights: mtpInputFusionWeights,
            scratch: mtpInputFusionScratch,
            epsilon: 1e-6)
    }

    static func finalHeadInput(logitsRows: MTLBuffer?,
                               finalHidden: MTLBuffer,
                               mixedInput: MTLBuffer) -> MTLBuffer {
        logitsRows == nil ? finalHidden : mixedInput
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

    private func captureSpeculativeState() -> Qwen38SpeculativeStateCheckpoint {
        Qwen38SpeculativeStateCheckpoint(
            position: continuationPosition,
            ngramContext: ngramContext,
            runtimeState: runtimeState.snapshot(),
            mtpState: mtpState?.snapshot())
    }

    private func restoreSpeculativeState(_ checkpoint: Qwen38SpeculativeStateCheckpoint) {
        runtimeState.restore(checkpoint.runtimeState)
        if let mtpSnapshot = checkpoint.mtpState {
            mtpState?.restore(mtpSnapshot)
        }
        ngramContext = checkpoint.ngramContext
        continuationPosition = checkpoint.position
    }

    private func captureRouterDiagnostics(layerIndex: Int,
                                          diagnosticSlot: Int = 0) {
        lastRouterDiagnostics = Qwen38RouterDiagnostics(
            layerIndex: layerIndex,
            tokenIndex: 0,
            routerLogits: moe.routerLogitValues(diagnosticSlot: diagnosticSlot),
            selectedExperts: moe.diagnosticSelectedExperts(diagnosticSlot: diagnosticSlot),
            routeWeightBits: moe.selectedRouteWeightBits(
                diagnosticSlot: diagnosticSlot),
            sharedGateValue: moe.sharedGateValue(diagnosticSlot: diagnosticSlot))
    }

    private func captureOwnedStage(layerIndex: Int,
                                   stage: String = "layer-output",
                                   streamCount: Int = 4,
                                   elementCount: Int? = nil,
                                   shape: [Int]? = nil,
                                   buffer: MTLBuffer,
                                   position: Int,
                                   inputToken: Int32,
                                   float32: Bool = false) throws {
         guard ["embedding", "layer-output", "layer_0", "ple_layer_1",
             "layer_1", "final_hidden", "layer-0-attention-input",
             "layer-1-attention-input", "layer-1-attn-hyper-normalized",
             "layer-1-attn-hyper-low-rank",
             "layer-1-attn-hyper-activated-low-rank",
             "layer-1-attn-hyper-mix-logits", "layer-0-hyper-input",
             "layer-0-hyper-normalized", "layer-0-hyper-low-rank",
             "layer-0-hyper-activated-low-rank", "layer-0-hyper-mix-logits",
             "layer-0-attention-output", "layer-0-after-attention",
             "layer-0-mlp-input", "layer-0-mlp-output",
             "layer-0-mlp-hyper-normalized", "layer-0-mlp-hyper-mix-logits",
             "layer-0-delta-qkv", "layer-0-delta-normalized", "layer-0-shared-output",
             "layer-0-routed-phase1-activation",
             "layer-1-attention-output", "layer-1-after-attention",
             "layer-1-mlp-hyper-normalized", "layer-1-mlp-hyper-mix-logits",
             "layer-1-delta-qkv", "layer-1-delta-recurrent",
             "layer-1-delta-normalized", "layer-1-mlp-input",
             "layer-1-mlp-output"]
            .contains(stage) else {
            return
        }
        let resolvedElementCount = elementCount ?? streamCount * config.hiddenSize
        let storageStride = float32
            ? MemoryLayout<Float>.stride
            : MemoryLayout<Float16>.stride
        let byteCount = resolvedElementCount * storageStride
        guard buffer.length >= byteCount else {
            throw ModelError.residentBufferWrapFailed
        }
        let values: [Float16]
        let float32Values: [Float]?
        if float32 {
            let exactValues = Array(UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: Float.self),
                count: resolvedElementCount))
            float32Values = exactValues
            values = exactValues.map(Float16.init)
        } else {
            values = Array(UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: Float16.self),
                count: resolvedElementCount))
            float32Values = nil
        }
        lastStageCaptures.append(Qwen38StageCapture(
            layerIndex: layerIndex,
            stage: stage,
            tokenPosition: position,
            inputToken: inputToken,
            shape: shape ?? [streamCount, config.hiddenSize],
            values: values,
            float32Values: float32Values))
    }

    private func diagnostics(buffer: MTLBuffer,
                             offset: Int,
                             count: Int) -> Qwen38LogitDiagnostics {
        let values = buffer.contents()
            .advanced(by: offset)
            .assumingMemoryBound(to: Float16.self)
        return Qwen38LogitDiagnostics(
            values: UnsafeBufferPointer(start: values, count: count))
    }

    private func diagnosticsFloat(buffer: MTLBuffer,
                                  offset: Int,
                                  count: Int) -> Qwen38LogitDiagnostics {
        let values = buffer.contents()
            .advanced(by: offset)
            .assumingMemoryBound(to: Float.self)
        let halfValues = UnsafeBufferPointer(start: values, count: count)
            .map(Float16.init)
        return halfValues.withUnsafeBufferPointer {
            Qwen38LogitDiagnostics(values: $0)
        }
    }

    private func greedyToken(from logits: MTLBuffer,
                             byteOffset: Int = 0) -> Int32 {
        let values = logits.contents()
            .advanced(by: byteOffset)
            .assumingMemoryBound(to: Float16.self)
        var bestIndex = 0
        var bestValue = values[0]
        for index in 1..<config.vocabSize where values[index] > bestValue {
            bestIndex = index
            bestValue = values[index]
        }
        return Int32(bestIndex)
    }

    private func runAsync(_ body: (MTLCommandBuffer) throws -> Void) throws {
        guard pendingCommandBuffer == nil else {
            throw ModelError.residentBufferWrapFailed
        }
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        pendingDeltaNetMarkerEncoded = false
        try body(commandBuffer)
        commandBuffer.commit()
        pendingCommandBuffer = commandBuffer
    }

    private func waitPending() throws -> UInt64 {
        guard let pending = pendingCommandBuffer else { return 0 }
        let markerEncoded = pendingDeltaNetMarkerEncoded
        pendingCommandBuffer = nil
        pendingDeltaNetMarkerEncoded = false
        pending.waitUntilCompleted()
        try checkCompleted(pending)
        if markerEncoded {
            completedDeltaNetNanos += deltaNetGPUStageTimer?.resolveDeltaNet() ?? 0
        }
        return gpuDurationNanos(pending)
    }

    private func commandBufferTiming() -> (encode: UInt64, wait: UInt64) {
        (commandBufferEncodeNanos, commandBufferWaitNanos)
    }

    @discardableResult
    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws -> UInt64 {
        let encodeStart = DispatchTime.now().uptimeNanoseconds
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let pendingMarkerEncoded = pendingDeltaNetMarkerEncoded
        pendingDeltaNetMarkerEncoded = false
        try body(commandBuffer)
        commandBuffer.commit()
        commandBufferEncodeNanos += DispatchTime.now().uptimeNanoseconds - encodeStart

        let waitStart = DispatchTime.now().uptimeNanoseconds
        commandBuffer.waitUntilCompleted()
        let pending = pendingCommandBuffer
        pendingCommandBuffer = nil
        var gpuActiveNanos: UInt64 = 0
        if let pending {
            try checkCompleted(pending)
            gpuActiveNanos += gpuDurationNanos(pending)
        }
        if pendingMarkerEncoded {
            completedDeltaNetNanos += deltaNetGPUStageTimer?.resolveDeltaNet() ?? 0
        }
        try checkCompleted(commandBuffer)
        gpuActiveNanos += gpuDurationNanos(commandBuffer)
        if pendingDeltaNetMarkerEncoded {
            completedDeltaNetNanos += deltaNetGPUStageTimer?.resolveDeltaNet() ?? 0
        }
        pendingDeltaNetMarkerEncoded = false
        commandBufferWaitNanos += DispatchTime.now().uptimeNanoseconds - waitStart
        return gpuActiveNanos
    }

    private func consumeCompletedDeltaNetNanos() -> UInt64 {
        let nanos = completedDeltaNetNanos
        completedDeltaNetNanos = 0
        return nanos
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
