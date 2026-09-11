import Darwin
import Foundation
import Metal
import TurboFieldfare

private let usage = """
Usage:
  TurboFieldfareQwenVerifierProbe \
    --model <model.gturbo> \
    (--prompt <text> | --prompt-sequence-file <path> | \
     --prompt-tokens <id,id,...> --proposed-tokens <id,id,...>) \
    [--max-new-tokens <tokens>] \
    [--repeat-count <count>] \
    [--max-context <tokens>] \
    [--chunk-tokens <32|64|128|256>] \
    [--prefill-mode <batch|scalar>] \
    [--compare-prefill-modes] \
    [--verify-mtp] \
    [--validate-native-mtp] \
    [--validate-native-mtp-boundary] \
    [--mtp-block-size <1..4>] \
    [--mtp-diagnostics] \
    [--fixture-capture] \
    [--target-layer-count <1..48>] \
    [--mtp-fc-orientation <normal|transpose-embedding|transpose-hidden|transpose-both>] \
    [--qwen-gpu-execution <ordered|parallel-delta-projections>] \
    [--ngram-read-concurrency <1|4|8|16>] \
    [--ngram-cache-bytes <0..2147483648>] \
    [--ngram-cache-rows <1..4194304>] \
    [--ngram-profile-rows <0..16384>] \
    [--ngram-pin-rows <address,address,...>] \
    [--ngram-pin-bytes <0..268435456>] \
    [--dense-cache-bytes <1..68719476736> (4 GiB reference)] \
    [--expert-cache-slots <8|16|24|32|64|128> (default 16)] \
    [--expert-cache-policy <lfu|lru>] \
    [--workload-id <id> \
     --workload-category <repetitive-code|editing-continuation|general-chat-novel> \
     --fixture-id <id> --workload-arm <drafting-disabled>]
""" // ചികിത?#+#+#+#+无码不卡高清免费

private enum PrefillMode: String, Sendable {
    case batch
    case scalar
}

private let workloadCategories: Set<String> = [
    "repetitive-code",
    "editing-continuation",
    "general-chat-novel",
]

private enum WorkloadArm: String, Codable {
    case draftingDisabled = "drafting-disabled"
    case draftingEnabled = "drafting-enabled"
}

private struct WorkloadMetadata: Codable {
    let workloadID: String
    let category: String
    let fixtureID: String
    let arm: WorkloadArm
    let draftingEnabled: Bool
    let measurementScope: String
    let qualityScope: String

    init(workloadID: String,
         category: String,
         fixtureID: String,
         arm: WorkloadArm) {
        self.workloadID = workloadID
        self.category = category
        self.fixtureID = fixtureID
        self.arm = arm
        self.draftingEnabled = arm == .draftingEnabled
        self.measurementScope = "qwen38-token-verifier"
        self.qualityScope = "token-id-verifier-only"
    }
}

private enum ArgumentError: Error, CustomStringConvertible {
    case helpRequested
    case invalid(String)

    var description: String {
        switch self {
        case .helpRequested:
            return usage
        case .invalid(let detail):
            return detail
        }
    }
}

private struct Arguments {
    let modelURL: URL
    let prompt: String?
    let promptSequenceURL: URL?
    let promptTokens: [Int32]
    let proposedTokens: [Int32]
    let maxNewTokens: Int
    let repeatCount: Int
    let maxContext: Int
    let chunkTokens: Int
    let prefillMode: PrefillMode
    let comparePrefillModes: Bool
    let verifyMTP: Bool
    let validateNativeMTP: Bool
    let validateNativeMTPBoundary: Bool
    let mtpBlockSize: Int?
    let mtpDiagnostics: Bool
    let fixtureCapture: Bool
    let targetLayerCount: Int?
    let mtpFCOrientation: Qwen38MTPFCOrientation
    let qwenGPUExecutionMode: QwenGPUExecutionMode
    let ngramReadConcurrency: Int
    let ngramRowCacheBytes: Int
    let ngramRowCacheMaxUniqueRows: Int
    let ngramRowProfileMaxRows: Int
    let ngramPinnedRows: [Int64]
    let ngramPinnedRowBytes: Int
    let denseCacheBytes: UInt64?
    let expertCacheSlots: Int
    let expertCachePolicy: ExpertCachePolicy
    let workload: WorkloadMetadata?

    static func parse(_ raw: [String]) throws -> Arguments {
        var modelPath: String?
        var prompt: String?
        var promptSequencePath: String?
        var promptTokens: [Int32]?
        var proposedTokens: [Int32]?
        var maxNewTokens = 64
        var repeatCount = 1
        var maxContext = 4_096
        var chunkTokens = 128
        var prefillMode: PrefillMode = .batch
        var comparePrefillModes = false
        var verifyMTP = false
        var validateNativeMTP = false
        var validateNativeMTPBoundary = false
        var mtpBlockSize: Int?
        var mtpDiagnostics = false
        var fixtureCapture = false
        var targetLayerCount: Int?
        var mtpFCOrientation: Qwen38MTPFCOrientation = .normal
        var qwenGPUExecutionMode: QwenGPUExecutionMode = .ordered
        var ngramReadConcurrency = RuntimeConfiguration.defaultNgramReadConcurrency
        var ngramRowCacheBytes = RuntimeConfiguration.defaultNgramRowCacheBytes
        var ngramRowCacheMaxUniqueRows = RuntimeConfiguration.defaultNgramRowCacheMaxUniqueRows
        var ngramRowProfileMaxRows = RuntimeConfiguration.defaultNgramRowProfileMaxRows
        var ngramPinnedRows = RuntimeConfiguration.defaultNgramPinnedRows
        var ngramPinnedRowBytes = RuntimeConfiguration.defaultNgramPinnedRowBytes
        var denseCacheBytes: UInt64?
        var expertCacheSlots = RuntimeConfiguration.defaultExpertCacheSlots
        var expertCachePolicy: ExpertCachePolicy = .lfu
        var workloadID: String?
        var workloadCategory: String?
        var fixtureID: String?
        var workloadArm: WorkloadArm?
        var index = 0

        while index < raw.count {
            let option = raw[index]
            if option == "--help" || option == "-h" {
                throw ArgumentError.helpRequested
            }
            if option == "--compare-prefill-modes" {
                comparePrefillModes = true
                index += 1
                continue
            }
            if option == "--verify-mtp" {
                verifyMTP = true
                index += 1
                continue
            }
            if option == "--validate-native-mtp" {
                validateNativeMTP = true
                index += 1
                continue
            }
            if option == "--validate-native-mtp-boundary" {
                validateNativeMTPBoundary = true
                index += 1
                continue
            }
            if option == "--mtp-diagnostics" {
                mtpDiagnostics = true
                index += 1
                continue
            }
            if option == "--fixture-capture" {
                fixtureCapture = true
                index += 1
                continue
            }
            guard index + 1 < raw.count else {
                throw ArgumentError.invalid("missing value for \(option)")
            }
            let value = raw[index + 1]
            switch option {
            case "--model":
                modelPath = value
            case "--prompt":
                guard prompt == nil else {
                    throw ArgumentError.invalid("--prompt may only be provided once")
                }
                prompt = value
            case "--prompt-sequence-file":
                guard promptSequencePath == nil else {
                    throw ArgumentError.invalid(
                        "--prompt-sequence-file may only be provided once")
                }
                promptSequencePath = value
            case "--prompt-tokens":
                promptTokens = try parseTokenList(value, option: option)
            case "--proposed-tokens":
                proposedTokens = try parseTokenList(value, option: option)
            case "--max-new-tokens":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgumentError.invalid("\(option) must be a positive integer")
                }
                maxNewTokens = parsed
            case "--repeat-count":
                guard let parsed = Int(value), (1...32).contains(parsed) else {
                    throw ArgumentError.invalid("\(option) must be between 1 and 32")
                }
                repeatCount = parsed
            case "--max-context":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgumentError.invalid("\(option) must be a positive integer")
                }
                maxContext = parsed
            case "--chunk-tokens":
                guard let parsed = Int(value),
                      PrefillRuntimeConfig.allowedChunkTokens.contains(parsed) else {
                    throw ArgumentError.invalid(
                        "\(option) must be one of \(PrefillRuntimeConfig.allowedChunkTokens)")
                }
                chunkTokens = parsed
            case "--mtp-block-size":
                guard let parsed = Int(value), (1...4).contains(parsed) else {
                    throw ArgumentError.invalid("\(option) must be between 1 and 4")
                }
                mtpBlockSize = parsed
            case "--mtp-fc-orientation":
                guard let parsed = Qwen38MTPFCOrientation(rawValue: value) else {
                    throw ArgumentError.invalid(
                        "\(option) must be normal, transpose-embedding, "
                            + "transpose-hidden, or transpose-both")
                }
                mtpFCOrientation = parsed
            case "--target-layer-count":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgumentError.invalid(
                        "\(option) must be a positive integer")
                }
                targetLayerCount = parsed
            case "--prefill-mode":
                guard let parsed = PrefillMode(rawValue: value) else {
                    throw ArgumentError.invalid("\(option) must be batch or scalar")
                }
                prefillMode = parsed
            case "--qwen-gpu-execution":
                guard let parsed = QwenGPUExecutionMode(rawValue: value) else {
                    throw ArgumentError.invalid(
                        "\(option) must be ordered or parallel-delta-projections")
                }
                qwenGPUExecutionMode = parsed
            case "--ngram-read-concurrency":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedNgramReadConcurrency.contains(parsed) else {
                    throw ArgumentError.invalid(
                        "\(option) must be one of \(RuntimeConfiguration.allowedNgramReadConcurrency)")
                }
                ngramReadConcurrency = parsed
            case "--ngram-cache-bytes":
                guard let parsed = Int(value),
                      parsed >= 0,
                      parsed <= RuntimeConfiguration.maxNgramRowCacheBytes else {
                    throw ArgumentError.invalid(
                        "\(option) must be between 0 and \(RuntimeConfiguration.maxNgramRowCacheBytes)")
                }
                ngramRowCacheBytes = parsed
            case "--ngram-cache-rows":
                guard let parsed = Int(value),
                      parsed > 0,
                      parsed <= RuntimeConfiguration.maxNgramRowCacheMaxUniqueRows else {
                    throw ArgumentError.invalid(
                        "\(option) must be between 1 and \(RuntimeConfiguration.maxNgramRowCacheMaxUniqueRows)")
                }
                ngramRowCacheMaxUniqueRows = parsed
            case "--ngram-profile-rows":
                guard let parsed = Int(value),
                      parsed >= 0,
                      parsed <= RuntimeConfiguration.maxNgramRowProfileMaxRows else {
                    throw ArgumentError.invalid(
                        "\(option) must be between 0 and \(RuntimeConfiguration.maxNgramRowProfileMaxRows)")
                }
                ngramRowProfileMaxRows = parsed
            case "--ngram-pin-rows":
                ngramPinnedRows = try parseAddressList(value, option: option)
            case "--ngram-pin-bytes":
                guard let parsed = Int(value),
                      parsed >= 0,
                      parsed <= RuntimeConfiguration.maxNgramPinnedRowBytes else {
                    throw ArgumentError.invalid(
                        "\(option) must be between 0 and \(RuntimeConfiguration.maxNgramPinnedRowBytes)")
                }
                ngramPinnedRowBytes = parsed
            case "--dense-cache-bytes":
                guard let parsed = UInt64(value),
                      parsed > 0,
                      parsed <= RuntimeConfiguration.maxDenseCacheBytes else {
                    throw ArgumentError.invalid(
                        "\(option) must be between 1 and \(RuntimeConfiguration.maxDenseCacheBytes)")
                }
                denseCacheBytes = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value), parsed >= 10 else {
                    throw ArgumentError.invalid(
                        "\(option) must be an integer greater than or equal to 10")
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                guard let parsed = ExpertCachePolicy(rawValue: value) else {
                    throw ArgumentError.invalid("\(option) must be lfu or lru")
                }
                expertCachePolicy = parsed
            case "--workload-id":
                workloadID = value
            case "--workload-category":
                workloadCategory = value
            case "--fixture-id":
                fixtureID = value
            case "--workload-arm":
                guard let parsed = WorkloadArm(rawValue: value) else {
                    throw ArgumentError.invalid(
                        "\(option) must be drafting-disabled or drafting-enabled")
                }
                workloadArm = parsed
            default:
                throw ArgumentError.invalid("unknown argument: \(option)")
            }
            index += 2
        }

        guard let modelPath, !modelPath.isEmpty else {
            throw ArgumentError.invalid("--model is required")
        }
        let hasTextPrompt = prompt != nil || promptSequencePath != nil
        let hasVerifierTokens = promptTokens != nil
            || proposedTokens != nil
            || mtpBlockSize != nil
        guard hasTextPrompt != hasVerifierTokens else {
            throw ArgumentError.invalid(
                "provide either --prompt, --prompt-sequence-file, or verifier token arguments")
        }
        guard !(prompt != nil && promptSequencePath != nil) else {
            throw ArgumentError.invalid(
                "--prompt and --prompt-sequence-file cannot be combined")
        }
        if let prompt, prompt.isEmpty {
            throw ArgumentError.invalid("--prompt must not be empty")
        }
        if let promptSequencePath, promptSequencePath.isEmpty {
            throw ArgumentError.invalid("--prompt-sequence-file must not be empty")
        }
        if hasVerifierTokens {
            guard let promptTokens, !promptTokens.isEmpty else {
                throw ArgumentError.invalid("--prompt-tokens must contain at least one token")
            }
            if let mtpBlockSize {
                guard verifyMTP else {
                    throw ArgumentError.invalid("--mtp-block-size requires --verify-mtp")
                }
                guard prefillMode == .scalar else {
                    throw ArgumentError.invalid(
                        "--mtp-block-size requires --prefill-mode scalar")
                }
                guard proposedTokens == nil else {
                    throw ArgumentError.invalid(
                        "--mtp-block-size cannot be combined with --proposed-tokens")
                }
                guard maxContext >= promptTokens.count + mtpBlockSize + 1 else {
                    throw ArgumentError.invalid(
                        "--max-context is too small for the prompt and verification block")
                }
            } else {
                guard let proposedTokens, !proposedTokens.isEmpty else {
                    throw ArgumentError.invalid("--proposed-tokens must contain at least one token")
                }
                guard maxContext >= promptTokens.count + proposedTokens.count + 1 else {
                    throw ArgumentError.invalid(
                        "--max-context is too small for the prompt and verification block")
                }
            }
            if (verifyMTP || validateNativeMTPBoundary) && promptTokens.count < 2 {
                throw ArgumentError.invalid(
                    "MTP boundary validation requires at least two prompt tokens")
            }
        } else if verifyMTP || validateNativeMTP || validateNativeMTPBoundary
                || comparePrefillModes || mtpBlockSize != nil || fixtureCapture
                || targetLayerCount != nil {
            throw ArgumentError.invalid(
                "MTP verification, native validation, fixture capture, block drafting, prefill comparison, and target layer limiting require token mode")
        }
        guard !(validateNativeMTP && validateNativeMTPBoundary) else {
            throw ArgumentError.invalid(
                "--validate-native-mtp and --validate-native-mtp-boundary cannot be combined")
        }
        if fixtureCapture && prefillMode != .scalar {
            throw ArgumentError.invalid(
                "--fixture-capture requires --prefill-mode scalar")
        }
            if targetLayerCount != nil && !(mtpDiagnostics || fixtureCapture) {
                throw ArgumentError.invalid(
                "--target-layer-count requires --mtp-diagnostics or --fixture-capture")
            }
        if !hasTextPrompt && repeatCount != 1 {
            throw ArgumentError.invalid("--repeat-count requires text completion mode")
        }
        if (validateNativeMTP || validateNativeMTPBoundary) && prefillMode != .scalar {
            throw ArgumentError.invalid(
                "native MTP validation requires --prefill-mode scalar")
        }

        let workloadArgumentsProvided = workloadID != nil
            || workloadCategory != nil
            || fixtureID != nil
            || workloadArm != nil
        let workload: WorkloadMetadata?
        if workloadArgumentsProvided {
            guard let workloadID, !workloadID.isEmpty,
                  let workloadCategory, !workloadCategory.isEmpty,
                  let fixtureID, !fixtureID.isEmpty,
                  let workloadArm else {
                throw ArgumentError.invalid(
                    "workload metadata requires --workload-id, --workload-category, "
                        + "--fixture-id, and --workload-arm")
            }
            guard workloadCategories.contains(workloadCategory) else {
                throw ArgumentError.invalid(
                    "--workload-category must be one of \(workloadCategories.sorted())")
            }
            workload = WorkloadMetadata(
                workloadID: workloadID,
                category: workloadCategory,
                fixtureID: fixtureID,
                arm: workloadArm)
        } else {
            workload = nil
        }

        return Arguments(
            modelURL: URL(fileURLWithPath: modelPath, isDirectory: true),
            prompt: prompt,
            promptSequenceURL: promptSequencePath.map {
                URL(fileURLWithPath: $0, isDirectory: false)
            },
            promptTokens: promptTokens ?? [],
            proposedTokens: proposedTokens ?? [],
            maxNewTokens: maxNewTokens,
            repeatCount: repeatCount,
            maxContext: maxContext,
            chunkTokens: chunkTokens,
            prefillMode: prefillMode,
            comparePrefillModes: comparePrefillModes,
            verifyMTP: verifyMTP,
            validateNativeMTP: validateNativeMTP,
            validateNativeMTPBoundary: validateNativeMTPBoundary,
            mtpBlockSize: mtpBlockSize,
            mtpDiagnostics: mtpDiagnostics,
            fixtureCapture: fixtureCapture,
            targetLayerCount: targetLayerCount,
            mtpFCOrientation: mtpFCOrientation,
            qwenGPUExecutionMode: qwenGPUExecutionMode,
            ngramReadConcurrency: ngramReadConcurrency,
            ngramRowCacheBytes: ngramRowCacheBytes,
            ngramRowCacheMaxUniqueRows: ngramRowCacheMaxUniqueRows,
            ngramRowProfileMaxRows: ngramRowProfileMaxRows,
            ngramPinnedRows: ngramPinnedRows,
            ngramPinnedRowBytes: ngramPinnedRowBytes,
            denseCacheBytes: denseCacheBytes,
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            workload: workload)
    }

    private static func parseTokenList(_ raw: String,
                                       option: String) throws -> [Int32] {
        let fields = raw.split(separator: ",", omittingEmptySubsequences: false)
        guard !fields.isEmpty else {
            throw ArgumentError.invalid("\(option) must be a comma-separated token list")
        }
        return try fields.map { field in
            guard !field.isEmpty, let token = Int32(field), token >= 0 else {
                throw ArgumentError.invalid(
                    "\(option) contains an invalid token ID: \(field)")
            }
            return token
        }
    }

    private static func parseAddressList(_ raw: String,
                                         option: String) throws -> [Int64] {
        let fields = raw.split(separator: ",", omittingEmptySubsequences: false)
        guard !fields.isEmpty else {
            throw ArgumentError.invalid("\(option) must be a comma-separated address list")
        }
        let addresses = try fields.map { field in
            guard !field.isEmpty, let address = Int64(field), address >= 0 else {
                throw ArgumentError.invalid(
                    "\(option) contains an invalid row address: \(field)")
            }
            return address
        }
        guard Set(addresses).count == addresses.count else {
            throw ArgumentError.invalid("\(option) must not contain duplicate addresses")
        }
        return addresses
    }
}

private let fixtureMaxPayloadBytes = 1_048_576

private struct FixtureTensor: Codable {
    let label: String
    let shape: [Int]
    let dtype: String
    let finiteCount: Int
    let nanCount: Int
    let positiveInfinityCount: Int
    let negativeInfinityCount: Int
    let checksum: UInt64
    let values: [Float16]

    init(label: String, shape: [Int], values: [Float16]) {
        var finiteCount = 0
        var nanCount = 0
        var positiveInfinityCount = 0
        var negativeInfinityCount = 0
        for value in values {
            let scalar = Float(value)
            if scalar.isNaN {
                nanCount += 1
            } else if scalar == .infinity {
                positiveInfinityCount += 1
            } else if scalar == -.infinity {
                negativeInfinityCount += 1
            } else {
                finiteCount += 1
            }
        }
        self.label = label
        self.shape = shape
        self.dtype = "float16"
        self.finiteCount = finiteCount
        self.nanCount = nanCount
        self.positiveInfinityCount = positiveInfinityCount
        self.negativeInfinityCount = negativeInfinityCount
        self.checksum = logitTraceChecksum([values])
        self.values = values
    }
}

private struct FixtureStageTensor: Codable {
    let layerIndex: Int
    let stage: String
    let tokenPosition: Int
    let inputToken: Int32
    let shape: [Int]
    let dtype: String
    let finiteCount: Int
    let nanCount: Int
    let positiveInfinityCount: Int
    let negativeInfinityCount: Int
    let checksum: UInt64
    let values: [Float16]

    init(capture: Qwen38StageCapture) {
        var finiteCount = 0
        var nanCount = 0
        var positiveInfinityCount = 0
        var negativeInfinityCount = 0
        for value in capture.values {
            let scalar = Float(value)
            if scalar.isNaN {
                nanCount += 1
            } else if scalar == .infinity {
                positiveInfinityCount += 1
            } else if scalar == -.infinity {
                negativeInfinityCount += 1
            } else {
                finiteCount += 1
            }
        }
        self.layerIndex = capture.layerIndex
        self.stage = capture.stage
        self.tokenPosition = capture.tokenPosition
        self.inputToken = capture.inputToken
        self.shape = capture.shape
        self.dtype = capture.dtype
        self.finiteCount = finiteCount
        self.nanCount = nanCount
        self.positiveInfinityCount = positiveInfinityCount
        self.negativeInfinityCount = negativeInfinityCount
        self.checksum = logitTraceChecksum([capture.values])
        self.values = capture.values
    }
}

private struct TargetBoundaryFixture: Codable {
    let schemaVersion: Int
    let evidenceClass: String
    let referenceStatus: String
    let targetPosition: Int
    let inputToken: Int32
    let streamCount: Int
    let hiddenSize: Int
    let payloadBytes: Int
    let tensors: [FixtureTensor]
    let stageTensors: [FixtureStageTensor]
    let routerDiagnostics: Qwen38RouterDiagnostics?
    let intermediateDiagnostics: NativeIntermediateDiagnostics?
}

private struct NativeIntermediateDiagnostics: Codable {
    let embedding: Qwen38LogitDiagnostics?
    let firstLayer: Qwen38LogitDiagnostics?
    let layerZeroAttentionOutput: Qwen38LogitDiagnostics?
    let layerZeroAfterAttention: Qwen38LogitDiagnostics?
    let layerZeroMLPInput: Qwen38LogitDiagnostics?
    let layerZeroMLPOutput: Qwen38LogitDiagnostics?
    let layerOneOutput: Qwen38LogitDiagnostics?
    let layerOneAttentionInput: Qwen38LogitDiagnostics?
    let layerOneQKV: Qwen38LogitDiagnostics?
    let layerOneRecurrent: Qwen38LogitDiagnostics?
    let layerOneNormalized: Qwen38LogitDiagnostics?
    let layerOneAttentionOutput: Qwen38LogitDiagnostics?
    let layerOneAfterAttention: Qwen38LogitDiagnostics?
    let layerOneMLPInput: Qwen38LogitDiagnostics?
    let layerOneMLPOutput: Qwen38LogitDiagnostics?
    let preFinalMixer: Qwen38LogitDiagnostics?
    let finalHidden: Qwen38LogitDiagnostics?
}

private struct NativeDraftStreamOrderDiagnostic: Codable {
    let order: [Int]
    let draftToken: Int32
}

private struct ProbeModeRun {
    let mode: PrefillMode
    let setupSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let decodeGPUActiveNanos: UInt64
    let decodeCommandBufferEncodeNanos: UInt64
    let decodeCommandBufferWaitNanos: UInt64
    let decodeTimingSamples: [Qwen38DecodeTimingSample]
    let speculativeReplay: Qwen38SpeculativeReplaySample
    let ngramCacheDiagnostics: NgramCacheDiagnostics
    let memoryDiagnostics: Qwen38MemoryDiagnostics
    let ngramRowProfile: [NgramRowProfileEntry]
    let prefillWork: PrefillWorkDiagnostics?
    let boundaryToken: Int32
    let verification: GreedyBlockVerification?
    let mtpRequestedBlockSize: Int?
    let mtpProposalTokens: [Int32]?
    let emittedTokens: [Int32]
    let statePosition: Int
    let logitTrace: [[Float16]]
    let fixture: TargetBoundaryFixture?
    let nativeDraftStreamOrderDrafts: [NativeDraftStreamOrderDiagnostic]?
    let nativeDraftAlternateEmbeddingToken: Int32?
    let nativeDraftBoundaryTargetToken: Int32?
    let nativeDraftToken: Int32?
    let nativeDraftFreshTargetToken: Int32?
    let nativeDraftRawTargetToken: Int32?
    let nativeDraftAlternateToken: Int32?
    let nativeDraftTargetToken: Int32?
    let nativeDraftMatchesTarget: Bool?
    let nativeDraftAlternateMatchesTarget: Bool?
    let nativeDraftError: String?
    let nativeDraftSeconds: Double?
    let nativeDraftTokensPerSecond: Double?
    let nativeDraftTargetPosition: Int?
    let nativeDraftMTPPosition: Int?
}

private struct PrefillParityResult: Codable {
    let comparedModes: [String]
    let logitsMatch: Bool
    let maxAbsoluteLogitDifference: Float
    let emittedTokensMatch: Bool
    let continuationPositionsMatch: Bool
    let batchEmittedTokens: [Int32]
    let scalarEmittedTokens: [Int32]
}

private struct ProbeResult: Codable {
    let prefillMode: String
    let qwenGPUExecutionMode: QwenGPUExecutionMode
    let workload: WorkloadMetadata?
    let ngramReadConcurrency: Int
    let ngramRowCacheBytes: Int
    let ngramRowCacheMaxUniqueRows: Int
    let ngramRowProfileMaxRows: Int
    let ngramPinnedRows: [Int64]
    let ngramPinnedRowBytes: Int
    let boundaryToken: Int32
    let targetTokens: [Int32]
    let semanticValidity: Qwen38SemanticValidity
    let acceptedTokenCount: Int
    let emittedTokens: [Int32]
    let statePosition: Int
    let mtpTargetTokens: [Int32]?
    let mtpAcceptedTokenCount: Int?
    let mtpStatePosition: Int?
    let mtpVerificationExecutionPath: String?
    let mtpVerificationProposalCount: Int?
    let mtpVerificationTargetRowCount: Int?
    let mtpRequestedBlockSize: Int?
    let mtpProposedTokens: [Int32]?
    let fixture: TargetBoundaryFixture?
    let nativeDraftStreamOrderDrafts: [NativeDraftStreamOrderDiagnostic]?
    let nativeDraftAlternateEmbeddingToken: Int32?
    let nativeDraftBoundaryTargetToken: Int32?
    let nativeDraftToken: Int32?
    let nativeDraftFreshTargetToken: Int32?
    let nativeDraftRawTargetToken: Int32?
    let nativeDraftAlternateToken: Int32?
    let nativeDraftTargetToken: Int32?
    let nativeDraftMatchesTarget: Bool?
    let nativeDraftAlternateMatchesTarget: Bool?
    let nativeDraftError: String?
    let nativeDraftSeconds: Double?
    let nativeDraftTokensPerSecond: Double?
    let nativeDraftTargetPosition: Int?
    let nativeDraftMTPPosition: Int?
    let seedCaptureBytes: Int
    let verificationCaptureBytes: Int
    let setupSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let decodeGPUActiveNanos: UInt64
    let decodeCommandBufferEncodeNanos: UInt64
    let decodeCommandBufferWaitNanos: UInt64
    let prefillCommandBufferEncodeNanos: UInt64?
    let prefillCommandBufferWaitNanos: UInt64?
    let decodeTimingSamples: [Qwen38DecodeTimingSample]
    let speculativeReplay: Qwen38SpeculativeReplaySample
    let ngramCacheDiagnostics: NgramCacheDiagnostics
    let memoryDiagnostics: Qwen38MemoryDiagnostics
    let ngramRowProfile: [NgramRowProfileEntry]
    let prefillWork: PrefillWorkDiagnostics?
    let endToEndSeconds: Double
    let prefillParity: PrefillParityResult?
    let logitTraceChecksum: UInt64
}

private struct TextCompletionReceipt: Codable {
    let receiptType: String
    let qwenGPUExecutionMode: QwenGPUExecutionMode
    let workload: WorkloadMetadata?
    let prompt: String
    let promptTokens: Int
    let promptTokenIDs: [Int32]
    let generatedTokens: Int
    let tokenIDs: [Int32]
    let semanticValidity: Qwen38SemanticValidity
    let outputSHA256: String
    let stopReason: String
    let kvPosition: Int
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    let setupSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let endToEndSeconds: Double
    let decodeTokensPerSecond: Double
    let prefillWork: PrefillWorkDiagnostics?
    let qwenDecodeDiagnostics: QwenDecodeDiagnosticsAggregate?
    let memoryDiagnostics: Qwen38MemoryDiagnostics
    let ngramCacheDiagnostics: NgramCacheDiagnostics
    let ngramRowProfile: [NgramRowProfileEntry]
    let embeddingDiagnostics: Qwen38LogitDiagnostics?
    let firstLayerDiagnostics: Qwen38LogitDiagnostics?
    let layerOneOutputDiagnostics: Qwen38LogitDiagnostics?
    let layerOneNgramEmbeddingDiagnostics: Qwen38LogitDiagnostics?
    let layerOnePLEProjectedKeyDiagnostics: Qwen38LogitDiagnostics?
    let layerOnePLEValueDiagnostics: Qwen38LogitDiagnostics?
    let layerOnePLEOutputDiagnostics: Qwen38LogitDiagnostics?
    let layerOneAttentionInputDiagnostics: Qwen38LogitDiagnostics?
    let layerOneQKVDiagnostics: Qwen38LogitDiagnostics?
    let layerOneRecurrentDiagnostics: Qwen38LogitDiagnostics?
    let layerOneNormalizedDiagnostics: Qwen38LogitDiagnostics?
    let layerOneAttentionOutputDiagnostics: Qwen38LogitDiagnostics?
    let layerOneAfterAttentionDiagnostics: Qwen38LogitDiagnostics?
    let layerOneMLPInputDiagnostics: Qwen38LogitDiagnostics?
    let layerOneMLPOutputDiagnostics: Qwen38LogitDiagnostics?
    let finalLayerInputDiagnostics: Qwen38LogitDiagnostics?
    let preFinalMixerDiagnostics: Qwen38LogitDiagnostics?
    let finalHiddenDiagnostics: Qwen38LogitDiagnostics?
    let logitDiagnostics: Qwen38LogitDiagnostics?
    let draftingDiagnostics: DraftingDiagnosticsAggregate?
    let sampling: String
    let maxNewTokens: Int
}

private struct TextCompletionSeriesReceipt: Codable {
    let receiptType: String
    let repeatCount: Int
    let sequenceLength: Int
    let runs: [TextCompletionReceipt]
}

private func elapsedSeconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func loadPromptSequence(from url: URL) throws -> [String] {
    let contents: String
    do {
        contents = try String(contentsOf: url, encoding: .utf8)
    } catch {
        throw ArgumentError.invalid(
            "could not read --prompt-sequence-file at \(url.path): \(error)")
    }
    let prompts = contents
        .split(whereSeparator: \.isNewline)
        .map { String($0).trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
    guard !prompts.isEmpty else {
        throw ArgumentError.invalid("--prompt-sequence-file must contain at least one prompt")
    }
    return prompts
}

private func greedyToken(from logits: MTLBuffer, vocabularySize: Int) -> Int32 {
    let values = logits.contents().assumingMemoryBound(to: Float16.self)
    var bestIndex = 0
    var bestValue = values[0]
    for index in 1..<vocabularySize where values[index] > bestValue {
        bestIndex = index
        bestValue = values[index]
    }
    return Int32(bestIndex)
}

private func copyLogits(_ logits: MTLBuffer, count: Int) -> [Float16] {
    let values = logits.contents().assumingMemoryBound(to: Float16.self)
    return Array(UnsafeBufferPointer(start: values, count: count))
}

private func logitTraceChecksum(_ trace: [[Float16]]) -> UInt64 {
    var checksum: UInt64 = 14695981039346656037
    for row in trace {
        checksum ^= UInt64(row.count)
        checksum &*= 1099511628211
        for value in row {
            checksum ^= UInt64(value.bitPattern)
            checksum &*= 1099511628211
        }
    }
    return checksum
}

private func targetBoundaryFixture(
    snapshot: Qwen38TargetBoundarySnapshot,
    logits: [Float16],
    vocabularySize: Int,
    intermediateDiagnostics: NativeIntermediateDiagnostics?) throws -> TargetBoundaryFixture {
    let tensors = [
        FixtureTensor(
            label: "native-target-hidden-streams",
            shape: [snapshot.streamCount, snapshot.hiddenSize],
            values: snapshot.targetHiddenStreams),
        snapshot.rawTargetHiddenStreams.map { values in
            FixtureTensor(
                label: "native-target-raw-hidden-streams",
                shape: [snapshot.streamCount, snapshot.hiddenSize],
                values: values)
        },
        FixtureTensor(
            label: "native-target-logits",
            shape: [vocabularySize],
            values: logits),
    ].compactMap { $0 }
    let stageTensors = snapshot.stageCaptures.map(FixtureStageTensor.init)
    let routerPayloadBytes = snapshot.routerDiagnostics.map {
        $0.routerLogits.count * MemoryLayout<Float>.stride
            + $0.selectedExperts.count * MemoryLayout<Int>.stride
            + $0.routeWeightBits.count * MemoryLayout<UInt16>.stride
    } ?? 0
    let payloadBytes = tensors.reduce(0) {
        $0 + $1.values.count * MemoryLayout<Float16>.stride
    } + stageTensors.reduce(0) {
        $0 + $1.values.count * MemoryLayout<Float16>.stride
    } + routerPayloadBytes
    guard payloadBytes <= fixtureMaxPayloadBytes else {
        throw ArgumentError.invalid(
            "fixture payload \(payloadBytes) exceeds limit \(fixtureMaxPayloadBytes)")
    }
    return TargetBoundaryFixture(
        schemaVersion: 3,
        evidenceClass: "native-self-observation",
        referenceStatus: "no-independent-reference",
        targetPosition: snapshot.targetPosition,
        inputToken: snapshot.inputToken,
        streamCount: snapshot.streamCount,
        hiddenSize: snapshot.hiddenSize,
        payloadBytes: payloadBytes,
        tensors: tensors,
        stageTensors: stageTensors,
        routerDiagnostics: snapshot.routerDiagnostics,
        intermediateDiagnostics: intermediateDiagnostics)
}

private func oppositeMode(_ mode: PrefillMode) -> PrefillMode {
    mode == .batch ? .scalar : .batch
}

private func writeDiagnostic(_ message: String) {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
}

private func runMode(arguments: Arguments,
                     mode: PrefillMode,
                     model: Model,
                     context: MetalContext) async throws -> ProbeModeRun {
    let setupStart = DispatchTime.now().uptimeNanoseconds
    let runtimeConfiguration = RuntimeConfiguration(
        qwenGPUExecutionMode: arguments.qwenGPUExecutionMode,
        ngramReadConcurrency: arguments.ngramReadConcurrency,
        ngramRowCacheBytes: arguments.ngramRowCacheBytes,
        ngramRowCacheMaxUniqueRows: arguments.ngramRowCacheMaxUniqueRows,
        ngramRowProfileMaxRows: arguments.ngramRowProfileMaxRows,
        ngramPinnedRows: arguments.ngramPinnedRows,
        ngramPinnedRowBytes: arguments.ngramPinnedRowBytes)
    let runner = try Qwen38ForwardRunner(
        model: model,
        context: context,
        maxContext: arguments.maxContext,
        runtimeConfiguration: runtimeConfiguration,
        enableMTPDiagnostics: arguments.mtpDiagnostics || arguments.fixtureCapture,
        mtpFCOrientation: arguments.mtpFCOrientation,
        targetLayerCount: arguments.targetLayerCount)
    if arguments.mtpDiagnostics && model.hasMTP {
        let mtpWeights = try Qwen38MTPWeights(model: model)
        writeDiagnostic("mtp inventory count=\(mtpWeights.tensorNames.count)")
        for name in mtpWeights.tensorNames {
            let tensor = try mtpWeights.tensor(relativeName: name)
            let shape = [tensor.shape.0, tensor.shape.1, tensor.shape.2, tensor.shape.3]
                .filter { $0 > 0 }
                .map(String.init)
                .joined(separator: "x")
            let quantization = tensor.quantization.map {
                "q\($0.bits)g\($0.groupSize)"
            } ?? "none"
            writeDiagnostic("mtp inventory name=\(name) shape=\(shape) dtype=\(tensor.dtype) \(quantization)")
            writeDiagnostic("mtp inventory offsets=\(tensor.offset)/\(tensor.length) scales=\(tensor.scaleOffset)/\(tensor.scaleLength) biases=\(tensor.biasOffset)/\(tensor.biasLength)")
        }
    }
    guard let logits = context.device.makeBuffer(
        length: model.config.vocabSize * MemoryLayout<Float16>.stride,
        options: .storageModeShared) else {
        throw NSError(
            domain: "TurboFieldfareQwenVerifierProbe",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "failed to allocate logits buffer"])
    }
    let setupSeconds = elapsedSeconds(since: setupStart)
    let prefillStart = DispatchTime.now().uptimeNanoseconds
    let prefillConfig = PrefillRuntimeConfig.production(
        chunkTokens: arguments.chunkTokens)
    let prefillTokens: ArraySlice<Int32> = arguments.verifyMTP
        || arguments.validateNativeMTPBoundary
        ? arguments.promptTokens.dropLast()
        : arguments.promptTokens[...]
    let prefillWork: PrefillWorkDiagnostics?
    switch mode {
    case .batch:
        let prefillResult = try await runner.prefillChunked(
            tokens: prefillTokens,
            startPosition: 0,
            outputMode: .logits,
            config: prefillConfig,
            into: logits,
            onProgress: { _ in })
        prefillWork = prefillResult.work
    case .scalar:
        prefillWork = nil
        for (index, token) in prefillTokens.enumerated() {
            try await runner.produce(
                token: token,
                position: runner.continuationPosition,
                into: logits)
            if (arguments.validateNativeMTP
                || arguments.validateNativeMTPBoundary
                || arguments.mtpBlockSize != nil)
                && index < prefillTokens.count - 1 {
                let nextIndex = prefillTokens.index(
                    prefillTokens.startIndex, offsetBy: index + 1)
                _ = try runner.primeNativeMTPState(token: prefillTokens[nextIndex])
            }
        }
    }
    let prefillSeconds = elapsedSeconds(since: prefillStart)

    let boundaryToken = arguments.verifyMTP || arguments.validateNativeMTPBoundary
        ? arguments.promptTokens[arguments.promptTokens.count - 1]
        : greedyToken(from: logits, vocabularySize: model.config.vocabSize)
    let verificationProposals: [Int32]
    if let mtpBlockSize = arguments.mtpBlockSize {
        let firstProposal = try runner.primeNativeMTPState(
            token: boundaryToken,
            freshTargetHiddenStreams: true)
        let remainingCount = mtpBlockSize - 1
        let remainingProposals = remainingCount > 0
            ? try runner.draftNativeMTPBlock(
                initialToken: firstProposal,
                tokenCount: remainingCount)
            : []
        verificationProposals = [firstProposal] + remainingProposals
    } else {
        verificationProposals = arguments.proposedTokens
    }
    let verification: GreedyBlockVerification?
    if arguments.verifyMTP {
        verification = try await runner.verifyGreedyBlock(
            boundaryToken: boundaryToken,
            proposedTokens: verificationProposals[...],
            startPosition: runner.continuationPosition,
            config: prefillConfig,
            into: logits)
    } else {
        verification = nil
    }

    var logitTrace = [copyLogits(logits, count: model.config.vocabSize)]
    let targetToken = greedyToken(from: logits, vocabularySize: model.config.vocabSize)
    let fixture: TargetBoundaryFixture?
    if arguments.fixtureCapture {
        let snapshot = try runner.targetBoundarySnapshot(
            maxPayloadBytes: fixtureMaxPayloadBytes)
        fixture = try targetBoundaryFixture(
            snapshot: snapshot,
            logits: logitTrace[0],
            vocabularySize: model.config.vocabSize,
            intermediateDiagnostics: NativeIntermediateDiagnostics(
                embedding: runner.lastEmbeddingDiagnostics,
                firstLayer: runner.lastFirstLayerDiagnostics,
                layerZeroAttentionOutput: runner.lastLayerZeroAttentionOutputDiagnostics,
                layerZeroAfterAttention: runner.lastLayerZeroAfterAttentionDiagnostics,
                layerZeroMLPInput: runner.lastLayerZeroMLPInputDiagnostics,
                layerZeroMLPOutput: runner.lastLayerZeroMLPOutputDiagnostics,
                layerOneOutput: runner.lastLayerOneOutputDiagnostics,
                layerOneAttentionInput: runner.lastLayerOneAttentionInputDiagnostics,
                layerOneQKV: runner.lastLayerOneQKVDiagnostics,
                layerOneRecurrent: runner.lastLayerOneRecurrentDiagnostics,
                layerOneNormalized: runner.lastLayerOneNormalizedDiagnostics,
                layerOneAttentionOutput: runner.lastLayerOneAttentionOutputDiagnostics,
                layerOneAfterAttention: runner.lastLayerOneAfterAttentionDiagnostics,
                layerOneMLPInput: runner.lastLayerOneMLPInputDiagnostics,
                layerOneMLPOutput: runner.lastLayerOneMLPOutputDiagnostics,
                preFinalMixer: runner.lastPreFinalMixerDiagnostics,
                finalHidden: runner.lastFinalHiddenDiagnostics))
    } else {
        fixture = nil
    }
    let nativeDraftBoundaryTargetToken = arguments.validateNativeMTP
        || arguments.validateNativeMTPBoundary
        ? boundaryToken
        : nil
    let nativeDraftStart = DispatchTime.now().uptimeNanoseconds
    let nativeDraftStreamOrderDrafts: [NativeDraftStreamOrderDiagnostic]?
    let nativeDraftToken: Int32?
    let nativeDraftFreshTargetToken: Int32?
    let nativeDraftRawTargetToken: Int32?
    let nativeDraftAlternateToken: Int32?
    let nativeDraftTargetToken: Int32?
    let nativeDraftError: String?
    if arguments.validateNativeMTP || arguments.validateNativeMTPBoundary {
        do {
            let validation = try await runner.validateNativeMTP(
                boundaryToken: boundaryToken,
                alternateEmbeddingToken: prefillTokens.last ?? boundaryToken,
                into: logits)
            nativeDraftStreamOrderDrafts = validation.streamOrderDrafts.map { result in
                NativeDraftStreamOrderDiagnostic(
                    order: result.0,
                    draftToken: result.1)
            }
            nativeDraftToken = validation.draftToken
            nativeDraftFreshTargetToken = validation.freshTargetDraftToken
            nativeDraftRawTargetToken = validation.rawTargetDraftToken
            nativeDraftAlternateToken = validation.alternateDraftToken
            nativeDraftTargetToken = validation.targetToken
            nativeDraftError = nil
        } catch {
            nativeDraftStreamOrderDrafts = nil
            nativeDraftToken = nil
            nativeDraftFreshTargetToken = nil
            nativeDraftRawTargetToken = nil
            nativeDraftAlternateToken = nil
            nativeDraftTargetToken = nil
            nativeDraftError = String(describing: error)
        }
    } else {
        nativeDraftStreamOrderDrafts = nil
        nativeDraftToken = nil
        nativeDraftFreshTargetToken = nil
        nativeDraftRawTargetToken = nil
        nativeDraftAlternateToken = nil
        nativeDraftTargetToken = nil
        nativeDraftError = nil
    }
    let nativeDraftSeconds = arguments.validateNativeMTP
        || arguments.validateNativeMTPBoundary
        ? elapsedSeconds(since: nativeDraftStart)
        : nil
    let nativeDraftTokensPerSecond = nativeDraftSeconds.map { seconds in
        seconds > 0 ? 1 / seconds : 0
    }
    let nativeDraftTargetPosition = arguments.validateNativeMTP
        || arguments.validateNativeMTPBoundary
        ? runner.continuationPosition + 1
        : nil
    let nativeDraftMTPPosition = arguments.validateNativeMTP
        || arguments.validateNativeMTPBoundary
        ? runner.mtpStatePosition.map { $0 + 2 }
        : nil

    let decodeStart = DispatchTime.now().uptimeNanoseconds
    var emittedTokens: [Int32] = []
    var decodeTimingSamples: [Qwen38DecodeTimingSample] = []
    var nextToken = targetToken
    for _ in verificationProposals {
        emittedTokens.append(nextToken)
        try await runner.produce(
            token: nextToken,
            position: runner.continuationPosition,
            into: logits)
        guard let timing = runner.lastDecodeTiming else {
            throw NSError(
                domain: "TurboFieldfareQwenVerifierProbe",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "decode timing sample was not recorded"])
        }
        decodeTimingSamples.append(timing)
        logitTrace.append(copyLogits(logits, count: model.config.vocabSize))
        nextToken = greedyToken(from: logits, vocabularySize: model.config.vocabSize)
    }
    let decodeSeconds = elapsedSeconds(since: decodeStart)
    let decodeTokensPerSecond = decodeSeconds > 0
        ? Double(emittedTokens.count) / decodeSeconds
        : 0
    let decodeGPUActiveNanos = decodeTimingSamples.reduce(0) {
        $0 + $1.gpuActiveNanos
    }
    let decodeCommandBufferEncodeNanos = decodeTimingSamples.reduce(0) {
        $0 + $1.commandBufferEncodeNanos
    }
    let decodeCommandBufferWaitNanos = decodeTimingSamples.reduce(0) {
        $0 + $1.commandBufferWaitNanos
    }
    return ProbeModeRun(
        mode: mode,
        setupSeconds: setupSeconds,
        prefillSeconds: prefillSeconds,
        decodeSeconds: decodeSeconds,
        decodeTokensPerSecond: decodeTokensPerSecond,
        decodeGPUActiveNanos: decodeGPUActiveNanos,
        decodeCommandBufferEncodeNanos: decodeCommandBufferEncodeNanos,
        decodeCommandBufferWaitNanos: decodeCommandBufferWaitNanos,
        decodeTimingSamples: decodeTimingSamples,
        speculativeReplay: runner.lastSpeculativeReplay,
        ngramCacheDiagnostics: runner.ngramCacheDiagnostics,
        memoryDiagnostics: runner.memoryDiagnostics,
        ngramRowProfile: runner.ngramRowProfile,
        prefillWork: prefillWork,
        boundaryToken: boundaryToken,
        verification: verification,
        mtpRequestedBlockSize: arguments.mtpBlockSize,
        mtpProposalTokens: arguments.verifyMTP ? verificationProposals : nil,
        emittedTokens: emittedTokens,
        statePosition: runner.continuationPosition,
        logitTrace: logitTrace,
        fixture: fixture,
        nativeDraftStreamOrderDrafts: nativeDraftStreamOrderDrafts,
        nativeDraftAlternateEmbeddingToken: arguments.validateNativeMTP
            || arguments.validateNativeMTPBoundary
            ? prefillTokens.last ?? boundaryToken
            : nil,
        nativeDraftBoundaryTargetToken: nativeDraftBoundaryTargetToken,
        nativeDraftToken: nativeDraftToken,
        nativeDraftFreshTargetToken: nativeDraftFreshTargetToken,
        nativeDraftRawTargetToken: nativeDraftRawTargetToken,
        nativeDraftAlternateToken: nativeDraftAlternateToken,
        nativeDraftTargetToken: nativeDraftTargetToken,
        nativeDraftMatchesTarget: nativeDraftToken.flatMap { draftToken in
            nativeDraftTargetToken.map { draftToken == $0 }
        },
        nativeDraftAlternateMatchesTarget: nativeDraftAlternateToken.flatMap { draftToken in
            nativeDraftTargetToken.map { draftToken == $0 }
        },
        nativeDraftError: nativeDraftError,
        nativeDraftSeconds: nativeDraftSeconds,
        nativeDraftTokensPerSecond: nativeDraftTokensPerSecond,
        nativeDraftTargetPosition: nativeDraftTargetPosition,
        nativeDraftMTPPosition: nativeDraftMTPPosition)
}

private func runTextCompletion(arguments: Arguments,
                                prompt: String,
                                model: Model,
                                context: MetalContext,
                                tokenizer: GFTokenizer,
                                sharedRunner: Qwen38ForwardRunner?,
                                setupSeconds: Double,
                                endToEndStart: UInt64) async throws -> TextCompletionReceipt {
    let promptIDs = tokenizer.encode(prompt, addBOS: true)
    guard !promptIDs.isEmpty else {
        throw ArgumentError.invalid("encoded prompt must not be empty")
    }
    guard promptIDs.count < arguments.maxContext else {
        throw ArgumentError.invalid(
            "prompt token count reaches maxContext \(arguments.maxContext)")
    }
    let maxNewTokens = min(
        arguments.maxNewTokens,
        arguments.maxContext - promptIDs.count)
    guard maxNewTokens > 0 else {
        throw ArgumentError.invalid(
            "prompt leaves no room for generated tokens in maxContext \(arguments.maxContext)")
    }
    let runner: Qwen38ForwardRunner
    if let sharedRunner {
        runner = sharedRunner
    } else {
        let runtimeConfiguration = RuntimeConfiguration(
            qwenGPUExecutionMode: arguments.qwenGPUExecutionMode,
            ngramReadConcurrency: arguments.ngramReadConcurrency,
            ngramRowCacheBytes: arguments.ngramRowCacheBytes,
            ngramRowCacheMaxUniqueRows: arguments.ngramRowCacheMaxUniqueRows,
            ngramRowProfileMaxRows: arguments.ngramRowProfileMaxRows,
            ngramPinnedRows: arguments.ngramPinnedRows,
            ngramPinnedRowBytes: arguments.ngramPinnedRowBytes)
        runner = try Qwen38ForwardRunner(
            model: model,
            context: context,
            maxContext: arguments.maxContext,
            runtimeConfiguration: runtimeConfiguration,
            enableMTPDiagnostics: arguments.mtpDiagnostics,
            mtpFCOrientation: arguments.mtpFCOrientation,
            draftingStrategy: arguments.workload?.draftingEnabled == true
                ? .experimentalNativeMTP
                : .disabled,
            targetLayerCount: arguments.targetLayerCount)
    }
    let scratch = try RawCompletionScratch(
        context: context,
        vocab: model.config.vocabSize)
    var output = ""
    let result = try await runRawCompletion(
        producer: runner,
        tokenizer: tokenizer,
        promptIds: promptIDs,
        config: GenerationConfig(
            maxNewTokens: maxNewTokens,
            temperature: 0,
            repetitionPenalty: 1),
        context: context,
        scratch: scratch,
        prefillConfig: .production(chunkTokens: arguments.chunkTokens)) { progress in
            switch progress {
            case .prefill:
                break
            case .token(_, _, let delta):
                output += delta
            case .tail(let tail):
                output += tail
            }
        }
    let generatedIDs = Array(result.kvBackedTokenIDs.dropFirst(result.prefillTokens))
        + result.uncommittedBoundaryTokenIDs
    return TextCompletionReceipt(
        receiptType: "qwen38-text-completion",
        qwenGPUExecutionMode: arguments.qwenGPUExecutionMode,
        workload: arguments.workload,
        prompt: prompt,
        promptTokens: result.prefillTokens,
        promptTokenIDs: promptIDs,
        generatedTokens: generatedIDs.count,
        tokenIDs: generatedIDs,
        semanticValidity: Qwen38SemanticValidity.from(tokenIDs: generatedIDs),
        outputSHA256: Sha256Verifier.hashData(Data(output.utf8)),
        stopReason: String(describing: result.reason),
        kvPosition: result.kvPosition,
        cachedPromptTokens: result.cachedPromptTokens,
        computedPrefillTokens: result.computedPrefillTokens,
        setupSeconds: setupSeconds,
        prefillSeconds: result.prefillSeconds,
        decodeSeconds: result.decodeSeconds,
        endToEndSeconds: elapsedSeconds(since: endToEndStart),
        decodeTokensPerSecond: result.decodeSeconds > 0
            ? Double(result.newTokens) / result.decodeSeconds
            : 0,
        prefillWork: result.prefillWork,
        qwenDecodeDiagnostics: result.qwenDecodeDiagnostics,
        memoryDiagnostics: runner.memoryDiagnostics,
        ngramCacheDiagnostics: runner.ngramCacheDiagnostics,
        ngramRowProfile: runner.ngramRowProfile,
        embeddingDiagnostics: runner.lastEmbeddingDiagnostics,
        firstLayerDiagnostics: runner.lastFirstLayerDiagnostics,
        layerOneOutputDiagnostics: runner.lastLayerOneOutputDiagnostics,
        layerOneNgramEmbeddingDiagnostics: runner.lastLayerOneNgramEmbeddingDiagnostics,
        layerOnePLEProjectedKeyDiagnostics: runner.lastLayerOnePLEProjectedKeyDiagnostics,
        layerOnePLEValueDiagnostics: runner.lastLayerOnePLEValueDiagnostics,
        layerOnePLEOutputDiagnostics: runner.lastLayerOnePLEOutputDiagnostics,
        layerOneAttentionInputDiagnostics: runner.lastLayerOneAttentionInputDiagnostics,
        layerOneQKVDiagnostics: runner.lastLayerOneQKVDiagnostics,
        layerOneRecurrentDiagnostics: runner.lastLayerOneRecurrentDiagnostics,
        layerOneNormalizedDiagnostics: runner.lastLayerOneNormalizedDiagnostics,
        layerOneAttentionOutputDiagnostics: runner.lastLayerOneAttentionOutputDiagnostics,
        layerOneAfterAttentionDiagnostics: runner.lastLayerOneAfterAttentionDiagnostics,
        layerOneMLPInputDiagnostics: runner.lastLayerOneMLPInputDiagnostics,
        layerOneMLPOutputDiagnostics: runner.lastLayerOneMLPOutputDiagnostics,
        finalLayerInputDiagnostics: runner.lastFinalLayerInputDiagnostics,
        preFinalMixerDiagnostics: runner.lastPreFinalMixerDiagnostics,
        finalHiddenDiagnostics: runner.lastFinalHiddenDiagnostics,
        logitDiagnostics: runner.lastLogitDiagnostics,
        draftingDiagnostics: result.draftingDiagnostics,
        sampling: "greedy",
        maxNewTokens: maxNewTokens)
}

private func parityResult(batch: ProbeModeRun,
                          scalar: ProbeModeRun) -> PrefillParityResult {
    var maxAbsoluteDifference: Float = 0
    guard batch.logitTrace.count == scalar.logitTrace.count else {
        return PrefillParityResult(
            comparedModes: [PrefillMode.batch.rawValue, PrefillMode.scalar.rawValue],
            logitsMatch: false,
            maxAbsoluteLogitDifference: .infinity,
            emittedTokensMatch: batch.emittedTokens == scalar.emittedTokens,
            continuationPositionsMatch: batch.statePosition == scalar.statePosition,
            batchEmittedTokens: batch.emittedTokens,
            scalarEmittedTokens: scalar.emittedTokens)
    }
    for (batchRow, scalarRow) in zip(batch.logitTrace, scalar.logitTrace) {
        guard batchRow.count == scalarRow.count else {
            maxAbsoluteDifference = .infinity
            break
        }
        for (batchValue, scalarValue) in zip(batchRow, scalarRow) {
            maxAbsoluteDifference = max(
                maxAbsoluteDifference,
                abs(Float(batchValue) - Float(scalarValue)))
        }
    }
    return PrefillParityResult(
        comparedModes: [PrefillMode.batch.rawValue, PrefillMode.scalar.rawValue],
        logitsMatch: maxAbsoluteDifference <= 0.02,
        maxAbsoluteLogitDifference: maxAbsoluteDifference,
        emittedTokensMatch: batch.emittedTokens == scalar.emittedTokens,
        continuationPositionsMatch: batch.statePosition == scalar.statePosition,
        batchEmittedTokens: batch.emittedTokens,
        scalarEmittedTokens: scalar.emittedTokens)
}

private func run(_ rawArguments: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(rawArguments)
    } catch ArgumentError.helpRequested {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    do {
        let endToEndStart = DispatchTime.now().uptimeNanoseconds
        let setupStart = DispatchTime.now().uptimeNanoseconds
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: arguments.modelURL,
            device: context.device,
            expecting: .qwen38FlashNextText,
            streamingMode: .pread(slotCount: arguments.expertCacheSlots),
            expertCachePolicy: arguments.expertCachePolicy,
            denseCacheBytes: arguments.denseCacheBytes)
        let baseSetupSeconds = elapsedSeconds(since: setupStart)
        if arguments.prompt != nil || arguments.promptSequenceURL != nil {
            let prompts: [String]
            if let prompt = arguments.prompt {
                prompts = [prompt]
            } else if let promptSequenceURL = arguments.promptSequenceURL {
                prompts = try loadPromptSequence(from: promptSequenceURL)
            } else {
                throw ArgumentError.invalid("text completion requires a prompt")
            }
            let tokenizer = try await GFTokenizer.load(forModelDirectory: arguments.modelURL)
            let sharedRunner: Qwen38ForwardRunner?
            if arguments.repeatCount > 1 || prompts.count > 1 {
                let runtimeConfiguration = RuntimeConfiguration(
                    qwenGPUExecutionMode: arguments.qwenGPUExecutionMode,
                    ngramReadConcurrency: arguments.ngramReadConcurrency,
                    ngramRowCacheBytes: arguments.ngramRowCacheBytes,
                    ngramRowCacheMaxUniqueRows: arguments.ngramRowCacheMaxUniqueRows,
                    ngramRowProfileMaxRows: arguments.ngramRowProfileMaxRows,
                    ngramPinnedRows: arguments.ngramPinnedRows,
                    ngramPinnedRowBytes: arguments.ngramPinnedRowBytes)
                sharedRunner = try Qwen38ForwardRunner(
                    model: model,
                    context: context,
                    maxContext: arguments.maxContext,
                    runtimeConfiguration: runtimeConfiguration,
                    enableMTPDiagnostics: arguments.mtpDiagnostics,
                    mtpFCOrientation: arguments.mtpFCOrientation,
                    draftingStrategy: arguments.workload?.draftingEnabled == true
                        ? .experimentalNativeMTP
                        : .disabled,
                    targetLayerCount: arguments.targetLayerCount)
            } else {
                sharedRunner = nil
            }
            let setupSeconds = elapsedSeconds(since: setupStart)
            var receipts: [TextCompletionReceipt] = []
            receipts.reserveCapacity(arguments.repeatCount * prompts.count)
            for repeatIndex in 0..<arguments.repeatCount {
                for (promptIndex, prompt) in prompts.enumerated() {
                    let isFirstRun = repeatIndex == 0 && promptIndex == 0
                    let runStart = isFirstRun
                        ? endToEndStart
                        : DispatchTime.now().uptimeNanoseconds
                    let receipt = try await runTextCompletion(
                        arguments: arguments,
                        prompt: prompt,
                        model: model,
                        context: context,
                        tokenizer: tokenizer,
                        sharedRunner: sharedRunner,
                        setupSeconds: isFirstRun ? setupSeconds : 0,
                        endToEndStart: runStart)
                    receipts.append(receipt)
                }
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if arguments.repeatCount == 1 && prompts.count == 1 {
                FileHandle.standardOutput.write(try encoder.encode(receipts[0]))
            } else {
                let series = TextCompletionSeriesReceipt(
                    receiptType: "qwen38-text-completion-series",
                    repeatCount: arguments.repeatCount,
                    sequenceLength: prompts.count,
                    runs: receipts)
                FileHandle.standardOutput.write(try encoder.encode(series))
            }
            FileHandle.standardOutput.write(Data("\n".utf8))
            return 0
        }
        let primary = try await runMode(
            arguments: arguments,
            mode: arguments.prefillMode,
            model: model,
            context: context)
        var prefillParity: PrefillParityResult?
        if arguments.comparePrefillModes {
            let secondary = try await runMode(
                arguments: arguments,
                mode: oppositeMode(arguments.prefillMode),
                model: model,
                context: context)
            let batch = primary.mode == .batch ? primary : secondary
            let scalar = primary.mode == .scalar ? primary : secondary
            prefillParity = parityResult(batch: batch, scalar: scalar)
        }
        let endToEndSeconds = elapsedSeconds(since: endToEndStart)
        let result = ProbeResult(
            prefillMode: primary.mode.rawValue,
            qwenGPUExecutionMode: arguments.qwenGPUExecutionMode,
            workload: arguments.workload,
            ngramReadConcurrency: arguments.ngramReadConcurrency,
            ngramRowCacheBytes: arguments.ngramRowCacheBytes,
            ngramRowCacheMaxUniqueRows: arguments.ngramRowCacheMaxUniqueRows,
            ngramRowProfileMaxRows: arguments.ngramRowProfileMaxRows,
            ngramPinnedRows: arguments.ngramPinnedRows,
            ngramPinnedRowBytes: arguments.ngramPinnedRowBytes,
            boundaryToken: primary.boundaryToken,
            targetTokens: primary.verification?.targetTokens ?? primary.emittedTokens,
            semanticValidity: Qwen38SemanticValidity.from(
                tokenIDs: primary.verification?.targetTokens ?? primary.emittedTokens),
            acceptedTokenCount: primary.verification?.acceptedTokenCount
                ?? primary.emittedTokens.count,
            emittedTokens: primary.emittedTokens,
            statePosition: primary.statePosition,
            mtpTargetTokens: primary.verification?.targetTokens,
            mtpAcceptedTokenCount: primary.verification?.acceptedTokenCount,
            mtpStatePosition: primary.verification?.statePosition,
            mtpVerificationExecutionPath: primary.verification.map { _ in "batched-target" },
            mtpVerificationProposalCount: primary.mtpProposalTokens?.count,
            mtpVerificationTargetRowCount: primary.verification?.targetTokens.count,
            mtpRequestedBlockSize: primary.mtpRequestedBlockSize,
            mtpProposedTokens: primary.mtpProposalTokens,
            fixture: primary.fixture,
            nativeDraftStreamOrderDrafts: primary.nativeDraftStreamOrderDrafts,
            nativeDraftAlternateEmbeddingToken: primary.nativeDraftAlternateEmbeddingToken,
            nativeDraftBoundaryTargetToken: primary.nativeDraftBoundaryTargetToken,
            nativeDraftToken: primary.nativeDraftToken,
            nativeDraftFreshTargetToken: primary.nativeDraftFreshTargetToken,
            nativeDraftRawTargetToken: primary.nativeDraftRawTargetToken,
            nativeDraftAlternateToken: primary.nativeDraftAlternateToken,
            nativeDraftTargetToken: primary.nativeDraftTargetToken,
            nativeDraftMatchesTarget: primary.nativeDraftMatchesTarget,
            nativeDraftAlternateMatchesTarget: primary.nativeDraftAlternateMatchesTarget,
            nativeDraftError: primary.nativeDraftError,
            nativeDraftSeconds: primary.nativeDraftSeconds,
            nativeDraftTokensPerSecond: primary.nativeDraftTokensPerSecond,
            nativeDraftTargetPosition: primary.nativeDraftTargetPosition,
            nativeDraftMTPPosition: primary.nativeDraftMTPPosition,
            seedCaptureBytes: 0,
            verificationCaptureBytes: 0,
            setupSeconds: baseSetupSeconds + primary.setupSeconds,
            prefillSeconds: primary.prefillSeconds,
            decodeSeconds: primary.decodeSeconds,
            decodeTokensPerSecond: primary.decodeTokensPerSecond,
            decodeGPUActiveNanos: primary.decodeGPUActiveNanos,
            decodeCommandBufferEncodeNanos: primary.decodeCommandBufferEncodeNanos,
            decodeCommandBufferWaitNanos: primary.decodeCommandBufferWaitNanos,
            prefillCommandBufferEncodeNanos: primary.prefillWork?.commandBufferEncodeNanos,
            prefillCommandBufferWaitNanos: primary.prefillWork?.commandBufferWaitNanos,
            decodeTimingSamples: primary.decodeTimingSamples,
            speculativeReplay: primary.speculativeReplay,
            ngramCacheDiagnostics: primary.ngramCacheDiagnostics,
            memoryDiagnostics: primary.memoryDiagnostics,
            ngramRowProfile: primary.ngramRowProfile,
            prefillWork: primary.prefillWork,
            endToEndSeconds: endToEndSeconds,
            prefillParity: prefillParity,
            logitTraceChecksum: logitTraceChecksum(primary.logitTrace))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(result))
        FileHandle.standardOutput.write(Data("\n".utf8))
        return 0
    } catch {
        printError("verification failed: \(error)")
        return 1
    }
}

exit(await run(Array(CommandLine.arguments.dropFirst())))
