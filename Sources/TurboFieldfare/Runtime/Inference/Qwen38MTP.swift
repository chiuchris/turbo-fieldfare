import Foundation
import Metal
import TurboFieldfareFormat

public enum Qwen38MTPRole: String, CaseIterable, Sendable {
    case fcEmbedding = "fc_embedding.weight"
    case fcHidden = "fc_hidden.weight"
    case preFCNormEmbedding = "pre_fc_norm_embedding.weight"
    case preFCNormHidden = "pre_fc_norm_hidden.weight"
    case hyperConnectionMixerNorm = "hyper_connection_mixer.hc_norm.weight"
    case hyperConnectionMixerInputMixDown =
        "hyper_connection_mixer.input_mix_weight_down.weight"
    case hyperConnectionMixerInputMixUp =
        "hyper_connection_mixer.input_mix_weight_up.weight"
    case attentionHyperConnectionNorm =
        "layers.0.attn_hyper_connection.hc_norm.weight"
    case attentionHyperConnectionInputMixDown =
        "layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    case attentionHyperConnectionInputMixUp =
        "layers.0.attn_hyper_connection.input_mix_weight_up.weight"
    case attentionHyperConnectionBlockInject =
        "layers.0.attn_hyper_connection.block_inject_weight.weight"
    case mlpHyperConnectionNorm = "layers.0.mlp_hyper_connection.hc_norm.weight"
    case mlpHyperConnectionInputMixDown =
        "layers.0.mlp_hyper_connection.input_mix_weight_down.weight"
    case mlpHyperConnectionInputMixUp =
        "layers.0.mlp_hyper_connection.input_mix_weight_up.weight"
    case mlpHyperConnectionBlockInject =
        "layers.0.mlp_hyper_connection.block_inject_weight.weight"
    case mlpRouter = "layers.0.mlp.gate.weight"
    case sharedExpertGate = "layers.0.mlp.shared_expert.gate_proj.weight"
    case sharedExpertUp = "layers.0.mlp.shared_expert.up_proj.weight"
    case sharedExpertDown = "layers.0.mlp.shared_expert.down_proj.weight"
    case sharedExpertMultiplier = "layers.0.mlp.shared_expert_gate.weight"
    case switchExpertGate = "layers.0.mlp.switch_mlp.gate_proj.weight"
    case switchExpertUp = "layers.0.mlp.switch_mlp.up_proj.weight"
    case switchExpertDown = "layers.0.mlp.switch_mlp.down_proj.weight"
    case indexerProjection =
        "layers.0.self_attn.indexer.index_qk_proj.weight"
    case indexerKeyNorm = "layers.0.self_attn.indexer.k_layernorm.weight"
    case indexerQueryNorm = "layers.0.self_attn.indexer.q_layernorm.weight"
    case queryProjection = "layers.0.self_attn.q_proj.weight"
    case keyProjection = "layers.0.self_attn.k_proj.weight"
    case valueProjection = "layers.0.self_attn.v_proj.weight"
    case outputProjection = "layers.0.self_attn.o_proj.weight"
    case queryNorm = "layers.0.self_attn.q_norm.weight"
    case keyNorm = "layers.0.self_attn.k_norm.weight"
}

/// Resident Qwen3.8 MTP weights discovered from the manifest prefix.
///
/// The external checkpoint role list is not part of the wire contract, so this
/// container keeps relative tensor names instead of guessing a fixed schema.
/// The forward executor can validate the roles it consumes when that path is
/// added. The model initializer is also used by diagnostic executable targets.
public struct Qwen38MTPWeights: @unchecked Sendable {
    public let predictLayers: Int
    public let tensorPrefix: String
    private let tensors: [String: TensorView]

    public var tensorNames: [String] {
        tensors.keys.sorted()
    }

    public var hasRequiredExecutionRoles: Bool {
        Set(Qwen38MTPRole.allCases.map(\.rawValue)).isSubset(of: tensors.keys)
    }

    public init(predictLayers: Int,
                tensorPrefix: String,
                tensors: [String: TensorView]) throws {
        guard predictLayers > 0, !tensorPrefix.isEmpty else {
            throw ModelError.archMismatch(
                field: "mtp",
                expected: "a positive layer count and non-empty tensor prefix",
                actual: "invalid descriptor")
        }
        guard !tensors.isEmpty else {
            throw ModelError.indexCorrupt(
                detail: "MTP descriptor is present but no tensors were loaded")
        }
        guard tensors.keys.allSatisfy({ !$0.isEmpty && !$0.contains(tensorPrefix) }) else {
            throw ModelError.indexCorrupt(
                detail: "MTP tensor names must be non-empty and manifest-relative")
        }
        self.predictLayers = predictLayers
        self.tensorPrefix = tensorPrefix
        self.tensors = tensors
    }

    public init(model: Model) throws {
        guard model.config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(model.config.modelFamily)")
        }
        guard let metadata = model.mtpMetadata,
              metadata.predictLayers > 0,
              !metadata.tensorPrefix.isEmpty else {
            throw ModelError.archMismatch(
                field: "mtp",
                expected: "a positive layer count and non-empty tensor prefix",
                actual: "missing or invalid")
        }

        let fullNames = model.residentIndex.entries.keys
            .filter { $0.hasPrefix(metadata.tensorPrefix) }
            .sorted()
        guard !fullNames.isEmpty else {
            throw ModelError.indexCorrupt(
                detail: "MTP descriptor is present but no prefixed resident tensors were found")
        }

        var tensors: [String: TensorView] = [:]
        tensors.reserveCapacity(fullNames.count)
        for fullName in fullNames {
            let relativeName = String(fullName.dropFirst(metadata.tensorPrefix.count))
            guard !relativeName.isEmpty else {
                throw ModelError.indexCorrupt(
                    detail: "MTP tensor prefix is also a tensor name")
            }
            guard tensors[relativeName] == nil else {
                throw ModelError.indexCorrupt(
                    detail: "duplicate MTP tensor name \(relativeName)")
            }
            tensors[relativeName] = try model.mtpTensor(name: fullName)
        }

        try Self.validateExecutionRoles(predictLayers: metadata.predictLayers,
                                        tensors: tensors)
        self.predictLayers = metadata.predictLayers
        self.tensorPrefix = metadata.tensorPrefix
        self.tensors = tensors
    }

    func validateExecutionRoles() throws {
        try Self.validateExecutionRoles(predictLayers: predictLayers, tensors: tensors)
    }

    private struct ExecutionRoleSpec {
        let shape: [UInt32]
        let quantized: Bool
    }

    private static func validateExecutionRoles(
        predictLayers: Int,
        tensors: [String: TensorView]
    ) throws {
        guard predictLayers == 1 else {
            throw ModelError.archMismatch(
                field: "mtp.predictLayers",
                expected: "1",
                actual: "\(predictLayers)")
        }

        for role in Qwen38MTPRole.allCases {
            guard let tensor = tensors[role.rawValue] else {
                throw ModelError.indexCorrupt(
                    detail: "missing required MTP tensor \(role.rawValue)")
            }
            try validate(tensor, role: role, spec: executionRoleSpec(role))
        }
    }

    private static func executionRoleSpec(_ role: Qwen38MTPRole) -> ExecutionRoleSpec {
        switch role {
        case .preFCNormEmbedding:
            return ExecutionRoleSpec(shape: [2_560], quantized: false)
        case .preFCNormHidden, .hyperConnectionMixerNorm,
             .attentionHyperConnectionNorm, .mlpHyperConnectionNorm:
            return ExecutionRoleSpec(shape: [10_240], quantized: false)
        case .mlpRouter:
            return ExecutionRoleSpec(shape: [512, 2_560], quantized: false)
        case .indexerKeyNorm, .indexerQueryNorm:
            return ExecutionRoleSpec(shape: [128], quantized: false)
        case .queryNorm, .keyNorm:
            return ExecutionRoleSpec(shape: [256], quantized: false)
        case .fcEmbedding, .fcHidden:
            return ExecutionRoleSpec(shape: [2_560, 2_560], quantized: true)
        case .hyperConnectionMixerInputMixDown,
             .attentionHyperConnectionInputMixDown,
             .mlpHyperConnectionInputMixDown:
            return ExecutionRoleSpec(shape: [320, 10_240], quantized: true)
        case .hyperConnectionMixerInputMixUp,
             .attentionHyperConnectionInputMixUp,
             .mlpHyperConnectionInputMixUp:
            return ExecutionRoleSpec(shape: [10_240, 320], quantized: true)
        case .attentionHyperConnectionBlockInject,
             .mlpHyperConnectionBlockInject:
            return ExecutionRoleSpec(shape: [4, 10_240], quantized: true)
        case .sharedExpertGate, .sharedExpertUp:
            return ExecutionRoleSpec(shape: [640, 2_560], quantized: true)
        case .sharedExpertDown:
            return ExecutionRoleSpec(shape: [2_560, 640], quantized: true)
        case .sharedExpertMultiplier:
            return ExecutionRoleSpec(shape: [1, 2_560], quantized: true)
        case .switchExpertGate, .switchExpertUp:
            return ExecutionRoleSpec(shape: [512, 640, 2_560], quantized: true)
        case .switchExpertDown:
            return ExecutionRoleSpec(shape: [512, 2_560, 640], quantized: true)
        case .indexerProjection:
            return ExecutionRoleSpec(shape: [640, 2_560], quantized: true)
        case .queryProjection:
            return ExecutionRoleSpec(shape: [12_288, 2_560], quantized: true)
        case .keyProjection, .valueProjection:
            return ExecutionRoleSpec(shape: [512, 2_560], quantized: true)
        case .outputProjection:
            return ExecutionRoleSpec(shape: [2_560, 6_144], quantized: true)
        }
    }

    private static func validate(
        _ tensor: TensorView,
        role: Qwen38MTPRole,
        spec: ExecutionRoleSpec
    ) throws {
        guard spec.shape.count <= 4 else {
            throw ModelError.indexCorrupt(
                detail: "MTP role \(role.rawValue) has unsupported rank")
        }
        let expectedShape = spec.shape + Array(
            repeating: UInt32(0), count: 4 - spec.shape.count)
        let actualShape = [tensor.shape.0, tensor.shape.1, tensor.shape.2, tensor.shape.3]
        guard actualShape == expectedShape else {
            throw ModelError.indexCorrupt(
                detail: "MTP role \(role.rawValue) shape mismatch")
        }

        var elementCount: UInt64 = 1
        for dimension in spec.shape {
            let (product, overflow) = elementCount.multipliedReportingOverflow(
                by: UInt64(dimension))
            guard !overflow else {
                throw ModelError.indexCorrupt(
                    detail: "MTP role \(role.rawValue) element count overflows UInt64")
            }
            elementCount = product
        }

        if spec.quantized {
            let groupSize: UInt64 = 32
            guard tensor.dtype == GTurboFormatV1.DType.u32.rawValue,
                  tensor.quantization == TensorQuantizationDescriptor(bits: 4, groupSize: 32),
                  elementCount.isMultiple(of: groupSize),
                  tensor.length == elementCount / 2,
                  tensor.scaleLength == (elementCount / groupSize) * 2,
                  tensor.biasLength == (elementCount / groupSize) * 2,
                  tensor.offset.isMultiple(of: UInt64(MemoryLayout<UInt32>.alignment)),
                  tensor.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
                  tensor.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
                throw ModelError.indexCorrupt(
                    detail: "MTP role \(role.rawValue) canonical Q4 metadata mismatch")
            }
        } else {
            guard tensor.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  tensor.length == elementCount * 2,
                  tensor.scaleLength == 0,
                  tensor.biasLength == 0,
                  tensor.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
                throw ModelError.indexCorrupt(
                    detail: "MTP role \(role.rawValue) BF16 metadata mismatch")
            }
        }
    }

    public func tensor(relativeName: String) throws -> TensorView {
        guard let tensor = tensors[relativeName] else {
            throw ModelError.tensorNotFound(name: tensorPrefix + relativeName)
        }
        return tensor
    }

    public func tensor(role: Qwen38MTPRole) throws -> TensorView {
        try tensor(relativeName: role.rawValue)
    }

    public func contains(relativeName: String) -> Bool {
        tensors[relativeName] != nil
    }
}

public enum Qwen38MTPStateLayout: String, Sendable, Equatable {
    case qsaAndFullAttention
}

public struct Qwen38MTPExecutionGeometry: Sendable, Equatable {
    public let hiddenSize: Int
    public let streamCount: Int
    public let lowRankSize: Int
    public let indexerHeadDimension: Int
    public let fullAttentionQueryHeads: Int
    public let fullAttentionKeyValueHeads: Int
    public let fullAttentionHeadDimension: Int

    public static let qwen = Qwen38MTPExecutionGeometry(
        hiddenSize: 2_560,
        streamCount: 4,
        lowRankSize: 320,
        indexerHeadDimension: 128,
        fullAttentionHeadDimension: 256,
        fullAttentionQueryHeads: 24,
        fullAttentionKeyValueHeads: 2)

    public init(hiddenSize: Int,
                streamCount: Int,
                lowRankSize: Int,
                indexerHeadDimension: Int,
                fullAttentionHeadDimension: Int,
                fullAttentionQueryHeads: Int = 24,
                fullAttentionKeyValueHeads: Int = 2) {
        self.hiddenSize = hiddenSize
        self.streamCount = streamCount
        self.lowRankSize = lowRankSize
        self.indexerHeadDimension = indexerHeadDimension
        self.fullAttentionQueryHeads = fullAttentionQueryHeads
        self.fullAttentionKeyValueHeads = fullAttentionKeyValueHeads
        self.fullAttentionHeadDimension = fullAttentionHeadDimension
    }
}

public enum Qwen38MTPFCOrientation: String, Codable, Sendable, Equatable {
    case normal
    case transposeEmbedding = "transpose-embedding"
    case transposeHidden = "transpose-hidden"
    case transposeBoth = "transpose-both"

    var transposeEmbedding: Bool {
        self == .transposeEmbedding || self == .transposeBoth
    }

    var transposeHidden: Bool {
        self == .transposeHidden || self == .transposeBoth
    }
}

/// The validated hand-off between MTP tensor loading and draft execution.
/// Keeping this contract explicit prevents a loaded weight container from being
/// mistaken for an executable draft model.
public struct Qwen38MTPExecutionContract: Sendable, Equatable {
    public let geometry: Qwen38MTPExecutionGeometry
    public let stateLayout: Qwen38MTPStateLayout
    public let predictLayers: Int

    public init(weights: Qwen38MTPWeights,
                geometry: Qwen38MTPExecutionGeometry = .qwen,
                stateLayout: Qwen38MTPStateLayout = .qsaAndFullAttention) throws {
        try weights.validateExecutionRoles()
        guard geometry.hiddenSize == 2_560,
              geometry.streamCount == 4,
              geometry.lowRankSize == 320,
              geometry.indexerHeadDimension == 128,
              geometry.fullAttentionQueryHeads == 24,
              geometry.fullAttentionKeyValueHeads == 2,
              geometry.fullAttentionHeadDimension == 256 else {
            throw ModelError.archMismatch(
                field: "mtp.executionGeometry",
                expected: "Qwen3.8 Flash-Next geometry",
                actual: "custom geometry")
        }
        self.geometry = geometry
        self.stateLayout = stateLayout
        self.predictLayers = weights.predictLayers
    }
}

struct Qwen38MTPInputFusionGeometry: Sendable, Equatable {
    let hiddenSize: Int
    let streamCount: Int

    var hyperWidth: Int { hiddenSize * streamCount }
    var hiddenNormWidth: Int { hyperWidth }

    static let qwen = Qwen38MTPInputFusionGeometry(
        hiddenSize: 2_560,
        streamCount: 4)

    init(hiddenSize: Int, streamCount: Int = 4) {
        self.hiddenSize = hiddenSize
        self.streamCount = streamCount
    }
}

struct Qwen38MTPInputFusionScratch {
    let normalizedEmbedding: MTLBuffer
    let normalizedHidden: MTLBuffer
    let projectedEmbedding: MTLBuffer
    let expandedEmbedding: MTLBuffer
    let projectedHidden: MTLBuffer
    let output: MTLBuffer

    init(device: MTLDevice,
         geometry: Qwen38MTPInputFusionGeometry = .qwen) throws {
        guard geometry.hiddenSize > 0 else {
            throw ModelError.archMismatch(
                field: "mtp.inputFusion.hiddenSize",
                expected: "positive",
                actual: "\(geometry.hiddenSize)")
        }
        guard geometry.streamCount > 0 else {
            throw ModelError.archMismatch(
                field: "mtp.inputFusion.streamCount",
                expected: "positive",
                actual: "\(geometry.streamCount)")
        }
        func makeBuffer(elements: Int) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: elements * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }
        self.normalizedEmbedding = try makeBuffer(elements: geometry.hiddenSize)
        self.normalizedHidden = try makeBuffer(elements: geometry.hiddenNormWidth)
        self.projectedEmbedding = try makeBuffer(elements: geometry.hiddenSize)
        self.expandedEmbedding = try makeBuffer(elements: geometry.hyperWidth)
        self.projectedHidden = try makeBuffer(elements: geometry.hyperWidth)
        self.output = try makeBuffer(elements: geometry.hyperWidth)
    }
}

struct Qwen38MTPInputFusionWeights {
    let embeddingNorm: TensorView
    let hiddenNorm: TensorView
    let embeddingProjection: Qwen38PLEQuantizedProjection
    let hiddenProjection: Qwen38PLEQuantizedProjection

    init(mtp: Qwen38MTP) throws {
        self.embeddingNorm = try mtp.tensor(role: .preFCNormEmbedding)
        self.hiddenNorm = try mtp.tensor(role: .preFCNormHidden)
        self.embeddingProjection = Self.projection(
            try mtp.tensor(role: .fcEmbedding))
        self.hiddenProjection = Self.projection(
            try mtp.tensor(role: .fcHidden))
    }

    private static func projection(
        _ view: TensorView
    ) -> Qwen38PLEQuantizedProjection {
        Qwen38PLEQuantizedProjection(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset))
    }
}

final class Qwen38MTPInputFusion {
    let geometry: Qwen38MTPInputFusionGeometry
    private let fcOrientation: Qwen38MTPFCOrientation
    private let gatedResidual: Qwen38GatedResidual
    private let projection: Qwen38PLEProjection
    private let elementwise: QwenElementwise

    init(context: MetalContext,
         geometry: Qwen38MTPInputFusionGeometry = .qwen,
         fcOrientation: Qwen38MTPFCOrientation = .normal) throws {
        precondition(geometry.hiddenSize > 0 && geometry.streamCount > 0)
        self.geometry = geometry
        self.fcOrientation = fcOrientation
        self.gatedResidual = try Qwen38GatedResidual(context: context)
        self.projection = try Qwen38PLEProjection(context: context)
        self.elementwise = try QwenElementwise(context: context)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                embedding: MTLBuffer,
                hidden: MTLBuffer,
                weights: Qwen38MTPInputFusionWeights,
                scratch: Qwen38MTPInputFusionScratch,
                epsilon: Float) {
        let hiddenBytes = geometry.hiddenSize * MemoryLayout<Float16>.stride
        let hiddenNormBytes = geometry.hiddenNormWidth * MemoryLayout<Float16>.stride
        let hyperBytes = geometry.hyperWidth * MemoryLayout<Float16>.stride
        precondition(embedding.length >= hiddenBytes)
        precondition(hidden.length >= hyperBytes)
        precondition(scratch.normalizedEmbedding.length >= hiddenBytes)
        precondition(scratch.normalizedHidden.length >= hiddenNormBytes)
        precondition(scratch.projectedEmbedding.length >= hiddenBytes)
        precondition(scratch.expandedEmbedding.length >= hyperBytes)
        precondition(scratch.projectedHidden.length >= hyperBytes)
        precondition(scratch.output.length >= hyperBytes)

        gatedResidual.encodeZeroCenteredRMSNorm(
            commandBuffer: commandBuffer,
            input: embedding,
            weight: weights.embeddingNorm.buffer,
            weightOffset: Int(weights.embeddingNorm.offset),
            output: scratch.normalizedEmbedding,
            tokenCount: 1,
            width: UInt32(geometry.hiddenSize),
            epsilon: epsilon)
        gatedResidual.encodeZeroCenteredRMSNorm(
            commandBuffer: commandBuffer,
            input: hidden,
            weight: weights.hiddenNorm.buffer,
            weightOffset: Int(weights.hiddenNorm.offset),
            output: scratch.normalizedHidden,
            tokenCount: 1,
            width: UInt32(geometry.hiddenNormWidth),
            epsilon: epsilon)
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.embeddingProjection.weights,
            weightsOffset: weights.embeddingProjection.weightsOffset,
            scales: weights.embeddingProjection.scales,
            scalesOffset: weights.embeddingProjection.scalesOffset,
            biases: weights.embeddingProjection.biases,
            biasesOffset: weights.embeddingProjection.biasesOffset,
            input: scratch.normalizedEmbedding,
            output: scratch.projectedEmbedding,
            tokenCount: 1,
            outputWidth: UInt32(geometry.hiddenSize),
            inputWidth: UInt32(geometry.hiddenSize),
            transposeWeights: fcOrientation.transposeEmbedding)
        gatedResidual.encodeRepeatStreams(
            commandBuffer: commandBuffer,
            input: scratch.projectedEmbedding,
            output: scratch.expandedEmbedding,
            tokenCount: 1,
            streamCount: UInt32(geometry.streamCount),
            hiddenSize: UInt32(geometry.hiddenSize))
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.hiddenProjection.weights,
            weightsOffset: weights.hiddenProjection.weightsOffset,
            scales: weights.hiddenProjection.scales,
            scalesOffset: weights.hiddenProjection.scalesOffset,
            biases: weights.hiddenProjection.biases,
            biasesOffset: weights.hiddenProjection.biasesOffset,
            input: scratch.normalizedHidden,
            output: scratch.projectedHidden,
            tokenCount: UInt32(geometry.streamCount),
            outputWidth: UInt32(geometry.hiddenSize),
            inputWidth: UInt32(geometry.hiddenSize),
            transposeWeights: fcOrientation.transposeHidden)
        elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer,
            lhs: scratch.expandedEmbedding,
            rhs: scratch.projectedHidden,
            output: scratch.output,
            count: UInt32(geometry.hyperWidth))
    }
}

public enum Qwen38DraftingStrategy: String, Codable, Sendable, Equatable {
    case disabled
    case experimentalNativeMTP = "experimental-native-mtp"

    public var isEnabled: Bool {
        self != .disabled
    }
}

public enum Qwen38MTPExecutionCapability: Sendable, Equatable {
    case unavailable
    /// The checkpoint passed metadata validation, but no native draft executor
    /// has been proven for its attention, hyper-connection, and switch-MoE layouts.
    case validatedWeightsOnly
    case nativeDraft

    public var supportsNativeDraftGeneration: Bool {
        self == .nativeDraft
    }
}

/// Production MTP weight boundary. Execution is intentionally kept separate
/// from loading so draft state can be owned independently of target state.
public final class Qwen38MTP: @unchecked Sendable {
    public let weights: Qwen38MTPWeights
    public let executionContract: Qwen38MTPExecutionContract
    public let executionCapability: Qwen38MTPExecutionCapability

    public var predictLayers: Int { weights.predictLayers }
    public var supportsNativeDraftGeneration: Bool {
        executionCapability.supportsNativeDraftGeneration
    }

    public init(model: Model) throws {
        let weights = try Qwen38MTPWeights(model: model)
        self.weights = weights
        self.executionContract = try Qwen38MTPExecutionContract(weights: weights)
        self.executionCapability = .nativeDraft
    }

    public func tensor(relativeName: String) throws -> TensorView {
        try weights.tensor(relativeName: relativeName)
    }

    public func tensor(role: Qwen38MTPRole) throws -> TensorView {
        try weights.tensor(role: role)
    }
}
