import Foundation
import Metal
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

@Suite
struct Qwen38MTPTests {
    @Test
    func loadsRelativeTensorNamesInStableOrder() throws {
        let context = try MetalContext()
        let buffer = try #require(context.device.makeBuffer(
            length: 128,
            options: .storageModeShared))
        let view = TensorView(
            buffer: buffer,
            offset: 0,
            length: 64,
            scaleOffset: 64,
            scaleLength: 4,
            biasOffset: 68,
            biasLength: 4,
            shape: (2, 32, 0, 0),
            dtype: GTurboFormatV1.DType.u32.rawValue,
            quantization: TensorQuantizationDescriptor(bits: 4, groupSize: 32))

        let weights = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: [
                "layers.0.output.weight": view,
                "layers.0.input_norm.weight": view,
            ])

        #expect(weights.predictLayers == 1)
        #expect(weights.tensorPrefix == "language_model.mtp.")
        #expect(weights.tensorNames == [
            "layers.0.input_norm.weight",
            "layers.0.output.weight",
        ])
        #expect(weights.contains(relativeName: "layers.0.input_norm.weight"))
        #expect(!weights.contains(relativeName: "layers.1.input_norm.weight"))
        #expect(try weights.tensor(relativeName: "layers.0.output.weight").shape.0 == 2)
        #expect(try weights.tensor(relativeName: "layers.0.output.weight").quantization
                == TensorQuantizationDescriptor(bits: 4, groupSize: 32))
    }

    @Test
    func rejectsInvalidDescriptorBeforeUsingTensorPayloads() {
        let result = Result {
            try Qwen38MTPWeights(
                predictLayers: 0,
                tensorPrefix: "language_model.mtp.",
                tensors: [:])
        }

        #expect {
            try result.get()
        } throws: { error in
            guard case ModelError.archMismatch(field: "mtp", _, _) = error else {
                return false
            }
            return true
        }
    }

    @Test
    func rejectsQwen38ModelWithoutMTPMetadata() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: .gemma4Toy())

        #expect {
            try Qwen38MTP(model: model)
        } throws: { error in
            guard case ModelError.archMismatch(field: "modelFamily", _, _) = error else {
                return false
            }
            return true
        }
    }

    @Test
    func rejectsMissingRequiredExecutionRole() throws {
        let context = try MetalContext()
        var tensors = try Self.validExecutionTensors(context: context)
        tensors.removeValue(forKey: Qwen38MTPRole.queryProjection.rawValue)
        let weights = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: tensors)

        #expect {
            try weights.validateExecutionRoles()
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else {
                return false
            }
            return detail.contains(Qwen38MTPRole.queryProjection.rawValue)
        }
    }

    @Test
    func rejectsMismatchedExecutionRoleShape() throws {
        let context = try MetalContext()
        var tensors = try Self.validExecutionTensors(context: context)
        tensors[Qwen38MTPRole.queryProjection.rawValue] = Self.tensor(
            buffer: tensors[Qwen38MTPRole.queryProjection.rawValue]!.buffer,
            shape: [12_288, 2_528],
            quantized: true)
        let weights = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: tensors)

        #expect {
            try weights.validateExecutionRoles()
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else {
                return false
            }
            return detail.contains("shape mismatch")
        }
    }

    @Test
    func buildsExecutionContractFromValidatedRoles() throws {
        let context = try MetalContext()
        let weights = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: Self.validExecutionTensors(context: context))

        let contract = try Qwen38MTPExecutionContract(weights: weights)

        #expect(contract.predictLayers == 1)
        #expect(contract.stateLayout == .qsaAndFullAttention)
        #expect(contract.geometry == .qwen)
        #expect(contract.geometry.fullAttentionQueryHeads == 24)
        #expect(contract.geometry.fullAttentionKeyValueHeads == 2)
    }

    @Test
    func bindsBothMTPHyperConnectionBranchesToCanonicalGeometry() throws {
        let context = try MetalContext()
        let weights = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: Self.validExecutionTensors(context: context))

        let attention = try Qwen38MTPHyperConnectionWeights(
            mtp: weights,
            branch: .attention)
        let mlp = try Qwen38MTPHyperConnectionWeights(
            mtp: weights,
            branch: .mlp)
        let executor = try Qwen38MTPHyperConnectionExecutor(context: context)

        for branch in [attention, mlp] {
            #expect(branch.geometry.streamCount == 4)
            #expect(branch.geometry.hiddenSize == 2_560)
            #expect(branch.geometry.lowRankSize == 320)
            #expect(branch.weights.blockInject != nil)
        }
        #expect(executor.geometry == attention.geometry)
    }

    @Test
    func draftingStrategyRequiresExplicitOptIn() {
        #expect(Qwen38DraftingStrategy.disabled.isEnabled == false)
        #expect(Qwen38DraftingStrategy.experimentalNativeMTP.isEnabled)
        #expect(Qwen38DraftingStrategy.experimentalNativeMTP.rawValue
                == "experimental-native-mtp")
    }

    @Test
    func draftingStrategyRoundTripsThroughCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let strategies: [Qwen38DraftingStrategy] = [
            .disabled,
            .experimentalNativeMTP,
        ]

        for strategy in strategies {
            let encoded = try encoder.encode(strategy)
            let decoded = try decoder.decode(
                Qwen38DraftingStrategy.self,
                from: encoded)

            #expect(decoded == strategy)
            #expect(String(data: encoded, encoding: .utf8)
                    == "\"\(strategy.rawValue)\"")
        }
    }

    @Test
    func validatedMTPWeightsUseTargetOnlyFallback() {
        let capability = Qwen38MTPExecutionCapability.validatedWeightsOnly

        #expect(Qwen38MTPExecutionCapability.unavailable
                .supportsNativeDraftGeneration == false)
        #expect(capability.supportsNativeDraftGeneration == false)
        #expect(capability != .nativeDraft)
        #expect(Qwen38MTPExecutionCapability.nativeDraft
                .supportsNativeDraftGeneration)
    }

    @Test
    func rejectsNonCanonicalExecutionGeometry() throws {
        let context = try MetalContext()
        let weights = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: Self.validExecutionTensors(context: context))
        let geometry = Qwen38MTPExecutionGeometry(
            hiddenSize: 2_528,
            streamCount: 4,
            lowRankSize: 320,
            indexerHeadDimension: 128,
            fullAttentionHeadDimension: 256)

        #expect {
            try Qwen38MTPExecutionContract(weights: weights, geometry: geometry)
        } throws: { error in
            guard case ModelError.archMismatch(let field, _, _) = error else {
                return false
            }
            return field == "mtp.executionGeometry"
        }
    }

    @Test
    func rejectsInvalidInputFusionGeometryBeforeAllocation() throws {
        let context = try MetalContext()

        #expect {
            try Qwen38MTPInputFusionScratch(
                device: context.device,
                geometry: Qwen38MTPInputFusionGeometry(hiddenSize: 0))
        } throws: { error in
            guard case ModelError.archMismatch(let field, _, _) = error else {
                return false
            }
            return field == "mtp.inputFusion.hiddenSize"
        }
    }

    @Test
    func inputFusionPreservesFourStreamLogicalWidth() throws {
        let context = try MetalContext()
        let geometry = Qwen38MTPInputFusionGeometry.qwen
        let scratch = try Qwen38MTPInputFusionScratch(
            device: context.device,
            geometry: geometry)
        let elementBytes = MemoryLayout<Float16>.stride

        #expect(geometry.hiddenSize == 2_560)
        #expect(geometry.streamCount == 4)
        #expect(geometry.hyperWidth == 10_240)
        #expect(geometry.hiddenNormWidth == 10_240)
        #expect(scratch.normalizedEmbedding.length == 2_560 * elementBytes)
        #expect(scratch.normalizedHidden.length == geometry.hiddenNormWidth * elementBytes)
        #expect(scratch.projectedEmbedding.length == 2_560 * elementBytes)
        #expect(scratch.expandedEmbedding.length == geometry.hyperWidth * elementBytes)
        #expect(scratch.projectedHidden.length == geometry.hyperWidth * elementBytes)
        #expect(scratch.output.length == geometry.hyperWidth * elementBytes)
    }

    @Test
    func rejectsInvalidInputFusionStreamCountBeforeAllocation() throws {
        let context = try MetalContext()

        #expect {
            try Qwen38MTPInputFusionScratch(
                device: context.device,
                geometry: Qwen38MTPInputFusionGeometry(
                    hiddenSize: 2_560,
                    streamCount: 0))
        } throws: { error in
            guard case ModelError.archMismatch(let field, _, _) = error else {
                return false
            }
            return field == "mtp.inputFusion.streamCount"
        }
    }

    private static func validExecutionTensors(
        context: MetalContext
    ) throws -> [String: TensorView] {
        guard let buffer = context.device.makeBuffer(
            length: 128,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return Dictionary(uniqueKeysWithValues: Qwen38MTPRole.allCases.map { role in
            (role.rawValue, tensor(
                buffer: buffer,
                shape: expectedShape(for: role),
                quantized: isQuantized(role)))
        })
    }

    private static func tensor(
        buffer: MTLBuffer,
        shape: [UInt32],
        quantized: Bool
    ) -> TensorView {
        let elementCount = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
        let shape4 = shape + Array(repeating: UInt32(0), count: 4 - shape.count)
        if quantized {
            let auxiliaryLength = elementCount / 32 * 2
            return TensorView(
                buffer: buffer,
                offset: 0,
                length: elementCount / 2,
                scaleOffset: 0,
                scaleLength: auxiliaryLength,
                biasOffset: 0,
                biasLength: auxiliaryLength,
                shape: (shape4[0], shape4[1], shape4[2], shape4[3]),
                dtype: GTurboFormatV1.DType.u32.rawValue,
                quantization: TensorQuantizationDescriptor(bits: 4, groupSize: 32))
        }
        return TensorView(
            buffer: buffer,
            offset: 0,
            length: elementCount * 2,
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (shape4[0], shape4[1], shape4[2], shape4[3]),
            dtype: GTurboFormatV1.DType.bf16.rawValue,
            quantization: nil)
    }

    private static func isQuantized(_ role: Qwen38MTPRole) -> Bool {
        switch role {
        case .preFCNormEmbedding, .preFCNormHidden,
             .hyperConnectionMixerNorm, .attentionHyperConnectionNorm,
             .mlpHyperConnectionNorm, .mlpRouter, .indexerKeyNorm,
             .indexerQueryNorm, .queryNorm, .keyNorm:
            return false
        default:
            return true
        }
    }

    private static func expectedShape(for role: Qwen38MTPRole) -> [UInt32] {
        switch role {
        case .preFCNormEmbedding:
            return [2_560]
        case .preFCNormHidden, .hyperConnectionMixerNorm,
             .attentionHyperConnectionNorm, .mlpHyperConnectionNorm:
            return [10_240]
        case .mlpRouter:
            return [512, 2_560]
        case .indexerKeyNorm, .indexerQueryNorm:
            return [128]
        case .queryNorm, .keyNorm:
            return [256]
        case .fcEmbedding, .fcHidden:
            return [2_560, 2_560]
        case .hyperConnectionMixerInputMixDown,
             .attentionHyperConnectionInputMixDown,
             .mlpHyperConnectionInputMixDown:
            return [320, 10_240]
        case .hyperConnectionMixerInputMixUp,
             .attentionHyperConnectionInputMixUp,
             .mlpHyperConnectionInputMixUp:
            return [10_240, 320]
        case .attentionHyperConnectionBlockInject,
             .mlpHyperConnectionBlockInject:
            return [4, 10_240]
        case .sharedExpertGate, .sharedExpertUp:
            return [640, 2_560]
        case .sharedExpertDown:
            return [2_560, 640]
        case .sharedExpertMultiplier:
            return [1, 2_560]
        case .switchExpertGate, .switchExpertUp:
            return [512, 640, 2_560]
        case .switchExpertDown:
            return [512, 2_560, 640]
        case .indexerProjection:
            return [640, 2_560]
        case .queryProjection:
            return [12_288, 2_560]
        case .keyProjection, .valueProjection:
            return [512, 2_560]
        case .outputProjection:
            return [2_560, 6_144]
        }
    }
}
