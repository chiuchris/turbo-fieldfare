import Metal
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

@Suite
struct Qwen38MTPExecutorTests {
    @Test
    func mtpDiagnosticsAreOptIn() {
        #expect(!Qwen38MTPDiagnosticMode.off.isEnabled)
        #expect(Qwen38MTPDiagnosticMode.on.isEnabled)
    }

    @Test
    func acceptsBoundedMTPDraftBlockAndAdvancesCursor() throws {
        let block = try Qwen38MTPDraftBlock(
            tokens: [11, 12, 13, 14],
            startPosition: 20)

        #expect(block.tokens == [11, 12, 13, 14])
        #expect(block.tokenCount == 4)
        #expect(block.startPosition == 20)
        #expect(block.endPosition == 24)
    }

    @Test
    func rejectsInvalidMTPDraftBlockCapacity() {
        #expect {
            try Qwen38MTPDraftBlock(tokens: [], startPosition: 0)
        } throws: { error in
            guard case PrefillError.chunkedUnsupported(let reason) = error else {
                return false
            }
            return reason.contains("at least one token")
        }

        #expect {
            try Qwen38MTPDraftBlock(
                tokens: Array(repeating: 1, count: Qwen38MTPDraftBlock.maxTokenCount + 1),
                startPosition: 0)
        } throws: { error in
            guard case PrefillError.chunkedUnsupported(let reason) = error else {
                return false
            }
            return reason.contains("at most")
        }
    }

    @Test
    func rejectsNegativeMTPDraftBlockStartPosition() {
        #expect {
            try Qwen38MTPDraftBlock(tokens: [1], startPosition: -1)
        } throws: { error in
            guard case PrefillError.prefillCursorMismatch(let reason) = error else {
                return false
            }
            return reason.contains("non-negative")
        }
    }

    @Test
    func usesTargetRowRotaryPositionForMTPAttention() {
        #expect(Qwen38MTPAttentionExecutor.rotaryPosition(for: 0) == 0)
        #expect(Qwen38MTPAttentionExecutor.rotaryPosition(for: 7) == 7)
    }

    @Test
    func usesMTPFullAttentionGeometry() throws {
        let context = try MetalContext()
        let executor = try Qwen38MTPAttentionExecutor(context: context)

        #expect(executor.geometry.queryHeads == 24)
        #expect(executor.geometry.keyValueHeads == 2)
        #expect(executor.geometry.headDimension == 256)
        #expect(executor.geometry.queryWidth == 6_144)
        #expect(executor.geometry.keyValueWidth == 512)
    }

    @Test
    func allocatesMTPAttentionScratchForWiderProjections() throws {
        let context = try MetalContext()
        let scratch = try Qwen38MTPAttentionScratch(
            device: context.device,
            maxContext: 8)
        let fp16Bytes = MemoryLayout<Float16>.stride

        #expect(scratch.projection.length == 12_288 * fp16Bytes)
        #expect(scratch.query.length == 6_144 * fp16Bytes)
        #expect(scratch.queryGate.length == 6_144 * fp16Bytes)
        #expect(scratch.key.length == 512 * fp16Bytes)
        #expect(scratch.value.length == 512 * fp16Bytes)
        #expect(scratch.normalizedQuery.length == 6_144 * fp16Bytes)
        #expect(scratch.normalizedKey.length == 512 * fp16Bytes)
        #expect(scratch.attentionOutput.length == 6_144 * fp16Bytes)
        #expect(scratch.gatedAttention.length == 6_144 * fp16Bytes)
        #expect(scratch.qsaTokenMask.length == 64)
    }

    @Test
    func rejectsNonPositiveAttentionContext() throws {
        let context = try MetalContext()

        #expect {
            try Qwen38MTPAttentionScratch(
                device: context.device,
                maxContext: 0)
        } throws: { error in
            guard case ModelError.archMismatch(let field, _, _) = error else {
                return false
            }
            return field == "mtp.attention.maxContext"
        }
    }

    @Test
    func exposesCanonicalSwitchMoEGeometry() {
        let geometry = Qwen38MTPSwitchMoEGeometry.qwen

        #expect(geometry.expertCount == 512)
        #expect(geometry.topK == 10)
        #expect(geometry.hiddenSize == 2_560)
        #expect(geometry.intermediateSize == 640)
        #expect(geometry.groupSize == 32)
    }

    @Test
    func derivesResidentStackedExpertOffsets() throws {
        let context = try MetalContext()
        let geometry = Qwen38MTPSwitchMoEGeometry(
            expertCount: 4,
            topK: 2,
            hiddenSize: 64,
            intermediateSize: 32,
            groupSize: 32)
        let buffer = try #require(context.device.makeBuffer(
            length: 32_768,
            options: .storageModeShared))
        let tensors = Self.switchMoETensors(
            buffer: buffer,
            geometry: geometry)
        let mtp = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: tensors)
        let weights = try Qwen38MTPSwitchMoEWeights(
            mtp: mtp,
            geometry: geometry)

        let views = try weights.expertViews(indices: [3, 1])
        #expect(views.map { $0.expertIndex } == [3, 1])
        #expect(views[0].gate.offset == 3 * 1_024)
        #expect(views[0].gate.scaleOffset == 4_096 + 3 * 128)
        #expect(views[0].gate.biasOffset == 4_608 + 3 * 128)
        #expect(views[1].down.offset == 10_240 + 1_024)
        #expect(views[1].down.scaleOffset == 14_336 + 128)
        #expect(views[1].down.biasOffset == 14_848 + 128)
    }

    @Test
    func acceptsQ8RouterAndSharedExpertMetadata() throws {
        let context = try MetalContext()
        let geometry = Qwen38MTPSwitchMoEGeometry(
            expertCount: 4,
            topK: 2,
            hiddenSize: 64,
            intermediateSize: 32,
            groupSize: 32)
        let buffer = try #require(context.device.makeBuffer(
            length: 32_768,
            options: .storageModeShared))
        let tensors = Self.switchMoETensors(
            buffer: buffer, geometry: geometry, routerQ8: true)
        let mtp = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: tensors)
        let weights = try Qwen38MTPSwitchMoEWeights(
            mtp: mtp, geometry: geometry)

        #expect(weights.router.quantization
                == TensorQuantizationDescriptor(bits: 8, groupSize: 64))
        #expect(weights.sharedExpertGate.quantization
                == TensorQuantizationDescriptor(bits: 8, groupSize: 64))
    }

    @Test
    func rejectsMalformedSwitchMoELeadingDimension() throws {
        let context = try MetalContext()
        let geometry = Qwen38MTPSwitchMoEGeometry(
            expertCount: 4,
            topK: 2,
            hiddenSize: 64,
            intermediateSize: 32,
            groupSize: 32)
        let buffer = try #require(context.device.makeBuffer(
            length: 32_768,
            options: .storageModeShared))
        var tensors = Self.switchMoETensors(
            buffer: buffer,
            geometry: geometry)
        tensors[Qwen38MTPRole.switchExpertGate.rawValue] = TensorView(
            buffer: buffer,
            offset: 0,
            length: 4_096,
            scaleOffset: 4_096,
            scaleLength: 512,
            biasOffset: 4_608,
            biasLength: 512,
            shape: (3, 32, 64, 0),
            dtype: GTurboFormatV1.DType.u32.rawValue,
            quantization: TensorQuantizationDescriptor(bits: 4, groupSize: 32))
        let mtp = try Qwen38MTPWeights(
            predictLayers: 1,
            tensorPrefix: "language_model.mtp.",
            tensors: tensors)

        #expect {
            try Qwen38MTPSwitchMoEWeights(mtp: mtp, geometry: geometry)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else {
                return false
            }
            return detail.contains(Qwen38MTPRole.switchExpertGate.rawValue)
        }
    }

    private static func switchMoETensors(
        buffer: MTLBuffer,
        geometry: Qwen38MTPSwitchMoEGeometry,
        routerQ8: Bool = false
    ) -> [String: TensorView] {
        let elementCount = geometry.expertCount * geometry.intermediateSize
            * geometry.hiddenSize
        let weightBytes = UInt64(elementCount / 2)
        let auxiliaryBytes = UInt64(elementCount / geometry.groupSize * 2)
        let gate = TensorView(
            buffer: buffer,
            offset: 0,
            length: weightBytes,
            scaleOffset: weightBytes,
            scaleLength: auxiliaryBytes,
            biasOffset: weightBytes + auxiliaryBytes,
            biasLength: auxiliaryBytes,
            shape: (UInt32(geometry.expertCount), UInt32(geometry.intermediateSize),
                    UInt32(geometry.hiddenSize), 0),
            dtype: GTurboFormatV1.DType.u32.rawValue,
            quantization: TensorQuantizationDescriptor(bits: 4, groupSize: 32))
        let up = TensorView(
            buffer: buffer,
            offset: 5_120,
            length: weightBytes,
            scaleOffset: 9_216,
            scaleLength: auxiliaryBytes,
            biasOffset: 9_728,
            biasLength: auxiliaryBytes,
            shape: gate.shape,
            dtype: gate.dtype,
            quantization: gate.quantization)
        let down = TensorView(
            buffer: buffer,
            offset: 10_240,
            length: weightBytes,
            scaleOffset: 14_336,
            scaleLength: auxiliaryBytes,
            biasOffset: 14_848,
            biasLength: auxiliaryBytes,
            shape: (UInt32(geometry.expertCount), UInt32(geometry.hiddenSize),
                    UInt32(geometry.intermediateSize), 0),
            dtype: gate.dtype,
            quantization: gate.quantization)
        let denseRouter = TensorView(
            buffer: buffer,
            offset: 0,
            length: UInt64(geometry.expertCount * geometry.hiddenSize * 2),
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (UInt32(geometry.expertCount), UInt32(geometry.hiddenSize), 0, 0),
            dtype: GTurboFormatV1.DType.bf16.rawValue,
            quantization: nil)
        let router = routerQ8
            ? Self.q8Projection(buffer: buffer, offset: 24_576,
                                rows: geometry.expertCount,
                                columns: geometry.hiddenSize)
            : denseRouter
        let sharedGate = Self.q8Projection(
            buffer: buffer, offset: 16_384,
            rows: geometry.intermediateSize, columns: geometry.hiddenSize)
        let sharedUp = Self.q8Projection(
            buffer: buffer, offset: 18_432,
            rows: geometry.intermediateSize, columns: geometry.hiddenSize)
        let sharedDown = Self.q8Projection(
            buffer: buffer, offset: 20_480,
            rows: geometry.hiddenSize, columns: geometry.intermediateSize)
        let sharedMultiplier = Self.q8Projection(
            buffer: buffer, offset: 22_528, rows: 1, columns: geometry.hiddenSize)
        return [
            Qwen38MTPRole.mlpRouter.rawValue: router,
            Qwen38MTPRole.sharedExpertGate.rawValue: sharedGate,
            Qwen38MTPRole.sharedExpertUp.rawValue: sharedUp,
            Qwen38MTPRole.sharedExpertDown.rawValue: sharedDown,
            Qwen38MTPRole.sharedExpertMultiplier.rawValue: sharedMultiplier,
            Qwen38MTPRole.switchExpertGate.rawValue: gate,
            Qwen38MTPRole.switchExpertUp.rawValue: up,
            Qwen38MTPRole.switchExpertDown.rawValue: down,
        ]
    }

    private static func q8Projection(
        buffer: MTLBuffer,
        offset: UInt64,
        rows: Int,
        columns: Int
    ) -> TensorView {
        let elementCount = UInt64(rows * columns)
        let auxiliaryLength = elementCount / 64 * 2
        return TensorView(
            buffer: buffer,
            offset: offset,
            length: elementCount,
            scaleOffset: offset + elementCount,
            scaleLength: auxiliaryLength,
            biasOffset: offset + elementCount + auxiliaryLength,
            biasLength: auxiliaryLength,
            shape: (UInt32(rows), UInt32(columns), 0, 0),
            dtype: GTurboFormatV1.DType.u32.rawValue,
            quantization: TensorQuantizationDescriptor(bits: 8, groupSize: 64))
    }
}
