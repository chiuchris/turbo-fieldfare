import Testing
@testable import TurboFieldfare

@Suite
struct Qwen38MoETests {
    @Test
    func qwen38RoutingGeometryMatchesCanonicalConfiguration() {
        let config = ArchConfig.qwen38FlashNextText

        #expect(Qwen38MoE.topK == 10)
        #expect(Qwen38MoE.numExperts == 512)
        #expect(Qwen38MoE.canonicalHiddenSize == 2560)
        #expect(Qwen38MoE.canonicalIntermediateSize == 640)
        #expect(config.topKExperts == Qwen38MoE.topK)
        #expect(config.numExperts == Qwen38MoE.numExperts)
        #expect(config.moeIntermediateSize == Qwen38MoE.canonicalIntermediateSize)
        #expect(config.intermediateSize == config.moeIntermediateSize)
        #expect(config.qwen38Architecture?.ngramSplitParts == 128)
        #expect(config.qwen38Architecture?.ngramVocabSizeDivisor == 128)
    }

    @Test
    func qwen36RoutingConstantsRemainTopEight() {
        #expect(QwenMoE.topK == 8)
        #expect(QwenMoE.qwen38TopK == 10)
        #expect(QwenMoE.qwen38NumExperts == 512)
    }

    @Test
    func rejectsRoutedExpertCountThatDoesNotMatchTopK() throws {
        let context = try MetalContext()
        let moe = try Qwen38MoE(context: context)
        let buffer = try #require(
            context.device.makeBuffer(length: 1, options: .storageModeShared))
        let view = TensorView(
            buffer: buffer,
            offset: 0,
            length: 1,
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (0, 0, 0, 0),
            dtype: 0)

        #expect {
            _ = try moe.makeRoutedArgumentBuffer(
                experts: Array(repeating: view, count: Qwen38MoE.topK - 1))
        } throws: { error in
            if case ModelError.archMismatch(let field, let expected, let actual) = error {
                return field == "qwen38RoutedExperts.count"
                    && expected == "10"
                    && actual == "9"
            }
            return false
        }
    }

    @Test
    func preservesTopTenSlotsWithNonZeroExpertOffsets() throws {
        let context = try MetalContext()
        let moe = try Qwen38MoE(context: context)
        let buffer = try #require(
            context.device.makeBuffer(length: 8192, options: .storageModeShared))
        var experts: [TensorView] = []
        for index in 0..<Qwen38MoE.topK {
            let offset: UInt64 = UInt64(64 + index * 128)
            let scaleOffset: UInt64 = UInt64(2048 + index * 64)
            let biasOffset: UInt64 = UInt64(3072 + index * 64)
            experts.append(TensorView(
                buffer: buffer,
                offset: offset,
                length: 64,
                scaleOffset: scaleOffset,
                scaleLength: 64,
                biasOffset: biasOffset,
                biasLength: 64,
                shape: (1, 2560, 0, 0),
                dtype: 0))
        }

        let first = try moe.makeRoutedArgumentBuffer(
            layer: 2, slot: 3, experts: experts)
        let repeated = try moe.makeRoutedArgumentBuffer(
            layer: 2, slot: 3, experts: experts)
        let differentSlot = try moe.makeRoutedArgumentBuffer(
            layer: 2, slot: 4, experts: experts)
        let differentLayer = try moe.makeRoutedArgumentBuffer(
            layer: 3, slot: 3, experts: experts)

        #expect(first === repeated)
        #expect(first !== differentSlot)
        #expect(first !== differentLayer)
    }
}
