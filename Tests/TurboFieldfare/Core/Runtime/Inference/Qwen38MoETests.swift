import Testing
@testable import TurboFieldfare

@Suite
struct Qwen38MoETests {
    @Test
    func qwen38RoutingGeometryMatchesCanonicalConfiguration() {
        let config = ArchConfig.qwen38FlashNextText

        #expect(Qwen38MoE.topK == 10)
        #expect(Qwen38MoE.numExperts == 512)
        #expect(config.topKExperts == Qwen38MoE.topK)
        #expect(config.numExperts == Qwen38MoE.numExperts)
        #expect(config.moeIntermediateSize == 640)
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
}
