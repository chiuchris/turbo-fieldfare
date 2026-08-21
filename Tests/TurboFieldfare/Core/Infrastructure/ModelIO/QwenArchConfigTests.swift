import Testing
@testable import TurboFieldfare

@Suite struct QwenArchConfigTests {

    @Test func canonicalQwen35ContractMatchesFrozenTarget() {
        let config = ArchConfig.qwen35MoeText

        #expect(config.modelFamily == .qwen35MoeText)
        #expect(config.hiddenSize == 2048)
        #expect(config.vocabSize == 248_320)
        #expect(config.numLayers == 40)
        #expect(config.numExperts == 256)
        #expect(config.topKExperts == 8)
        #expect(config.numHeads == 16)
        #expect(config.numKVHeads == 2)
        #expect(config.headDim == 256)
        #expect(config.fullAttentionLayerMask.count == 40)
        #expect(config.fullAttentionLayerMask.filter { $0 == 1 }.count == 10)
        #expect(config.linearNumKeyHeads == 16)
        #expect(config.linearNumValueHeads == 32)
        #expect(config.linearKeyHeadDim == 128)
        #expect(config.linearValueHeadDim == 128)
        #expect(config.linearConvKernelDim == 4)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test func GemmaDefaultsRemainUnchanged() {
        let config = ArchConfig.gemma4_26B_A4B

        #expect(config.modelFamily == .gemma4)
        #expect(config.linearNumKeyHeads == 0)
        #expect(config.linearNumValueHeads == 0)
        #expect(config.linearKeyHeadDim == 0)
        #expect(config.linearValueHeadDim == 0)
        #expect(config.linearConvKernelDim == 0)
    }
}