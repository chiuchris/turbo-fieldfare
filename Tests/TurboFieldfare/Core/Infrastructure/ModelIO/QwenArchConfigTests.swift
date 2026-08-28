import Testing
@testable import TurboFieldfare

@Suite struct QwenArchConfigTests {

    @Test func canonicalQwen36MoeContractMatchesFrozenTarget() {
        let config = ArchConfig.qwen36MoeText

        #expect(config.modelFamily == .qwen36MoeText)
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
        #expect(config.qwen38Architecture == nil)
    }

    @Test func canonicalQwen38FlashNextContractMatchesFrozenTarget() throws {
        let config = ArchConfig.qwen38FlashNextText
        let extensionConfig = try #require(config.qwen38Architecture)

        #expect(config.modelFamily == .qwen38FlashNextText)
        #expect(config.hiddenSize == 2560)
        #expect(config.vocabSize == 248_320)
        #expect(config.numLayers == 48)
        #expect(config.numExperts == 512)
        #expect(config.topKExperts == 10)
        #expect(config.numHeads == 24)
        #expect(config.numKVHeads == 2)
        #expect(config.headDim == 256)
        #expect(config.fullAttentionLayerMask.count == 48)
        #expect(config.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
        #expect(config.linearNumKeyHeads == 16)
        #expect(config.linearNumValueHeads == 48)
        #expect(extensionConfig.indexerHeads == 4)
        #expect(extensionConfig.indexerKeyValueHeads == 1)
        #expect(extensionConfig.indexerHeadDim == 128)
        #expect(extensionConfig.indexerCompressRatio == 4)
        #expect(extensionConfig.indexerBudget == 2048)
        #expect(extensionConfig.hyperConnectionCount == 4)
        #expect(extensionConfig.hyperConnectionLowRank == 320)
        #expect(extensionConfig.pleLayerIDs == [2])
        #expect(extensionConfig.ngramSize == 3)
        #expect(extensionConfig.headsPerNgram == 8)
        #expect(extensionConfig.ngramVocabSizeBase == 20_000_000)
        #expect(extensionConfig.ngramSplitParts == 128)
    }

    @Test func GemmaDefaultsRemainUnchanged() {
        let config = ArchConfig.gemma4_26B_A4B

        #expect(config.modelFamily == .gemma4)
        #expect(config.linearNumKeyHeads == 0)
        #expect(config.linearNumValueHeads == 0)
        #expect(config.linearKeyHeadDim == 0)
        #expect(config.linearValueHeadDim == 0)
        #expect(config.linearConvKernelDim == 0)
        #expect(config.qwen38Architecture == nil)
    }
}