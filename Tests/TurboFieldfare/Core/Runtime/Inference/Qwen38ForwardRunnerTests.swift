import Metal
import Testing
@testable import TurboFieldfare

@Suite
struct Qwen38ForwardRunnerTests {
    @Test
    func rejectsNonQwen38ModelBeforeRuntimeAllocation() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: .gemma4Toy())

        #expect(throws: ModelError.self) {
            try Qwen38ForwardRunner(
                model: model,
                context: context,
                maxContext: 4)
        }
    }

    @Test
    func qwen38RunnerUsesCanonicalGeometry() {
        let config = ArchConfig.qwen38FlashNextText

        #expect(config.hiddenSize == 2_560)
        #expect(config.numLayers == 48)
        #expect(config.numExperts == 512)
        #expect(config.topKExperts == 10)
        #expect(config.qwen38Architecture?.pleLayerIDs == [2])
        #expect(config.qwen38Architecture?.indexerBudget == 2_048)
    }
}
