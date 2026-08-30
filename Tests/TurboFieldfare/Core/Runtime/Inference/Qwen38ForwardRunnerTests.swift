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
    func factoryDispatchesQwen38IntoRunnerValidation() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let context = try MetalContext()
        let base = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: .gemma4Toy())
        let model = Model(
            device: base.device,
            config: .qwen38FlashNextText,
            streamingMode: base.streamingMode,
            expertCachePolicy: base.expertCachePolicy,
            integrityPolicy: base.integrityPolicy,
            residentBuffer: base.residentBuffer,
            residentIndex: base.residentIndex,
            packedExpertsLayout: base.packedExpertsLayout,
            manifest: base.manifest,
            directoryURL: base.directoryURL,
            modelDirectory: base.modelDirectory,
            trustedInstallReceipt: base.trustedInstallReceipt)
        let expected = Qwen38TensorNames.ple(
            layer: 1, tensor: .keyProjection)

        #expect {
            _ = try ForwardRunnerFactory.make(
                model: model,
                context: context,
                maxContext: 4)
        } throws: { error in
            if case ModelError.tensorNotFound(let name) = error {
                return name == expected
            }
            return false
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

    @Test
    func decodeTimingSampleHasStableZeroAndCodableFields() throws {
        #expect(Qwen38DecodeTimingSample.zero == Qwen38DecodeTimingSample(
            embeddingNanos: 0,
            pleNanos: 0,
            attentionRouterNanos: 0,
            expertFetchNanos: 0,
            moeNanos: 0,
            finalHeadNanos: 0,
            gpuActiveNanos: 0,
            commandBufferCount: 0))

        let sample = Qwen38DecodeTimingSample(
            embeddingNanos: 1,
            pleNanos: 2,
            attentionRouterNanos: 3,
            expertFetchNanos: 4,
            expertCacheHits: 8,
            expertCacheMisses: 9,
            moeNanos: 5,
            finalHeadNanos: 6,
            gpuActiveNanos: 7,
            commandBufferCount: 7)
        let encoded = try JSONEncoder().encode(sample)
        let decoded = try JSONDecoder().decode(
            Qwen38DecodeTimingSample.self, from: encoded)

        #expect(decoded == sample)
        #expect(decoded.commandBufferCount == 7)
        #expect(decoded.expertFetchNanos == 4)
        #expect(decoded.expertCacheHits == 8)
        #expect(decoded.expertCacheMisses == 9)
        #expect(decoded.gpuActiveNanos == 7)
    }
}
