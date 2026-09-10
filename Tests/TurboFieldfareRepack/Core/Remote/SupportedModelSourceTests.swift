import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct SupportedModelSourceTests {
    @Test
    func defaultProfileRemainsGemma() {
        #expect(SupportedModelSource.defaultProfile == SupportedModelSource.gemma4)
        #expect(SupportedModelSource.repoID ==
                "mlx-community/gemma-4-26b-a4b-it-4bit")
        let options = SupportedModelSource.installOptions(
            outputDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            overwrite: false,
            token: nil)
        #expect(options.repoID == SupportedModelSource.gemma4.repoID)
        #expect(options.rangeChunkBytes == RemoteChunkPolicy.defaultBytes)
    }

    @Test
    func rangeChunkSizeCanBeSelectedPerProfile() {
        let options = SupportedModelSource.qwen38Mtplx.installOptions(
            outputDirectory: URL(fileURLWithPath: "/tmp/qwen38-mtplx.gturbo"),
            overwrite: false,
            token: nil,
            rangeChunkBytes: 128 * 1024 * 1024)

        #expect(options.rangeChunkBytes == 128 * 1024 * 1024)
    }

    @Test
    func residentConcurrencyCanBeSelectedPerProfile() {
        let options = SupportedModelSource.qwen38Mtplx.installOptions(
            outputDirectory: URL(fileURLWithPath: "/tmp/qwen38-mtplx.gturbo"),
            overwrite: false,
            token: nil,
            remoteConcurrency: 2,
            residentConcurrency: 3)

        #expect(options.remoteConcurrency == 2)
        #expect(options.residentConcurrency == 3)
    }

    @Test
    func qwenProfilePinsSourceAndArchitecture() {
        let profile = SupportedModelSource.qwen36

        #expect(profile.repoID == "mlx-community/Qwen3.6-35B-A3B-4bit")
        #expect(profile.revision ==
                "38740b847e4cb78f352aba30aa41c76e08e6eb46")
        #expect(profile.sourceIndexSHA256 ==
                "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea")
        #expect(profile.architecture == "qwen3_5_moe_text")
        #expect(profile.numLayers == 40)
        #expect(profile.expertsPerLayer == 256)
        #expect(profile.topKExperts == 8)
        #expect(profile.hiddenSize == 2_048)
        #expect(profile.vocabularySize == 248_320)
        #expect(profile.expectedTensorCount == 2_090)
        #expect(profile.expectedRoutedExpertTensorCount == 360)
        #expect(profile.sourceFileSHA256.count == 4)
        #expect(SupportedModelSource.profile(forRepoID: profile.repoID) == profile)
        #expect(SupportedModelSource.profile(forName: "qwen36") == profile)
        #expect(SupportedModelSource.profile(forName: "unknown") == nil)
    }

    @Test
    func qwenMtplxProfilePinsSourceAndArchitecture() {
        let profile = SupportedModelSource.qwen38Mtplx

        #expect(profile.repoID ==
                "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed")
        #expect(profile.revision ==
                "6bc2f6e8426ccb4af73c81bc56ba7718afc92cc6")
        #expect(profile.sourceIndexSHA256 ==
                "62082a9fe763544d805851edff6a024fea8d26e0b476064b4402e388abc1a39c")
        #expect(profile.architecture == "qwen4_exp_text")
        #expect(profile.numLayers == 48)
        #expect(profile.expertsPerLayer == 512)
        #expect(profile.topKExperts == 10)
        #expect(profile.hiddenSize == 2_560)
        #expect(profile.vocabularySize == 248_320)
        #expect(profile.expectedTensorCount == 2_799)
        #expect(profile.expectedRoutedExpertTensorCount == 432)
        #expect(profile.sourceFileSHA256.count == 4)
        #expect(SupportedModelSource.profile(forRepoID: profile.repoID) == profile)
        #expect(SupportedModelSource.profile(forName: "qwen38-mtplx") == profile)
    }

    @Test
    func fingerprintsRecognizeAllPinnedSources() {
        #expect(SourceFingerprint.knownFingerprints.count == 4)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.gemma4.sourceIndexSHA256) ==
                SupportedModelSource.gemma4.repoID)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.qwen36.sourceIndexSHA256) ==
                SupportedModelSource.qwen36.repoID)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.qwen38.sourceIndexSHA256) ==
                SupportedModelSource.qwen38.repoID)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.qwen38Mtplx.sourceIndexSHA256) ==
                SupportedModelSource.qwen38Mtplx.repoID)
    }
}