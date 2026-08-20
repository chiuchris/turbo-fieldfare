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
        #expect(SupportedModelSource.installOptions(
            outputDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            overwrite: false,
            token: nil).repoID == SupportedModelSource.gemma4.repoID)
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
    func fingerprintsRecognizeBothPinnedSources() {
        #expect(SourceFingerprint.knownFingerprints.count == 2)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.gemma4.sourceIndexSHA256) ==
                SupportedModelSource.gemma4.repoID)
        #expect(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.qwen36.sourceIndexSHA256) ==
                SupportedModelSource.qwen36.repoID)
    }
}