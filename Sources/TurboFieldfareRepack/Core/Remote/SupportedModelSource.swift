import Foundation

public struct SupportedModelSourceProfile: Sendable, Equatable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceFileSHA256: [String: String]
    public let architecture: String?
    public let numLayers: Int?
    public let expertsPerLayer: Int?
    public let topKExperts: Int?
    public let hiddenSize: Int?
    public let vocabularySize: Int?
    public let expectedTensorCount: Int?
    public let expectedRoutedExpertTensorCount: Int?
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public var sourceIndexSHA256: String {
        sourceFileSHA256["model.safetensors.index.json"] ?? ""
    }

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceFileSHA256: [String: String],
                architecture: String? = nil,
                numLayers: Int? = nil,
                expertsPerLayer: Int? = nil,
                topKExperts: Int? = nil,
                hiddenSize: Int? = nil,
                vocabularySize: Int? = nil,
                expectedTensorCount: Int? = nil,
                expectedRoutedExpertTensorCount: Int? = nil,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceFileSHA256 = sourceFileSHA256
        self.architecture = architecture
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.topKExperts = topKExperts
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
        self.expectedTensorCount = expectedTensorCount
        self.expectedRoutedExpertTensorCount = expectedRoutedExpertTensorCount
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}

public enum SupportedModelSource {
    public static let gemma4 = SupportedModelSourceProfile(
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceFileSHA256: [
            "model.safetensors.index.json":
                "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        ],
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    public static let qwen36 = SupportedModelSourceProfile(
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceFileSHA256: [
            "model.safetensors.index.json":
                "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
            "config.json":
                "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767",
            "tokenizer.json":
                "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
            "tokenizer_config.json":
                "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
        ],
        architecture: "qwen3_5_moe_text",
        numLayers: 40,
        expertsPerLayer: 256,
        topKExperts: 8,
        hiddenSize: 2_048,
        vocabularySize: 248_320,
        expectedTensorCount: 2_090,
        expectedRoutedExpertTensorCount: 360,
        approximateDownloadBytes: 20_401_929_952,
        installedBytes: 19_508_787_456,
        reserveBytes: 1_073_741_824)

    public static let defaultProfile = gemma4
    public static let knownProfiles = [gemma4, qwen36]

    public static func profile(forRepoID repoID: String)
        -> SupportedModelSourceProfile? {
        knownProfiles.first { $0.repoID == repoID }
    }

    public static func profile(forName name: String)
        -> SupportedModelSourceProfile? {
        switch name {
        case "gemma4": return gemma4
        case "qwen36": return qwen36
        default: return nil
        }
    }

    public static let displayName = defaultProfile.displayName
    public static let repoID = defaultProfile.repoID
    public static let revision = defaultProfile.revision
    public static let sourceIndexSHA256 = defaultProfile.sourceIndexSHA256
    public static let approximateDownloadBytes =
        defaultProfile.approximateDownloadBytes
    public static let installedBytes = defaultProfile.installedBytes
    public static let reserveBytes = defaultProfile.reserveBytes

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        defaultProfile.installOptions(outputDirectory: outputDirectory,
                                      overwrite: overwrite,
                                      token: token,
                                      resume: resume)
    }
}
