import Foundation

package enum GTurboFormatV3 {
    package static let versionMajor = 3
    package static let knownFlags: Set<String> = [
        "streamingPresent", "untiedHead", "ngramStreamingPresent",
    ]
    package static let knownQuantRoles: Set<String> = [
        "embedding", "attention", "attentionIndexer", "deltaNet",
        "hyperConnection", "ple", "ngramEmbedding", "router",
        "sharedExpert", "sharedExpertGate", "routedExpert", "lmHead", "norm",
    ]
}

package struct GTurboManifestV3SparseAttention: Codable, Equatable, Sendable {
    package let queryHeads: Int
    package let keyValueHeads: Int
    package let headDim: Int
    package let ropeTheta: Double
    package let partialRotaryFactor: Double
    package let indexerHeads: Int
    package let indexerKeyValueHeads: Int
    package let indexerHeadDim: Int
    package let indexerCompressRatio: Int
    package let indexerBudget: Int

    package init(queryHeads: Int, keyValueHeads: Int, headDim: Int,
                 ropeTheta: Double, partialRotaryFactor: Double,
                 indexerHeads: Int, indexerKeyValueHeads: Int,
                 indexerHeadDim: Int, indexerCompressRatio: Int,
                 indexerBudget: Int) {
        self.queryHeads = queryHeads
        self.keyValueHeads = keyValueHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.indexerHeads = indexerHeads
        self.indexerKeyValueHeads = indexerKeyValueHeads
        self.indexerHeadDim = indexerHeadDim
        self.indexerCompressRatio = indexerCompressRatio
        self.indexerBudget = indexerBudget
    }
}

package struct GTurboManifestV3HyperConnection: Codable, Equatable, Sendable {
    package let streamCount: Int
    package let lowRankSize: Int

    package init(streamCount: Int, lowRankSize: Int) {
        self.streamCount = streamCount
        self.lowRankSize = lowRankSize
    }
}

package struct GTurboManifestV3PLE: Codable, Equatable, Sendable {
    package let layerIDs: [Int]
    package let embeddingSize: Int
    package let convolutionKernel: Int
    package let ngramSize: Int
    package let headsPerNgram: Int
    package let vocabSizeBase: Int
    package let splitParts: Int
    package let vocabSizeDivisor: Int
    package let layoutFile: String

    package init(layerIDs: [Int], embeddingSize: Int, convolutionKernel: Int,
                 ngramSize: Int, headsPerNgram: Int, vocabSizeBase: Int,
                 splitParts: Int, vocabSizeDivisor: Int, layoutFile: String) {
        self.layerIDs = layerIDs
        self.embeddingSize = embeddingSize
        self.convolutionKernel = convolutionKernel
        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.vocabSizeBase = vocabSizeBase
        self.splitParts = splitParts
        self.vocabSizeDivisor = vocabSizeDivisor
        self.layoutFile = layoutFile
    }
}

package struct GTurboManifestV3MTPContract: Codable, Equatable, Sendable {
    package let baseHiddenVariant: String
    package let concatOrder: String
    package let hiddenVariant: String
    package let mtpPositionMode: String
    package let mtpQuantGroupSize: Int
    package let mtpQuantMode: String

    package init(baseHiddenVariant: String, concatOrder: String,
                 hiddenVariant: String, mtpPositionMode: String,
                 mtpQuantGroupSize: Int, mtpQuantMode: String) {
        self.baseHiddenVariant = baseHiddenVariant
        self.concatOrder = concatOrder
        self.hiddenVariant = hiddenVariant
        self.mtpPositionMode = mtpPositionMode
        self.mtpQuantGroupSize = mtpQuantGroupSize
        self.mtpQuantMode = mtpQuantMode
    }
}

package struct GTurboManifestV3MTPQuantization: Codable, Equatable, Sendable {
    package let bits: Int
    package let groupSize: Int

    package init(bits: Int, groupSize: Int) {
        self.bits = bits
        self.groupSize = groupSize
    }
}

package struct GTurboManifestV3MTP: Codable, Equatable, Sendable {
    package let predictLayers: Int
    package let tensorPrefix: String
    package let usesDedicatedEmbeddings: Bool
    package let depthMax: Int?
    package let contract: GTurboManifestV3MTPContract?
    package let tensorQuantization: [String: GTurboManifestV3MTPQuantization]

    private enum CodingKeys: String, CodingKey {
        case predictLayers
        case tensorPrefix
        case usesDedicatedEmbeddings
        case depthMax
        case contract
        case tensorQuantization
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        predictLayers = try container.decode(Int.self, forKey: .predictLayers)
        tensorPrefix = try container.decode(String.self, forKey: .tensorPrefix)
        usesDedicatedEmbeddings = try container.decode(
            Bool.self, forKey: .usesDedicatedEmbeddings)
        depthMax = try container.decodeIfPresent(Int.self, forKey: .depthMax)
        contract = try container.decodeIfPresent(
            GTurboManifestV3MTPContract.self, forKey: .contract)
        tensorQuantization = try container.decodeIfPresent(
            [String: GTurboManifestV3MTPQuantization].self,
            forKey: .tensorQuantization) ?? [:]
    }

    package init(predictLayers: Int, tensorPrefix: String,
                 usesDedicatedEmbeddings: Bool, depthMax: Int? = nil,
                 contract: GTurboManifestV3MTPContract? = nil,
                 tensorQuantization: [String: GTurboManifestV3MTPQuantization] = [:]) {
        self.predictLayers = predictLayers
        self.tensorPrefix = tensorPrefix
        self.usesDedicatedEmbeddings = usesDedicatedEmbeddings
        self.depthMax = depthMax
        self.contract = contract
        self.tensorQuantization = tensorQuantization
    }
}

package struct GTurboManifestV3Arch: Codable, Equatable, Sendable {
    package let modelFamily: String
    package let hiddenSize: Int
    package let vocabSize: Int
    package let numLayers: Int
    package let layerKinds: [String]
    package let numRoutedExperts: Int
    package let topKExperts: Int
    package let routedExpertIntermediateSize: Int
    package let sharedExpertIntermediateSize: Int
    package let routerActivation: String
    package let routedExpertActivation: String
    package let sharedExpertActivation: String
    package let sharedExpertGateActivation: String
    package let tieWordEmbeddings: Bool
    package let sparseAttention: GTurboManifestV3SparseAttention
    package let gatedDeltaNet: GTurboManifestV2GatedDeltaNet
    package let hyperConnection: GTurboManifestV3HyperConnection
    package let ple: GTurboManifestV3PLE

    package init(modelFamily: String, hiddenSize: Int, vocabSize: Int,
                 numLayers: Int, layerKinds: [String], numRoutedExperts: Int,
                 topKExperts: Int, routedExpertIntermediateSize: Int,
                 sharedExpertIntermediateSize: Int, routerActivation: String,
                 routedExpertActivation: String, sharedExpertActivation: String,
                 sharedExpertGateActivation: String, tieWordEmbeddings: Bool,
                 sparseAttention: GTurboManifestV3SparseAttention,
                 gatedDeltaNet: GTurboManifestV2GatedDeltaNet,
                 hyperConnection: GTurboManifestV3HyperConnection,
                 ple: GTurboManifestV3PLE) {
        self.modelFamily = modelFamily
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.layerKinds = layerKinds
        self.numRoutedExperts = numRoutedExperts
        self.topKExperts = topKExperts
        self.routedExpertIntermediateSize = routedExpertIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.routerActivation = routerActivation
        self.routedExpertActivation = routedExpertActivation
        self.sharedExpertActivation = sharedExpertActivation
        self.sharedExpertGateActivation = sharedExpertGateActivation
        self.tieWordEmbeddings = tieWordEmbeddings
        self.sparseAttention = sparseAttention
        self.gatedDeltaNet = gatedDeltaNet
        self.hyperConnection = hyperConnection
        self.ple = ple
    }
}

package struct GTurboManifestV3: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestV3Arch
    package let quant: GTurboManifestQuantV2
    package let mtp: GTurboManifestV3MTP?
    package let files: [String: GTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64

    package init(magic: String = GTurboFormatV1.magic,
                 versionMajor: Int = GTurboFormatV3.versionMajor,
                 versionMinor: Int = 0, flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestV3Arch,
                 quant: GTurboManifestQuantV2,
                 mtp: GTurboManifestV3MTP? = nil,
                 files: [String: GTurboManifestFileV1], expertsPerLayer: Int,
                 numLayers: Int, expertStride: UInt64) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.mtp = mtp
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
    }
}

package enum GTurboManifestV3Codec {
    package static func decode(_ data: Data) throws -> GTurboManifestV3 {
        let manifest: GTurboManifestV3
        do {
            manifest = try JSONDecoder().decode(GTurboManifestV3.self, from: data)
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
        try validate(manifest, data: data)
        return manifest
    }

    package static func encode(_ manifest: GTurboManifestV3) throws -> Data {
        try validate(manifest, data: nil)
        do {
            let encoded = try JSONEncoder().encode(manifest)
            let object = try JSONSerialization.jsonObject(with: encoded)
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func validate(_ manifest: GTurboManifestV3,
                                 data: Data?) throws {
        guard manifest.magic == GTurboFormatV1.magic else {
            throw invalid("manifest.magic", "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV3.versionMajor,
              manifest.versionMinor >= 0 else {
            throw invalid("manifest.version", "unsupported version")
        }
        guard manifest.arch.modelFamily == "qwen4_exp_text" else {
            throw invalid("manifest.arch.modelFamily", "unsupported model family")
        }
        for flag in manifest.flags.keys where !GTurboFormatV3.knownFlags.contains(flag) {
            throw invalid("manifest.flags.\(flag)", "unknown v3 flag")
        }
        try validateKeys(data)
        let arch = manifest.arch
        guard !manifest.modelID.isEmpty,
              manifest.numLayers == arch.numLayers,
              manifest.expertsPerLayer == arch.numRoutedExperts,
              manifest.expertStride > 0,
              manifest.expertStride % GTurboFormatV1.alignmentBytes == 0,
              arch.hiddenSize > 0, arch.vocabSize > 0, arch.numLayers > 0,
              arch.layerKinds.count == arch.numLayers,
              arch.layerKinds.allSatisfy({ $0 == "gatedDeltaNet" || $0 == "sparseAttention" }),
              arch.layerKinds.contains("gatedDeltaNet"),
              arch.layerKinds.contains("sparseAttention"),
              arch.numRoutedExperts > 0,
              arch.topKExperts > 0, arch.topKExperts <= arch.numRoutedExperts,
              arch.routedExpertIntermediateSize > 0,
              arch.sharedExpertIntermediateSize > 0,
              !arch.routerActivation.isEmpty,
              !arch.routedExpertActivation.isEmpty,
              !arch.sharedExpertActivation.isEmpty,
              !arch.sharedExpertGateActivation.isEmpty,
              !arch.tieWordEmbeddings else {
            throw invalid("manifest.arch", "invalid architecture values")
        }
        try validateSparseAttention(arch.sparseAttention)
        try validateDeltaNet(arch.gatedDeltaNet)
        guard arch.hyperConnection.streamCount > 1,
              arch.hyperConnection.lowRankSize > 0 else {
            throw invalid("manifest.arch.hyperConnection", "invalid hyper-connection values")
        }
        try validatePLE(arch.ple, numLayers: arch.numLayers)
        try validateQuant(manifest.quant)
        if let mtp = manifest.mtp {
            guard mtp.predictLayers > 0,
                  mtp.predictLayers <= arch.numLayers,
                  mtp.tensorPrefix == "language_model.mtp.",
                  !mtp.tensorPrefix.isEmpty,
                  mtp.depthMax == nil || mtp.depthMax! >= mtp.predictLayers else {
                throw invalid("manifest.mtp", "invalid MTP metadata")
            }
            try validateMTPContract(mtp.contract)
            for (relativeName, descriptor) in mtp.tensorQuantization {
                guard !relativeName.isEmpty,
                      !relativeName.contains("/"),
                      !relativeName.hasPrefix(mtp.tensorPrefix),
                      descriptor.bits == 4 || descriptor.bits == 8,
                      descriptor.groupSize > 0 else {
                    throw invalid(
                        "manifest.mtp.tensorQuantization.\(relativeName)",
                        "invalid tensor quantization descriptor")
                }
            }
        }
        try validateFiles(manifest.files, requiredLayout: arch.ple.layoutFile)
    }

    private static func validateMTPContract(
        _ contract: GTurboManifestV3MTPContract?
    ) throws {
        guard let contract else { return }
        guard !contract.baseHiddenVariant.isEmpty,
              !contract.concatOrder.isEmpty,
              !contract.hiddenVariant.isEmpty,
              !contract.mtpPositionMode.isEmpty,
              contract.mtpQuantGroupSize > 0,
              !contract.mtpQuantMode.isEmpty else {
            throw invalid("manifest.mtp.contract", "invalid MTP contract")
        }
    }

    private static func validateSparseAttention(
        _ attention: GTurboManifestV3SparseAttention
    ) throws {
        guard attention.queryHeads > 0,
              attention.keyValueHeads > 0,
              attention.keyValueHeads <= attention.queryHeads,
              attention.headDim > 0,
              attention.ropeTheta.isFinite, attention.ropeTheta > 0,
              attention.partialRotaryFactor.isFinite,
              attention.partialRotaryFactor > 0,
              attention.partialRotaryFactor <= 1,
              attention.indexerHeads > 0,
              attention.indexerKeyValueHeads > 0,
              attention.indexerKeyValueHeads <= attention.indexerHeads,
              attention.indexerHeadDim > 0,
              attention.indexerCompressRatio > 0,
              attention.indexerBudget > 0 else {
            throw invalid("manifest.arch.sparseAttention", "invalid sparse-attention values")
        }
    }

    private static func validateDeltaNet(
        _ deltaNet: GTurboManifestV2GatedDeltaNet
    ) throws {
        guard deltaNet.keyHeads > 0, deltaNet.valueHeads > 0,
              deltaNet.keyHeadDim > 0, deltaNet.valueHeadDim > 0,
              deltaNet.convolutionKernel > 0,
              deltaNet.stateDType == "FP32" else {
            throw invalid("manifest.arch.gatedDeltaNet", "invalid DeltaNet values")
        }
    }

    private static func validatePLE(_ ple: GTurboManifestV3PLE,
                                    numLayers: Int) throws {
        guard !ple.layerIDs.isEmpty,
              Set(ple.layerIDs).count == ple.layerIDs.count,
              ple.layerIDs.allSatisfy({ $0 > 0 && $0 <= numLayers }),
              ple.embeddingSize > 0, ple.convolutionKernel > 0,
              ple.ngramSize > 1, ple.headsPerNgram > 0,
              ple.vocabSizeBase > 0, ple.splitParts > 0,
              ple.vocabSizeDivisor > 0 else {
            throw invalid("manifest.arch.ple", "invalid PLE values")
        }
        try GTurboPathValidator.validateRelativePath(
            ple.layoutFile, field: "manifest.arch.ple.layoutFile")
    }

    private static func validateQuant(_ quant: GTurboManifestQuantV2) throws {
        guard Set(quant.roles.keys) == GTurboFormatV3.knownQuantRoles else {
            throw invalid("manifest.quant.roles", "expected all v3 quantization roles")
        }
        for (role, slot) in quant.roles {
            guard slot.weightBits > 0, slot.weightBits <= 32,
                  !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                  !slot.biasType.isEmpty, slot.groupSize > 0 else {
                throw invalid("manifest.quant.roles.\(role)", "invalid quantization role")
            }
        }
    }

    private static func validateFiles(
        _ files: [String: GTurboManifestFileV1], requiredLayout: String
    ) throws {
        guard files[requiredLayout] != nil else {
            throw invalid("manifest.files", "missing PLE layout file")
        }
        var canonicalPaths: Set<String> = []
        for (path, entry) in files {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.insert(key).inserted else {
                throw invalid("manifest.files.\(path)", "filesystem-equivalent duplicate path")
            }
            guard key != "manifest.json", key != "verified-install.json",
                  key != "tokenizer", !key.hasPrefix("manifest.json/"),
                  !key.hasPrefix("verified-install.json/") else {
                throw invalid("manifest.files.\(path)", "reserved artifact filename")
            }
            guard entry.sha256.count == 64,
                  entry.sha256.allSatisfy({ $0.isHexDigit }) else {
                throw invalid("manifest.files.\(path).sha256", "expected 64 hexadecimal characters")
            }
        }
    }

    private static func validateKeys(_ data: Data?) throws {
        guard let data else { return }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw invalid("manifest.json", "expected object")
        }
        try rejectUnknownKeys(root, allowed: [
            "magic", "versionMajor", "versionMinor", "flags", "modelID",
            "sourceSnapshotHash", "arch", "quant", "mtp", "files", "expertsPerLayer",
            "numLayers", "expertStride",
        ], field: "manifest")
        try rejectNestedKeys(root["mtp"], allowed: [
            "predictLayers", "tensorPrefix", "usesDedicatedEmbeddings",
        ], field: "manifest.mtp")
        if let arch = root["arch"] as? [String: Any] {
            try rejectUnknownKeys(arch, allowed: [
                "modelFamily", "hiddenSize", "vocabSize", "numLayers", "layerKinds",
                "numRoutedExperts", "topKExperts", "routedExpertIntermediateSize",
                "sharedExpertIntermediateSize", "routerActivation", "routedExpertActivation",
                "sharedExpertActivation", "sharedExpertGateActivation", "tieWordEmbeddings",
                "sparseAttention", "gatedDeltaNet", "hyperConnection", "ple",
            ], field: "manifest.arch")
            try rejectNestedKeys(arch["sparseAttention"], allowed: [
                "queryHeads", "keyValueHeads", "headDim", "ropeTheta",
                "partialRotaryFactor", "indexerHeads", "indexerKeyValueHeads",
                "indexerHeadDim", "indexerCompressRatio", "indexerBudget",
            ], field: "manifest.arch.sparseAttention")
            try rejectNestedKeys(arch["gatedDeltaNet"], allowed: [
                "keyHeads", "valueHeads", "keyHeadDim", "valueHeadDim",
                "convolutionKernel", "stateDType",
            ], field: "manifest.arch.gatedDeltaNet")
            try rejectNestedKeys(arch["hyperConnection"], allowed: [
                "streamCount", "lowRankSize",
            ], field: "manifest.arch.hyperConnection")
            try rejectNestedKeys(arch["ple"], allowed: [
                "layerIDs", "embeddingSize", "convolutionKernel", "ngramSize",
                "headsPerNgram", "vocabSizeBase", "splitParts", "vocabSizeDivisor",
                "layoutFile",
            ], field: "manifest.arch.ple")
        }
        if let quant = root["quant"] as? [String: Any] {
            try rejectUnknownKeys(quant, allowed: ["roles"], field: "manifest.quant")
            if let roles = quant["roles"] as? [String: Any] {
                for (role, value) in roles {
                    try rejectNestedKeys(value, allowed: [
                        "weightBits", "scheme", "scaleType", "biasType", "groupSize",
                    ], field: "manifest.quant.roles.\(role)")
                }
            }
        }
        if let files = root["files"] as? [String: Any] {
            for (path, value) in files {
                try rejectNestedKeys(value, allowed: ["size", "sha256"],
                                     field: "manifest.files.\(path)")
            }
        }
    }

    private static func rejectNestedKeys(_ value: Any?, allowed: Set<String>,
                                         field: String) throws {
        guard let object = value as? [String: Any] else { return }
        try rejectUnknownKeys(object, allowed: allowed, field: field)
    }

    private static func rejectUnknownKeys(_ object: [String: Any],
                                          allowed: Set<String>, field: String) throws {
        if let unknown = object.keys.first(where: { !allowed.contains($0) }) {
            throw invalid("\(field).\(unknown)", "unknown v3 field")
        }
    }

    private static func invalid(_ field: String, _ reason: String) -> GTurboFormatError {
        GTurboFormatError.invalid(field: field, reason: reason)
    }
}
