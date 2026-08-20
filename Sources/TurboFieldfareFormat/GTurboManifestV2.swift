import Foundation

package enum GTurboFormatV2 {
    package static let versionMajor = 2
    package static let knownFlags: Set<String> = [
        "streamingPresent", "untiedHead",
    ]
    package static let knownQuantRoles: Set<String> = [
        "embedding", "attention", "deltaNet", "router", "sharedExpert",
        "sharedExpertGate", "routedExpert", "lmHead", "norm",
    ]
}

package struct GTurboManifestV2FullAttention: Codable, Equatable, Sendable {
    package let queryHeads: Int
    package let keyValueHeads: Int
    package let headDim: Int
    package let ropeTheta: Double
    package let partialRotaryFactor: Double

    package init(queryHeads: Int, keyValueHeads: Int, headDim: Int,
                 ropeTheta: Double, partialRotaryFactor: Double) {
        self.queryHeads = queryHeads
        self.keyValueHeads = keyValueHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
    }
}

package struct GTurboManifestV2GatedDeltaNet: Codable, Equatable, Sendable {
    package let keyHeads: Int
    package let valueHeads: Int
    package let keyHeadDim: Int
    package let valueHeadDim: Int
    package let convolutionKernel: Int
    package let stateDType: String

    package init(keyHeads: Int, valueHeads: Int, keyHeadDim: Int,
                 valueHeadDim: Int, convolutionKernel: Int, stateDType: String) {
        self.keyHeads = keyHeads
        self.valueHeads = valueHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convolutionKernel = convolutionKernel
        self.stateDType = stateDType
    }
}

package struct GTurboManifestV2Arch: Codable, Equatable, Sendable {
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
    package let fullAttention: GTurboManifestV2FullAttention
    package let gatedDeltaNet: GTurboManifestV2GatedDeltaNet
    package let finalRopeTheta: Double

    package init(modelFamily: String, hiddenSize: Int, vocabSize: Int,
                 numLayers: Int, layerKinds: [String], numRoutedExperts: Int,
                 topKExperts: Int, routedExpertIntermediateSize: Int,
                 sharedExpertIntermediateSize: Int, routerActivation: String,
                 routedExpertActivation: String, sharedExpertActivation: String,
                 sharedExpertGateActivation: String, tieWordEmbeddings: Bool,
                 fullAttention: GTurboManifestV2FullAttention,
                 gatedDeltaNet: GTurboManifestV2GatedDeltaNet,
                 finalRopeTheta: Double) {
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
        self.fullAttention = fullAttention
        self.gatedDeltaNet = gatedDeltaNet
        self.finalRopeTheta = finalRopeTheta
    }
}

package struct GTurboManifestQuantSlotV2: Codable, Equatable, Sendable {
    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct GTurboManifestQuantV2: Codable, Equatable, Sendable {
    package let roles: [String: GTurboManifestQuantSlotV2]

    package init(roles: [String: GTurboManifestQuantSlotV2]) {
        self.roles = roles
    }
}

package struct GTurboManifestV2: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestV2Arch
    package let quant: GTurboManifestQuantV2
    package let files: [String: GTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64

    package init(magic: String = GTurboFormatV1.magic,
                 versionMajor: Int = GTurboFormatV2.versionMajor,
                 versionMinor: Int = 0, flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestV2Arch,
                 quant: GTurboManifestQuantV2,
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
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
    }
}

package enum GTurboManifestV2Codec {
    package static func decode(_ data: Data) throws -> GTurboManifestV2 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest, data: data)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> GTurboManifestV2 {
        do {
            return try JSONDecoder().decode(GTurboManifestV2.self, from: data)
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func encode(_ manifest: GTurboManifestV2) throws -> Data {
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

    package static func validate(_ manifest: GTurboManifestV2,
                                 data: Data?) throws {
        guard manifest.magic == GTurboFormatV1.magic else {
            throw GTurboFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV2.versionMajor,
              manifest.versionMinor >= 0 else {
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        guard manifest.arch.modelFamily == "qwen3_5_moe_text" else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch.modelFamily", reason: "unsupported model family")
        }
        for flag in manifest.flags.keys where !GTurboFormatV2.knownFlags.contains(flag) {
            throw GTurboFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v2 flag")
        }
        try validateKeys(data)
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0,
              manifest.expertsPerLayer == manifest.arch.numRoutedExperts,
              manifest.numLayers == manifest.arch.numLayers,
              manifest.expertStride > 0,
              manifest.expertStride % GTurboFormatV1.alignmentBytes == 0 else {
            throw GTurboFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        let arch = manifest.arch
        guard arch.hiddenSize > 0, arch.vocabSize > 0,
              arch.numRoutedExperts > 0,
              arch.topKExperts > 0, arch.topKExperts <= arch.numRoutedExperts,
              arch.routedExpertIntermediateSize > 0,
              arch.sharedExpertIntermediateSize > 0,
              arch.layerKinds.count == arch.numLayers,
              arch.layerKinds.allSatisfy({ $0 == "gatedDeltaNet" || $0 == "fullAttention" }),
              arch.layerKinds.contains("fullAttention"),
              arch.layerKinds.contains("gatedDeltaNet"),
              !arch.routerActivation.isEmpty,
              !arch.routedExpertActivation.isEmpty,
              !arch.sharedExpertActivation.isEmpty,
              !arch.sharedExpertGateActivation.isEmpty,
              !arch.tieWordEmbeddings,
              arch.fullAttention.queryHeads > 0,
              arch.fullAttention.keyValueHeads > 0,
              arch.fullAttention.keyValueHeads <= arch.fullAttention.queryHeads,
              arch.fullAttention.headDim > 0,
              arch.fullAttention.ropeTheta.isFinite,
              arch.fullAttention.ropeTheta > 0,
              arch.fullAttention.partialRotaryFactor.isFinite,
              arch.fullAttention.partialRotaryFactor > 0,
              arch.fullAttention.partialRotaryFactor <= 1,
              arch.gatedDeltaNet.keyHeads > 0,
              arch.gatedDeltaNet.valueHeads > 0,
              arch.gatedDeltaNet.keyHeadDim > 0,
              arch.gatedDeltaNet.valueHeadDim > 0,
              arch.gatedDeltaNet.convolutionKernel > 0,
              arch.gatedDeltaNet.stateDType == "FP32",
              arch.finalRopeTheta.isFinite,
              arch.finalRopeTheta > 0 else {
            throw GTurboFormatError.invalid(field: "manifest.arch", reason: "invalid architecture values")
        }
        guard !manifest.quant.roles.isEmpty else {
            throw GTurboFormatError.invalid(field: "manifest.quant.roles", reason: "must not be empty")
        }
        for (role, slot) in manifest.quant.roles {
            guard GTurboFormatV2.knownQuantRoles.contains(role),
                  slot.weightBits > 0, slot.weightBits <= 32,
                  !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                  !slot.biasType.isEmpty, slot.groupSize > 0 else {
                throw GTurboFormatError.invalid(
                    field: "manifest.quant.roles.\(role)", reason: "invalid quantization role")
            }
        }
        try validateFiles(manifest.files)
    }

    private static func validateFiles(_ files: [String: GTurboManifestFileV1]) throws {
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        var canonicalPaths: [String: String] = [:]
        for path in files.keys.sorted() {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path).sha256",
                    reason: "expected 64 hexadecimal characters")
            }
        }
    }

    private static func validateKeys(_ data: Data?) throws {
        guard let data else { return }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "expected object")
        }
        try rejectUnknownKeys(root, allowed: [
            "magic", "versionMajor", "versionMinor", "flags", "modelID",
            "sourceSnapshotHash", "arch", "quant", "files", "expertsPerLayer",
            "numLayers", "expertStride",
        ], field: "manifest")
        if let arch = root["arch"] as? [String: Any] {
            try rejectUnknownKeys(arch, allowed: [
                "modelFamily", "hiddenSize", "vocabSize", "numLayers", "layerKinds",
                "numRoutedExperts", "topKExperts", "routedExpertIntermediateSize",
                "sharedExpertIntermediateSize", "routerActivation", "routedExpertActivation",
                "sharedExpertActivation", "sharedExpertGateActivation", "tieWordEmbeddings",
                "fullAttention", "gatedDeltaNet", "finalRopeTheta",
            ], field: "manifest.arch")
            if let fullAttention = arch["fullAttention"] as? [String: Any] {
                try rejectUnknownKeys(fullAttention, allowed: [
                    "queryHeads", "keyValueHeads", "headDim", "ropeTheta",
                    "partialRotaryFactor",
                ], field: "manifest.arch.fullAttention")
            }
            if let gatedDeltaNet = arch["gatedDeltaNet"] as? [String: Any] {
                try rejectUnknownKeys(gatedDeltaNet, allowed: [
                    "keyHeads", "valueHeads", "keyHeadDim", "valueHeadDim",
                    "convolutionKernel", "stateDType",
                ], field: "manifest.arch.gatedDeltaNet")
            }
        }
        if let quant = root["quant"] as? [String: Any],
           let roles = quant["roles"] as? [String: Any] {
            try rejectUnknownKeys(quant, allowed: ["roles"], field: "manifest.quant")
            for (role, value) in roles {
                guard let slot = value as? [String: Any] else { continue }
                try rejectUnknownKeys(slot, allowed: [
                    "weightBits", "scheme", "scaleType", "biasType", "groupSize",
                ], field: "manifest.quant.roles.\(role)")
            }
        }
        if let files = root["files"] as? [String: Any] {
            for (path, value) in files {
                guard let entry = value as? [String: Any] else { continue }
                try rejectUnknownKeys(entry, allowed: ["size", "sha256"],
                                      field: "manifest.files.\(path)")
            }
        }
    }

    private static func rejectUnknownKeys(_ object: [String: Any],
                                          allowed: Set<String>, field: String) throws {
        if let unknown = object.keys.first(where: { !allowed.contains($0) }) {
            throw GTurboFormatError.invalid(field: "\(field).\(unknown)", reason: "unknown v2 field")
        }
    }
}

package enum GTurboManifestDocument: Equatable, Sendable {
    case v1(GTurboManifestV1)
    case v2(GTurboManifestV2)
}

package enum GTurboManifestVersionedCodec {
    package static func decode(_ data: Data) throws -> GTurboManifestDocument {
        let root: [String: Any]
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw GTurboFormatError.invalid(field: "manifest.json", reason: "expected object")
            }
            root = object
        } catch let error as GTurboFormatError {
            throw error
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
        guard let major = root["versionMajor"] as? Int else {
            throw GTurboFormatError.invalid(field: "manifest.versionMajor", reason: "missing")
        }
        switch major {
        case GTurboFormatV1.versionMajor:
            return .v1(try GTurboManifestCodec.decode(data))
        case GTurboFormatV2.versionMajor:
            return .v2(try GTurboManifestV2Codec.decode(data))
        default:
            throw GTurboFormatError.invalid(
                field: "manifest.version", reason: "unsupported major version \(major)")
        }
    }
}