import Foundation
import TurboFieldfareFormat

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let sharedExpertGate: ManifestQuantSlot?
    public let routedExpert: ManifestQuantSlot
    public let lmHead: ManifestQuantSlot?
}

public struct ManifestMTPContract: Decodable, Equatable, Sendable {
    public let baseHiddenVariant: String
    public let concatOrder: String
    public let hiddenVariant: String
    public let mtpPositionMode: String
    public let mtpQuantGroupSize: Int
    public let mtpQuantMode: String
}

public struct ManifestMTPQuantization: Decodable, Equatable, Sendable {
    public let bits: Int
    public let groupSize: Int
}

public struct ManifestMTP: Decodable, Equatable, Sendable {
    public let predictLayers: Int
    public let tensorPrefix: String
    public let usesDedicatedEmbeddings: Bool
    public let depthMax: Int?
    public let contract: ManifestMTPContract?
    public let tensorQuantization: [String: ManifestMTPQuantization]

    private enum CodingKeys: String, CodingKey {
        case predictLayers
        case tensorPrefix
        case usesDedicatedEmbeddings
        case depthMax
        case contract
        case tensorQuantization
    }

    init(predictLayers: Int, tensorPrefix: String,
         usesDedicatedEmbeddings: Bool, depthMax: Int?,
         contract: ManifestMTPContract?,
         tensorQuantization: [String: ManifestMTPQuantization]) {
        self.predictLayers = predictLayers
        self.tensorPrefix = tensorPrefix
        self.usesDedicatedEmbeddings = usesDedicatedEmbeddings
        self.depthMax = depthMax
        self.contract = contract
        self.tensorQuantization = tensorQuantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        predictLayers = try container.decode(Int.self, forKey: .predictLayers)
        tensorPrefix = try container.decode(String.self, forKey: .tensorPrefix)
        usesDedicatedEmbeddings = try container.decode(
            Bool.self, forKey: .usesDedicatedEmbeddings)
        depthMax = try container.decodeIfPresent(Int.self, forKey: .depthMax)
        contract = try container.decodeIfPresent(
            ManifestMTPContract.self, forKey: .contract)
        tensorQuantization = try container.decodeIfPresent(
            [String: ManifestMTPQuantization].self,
            forKey: .tensorQuantization) ?? [:]
    }
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let mtp: ManifestMTP?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64

    init(magic: String, versionMajor: Int, versionMinor: Int,
         flags: [String: Bool], modelID: String, sourceSnapshotHash: String?,
         arch: ManifestArch, quant: ManifestQuant?, mtp: ManifestMTP? = nil,
         files: [String: ManifestFileEntry], expertsPerLayer: Int,
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

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = GTurboFormatV1.knownFlags

    /// Select the runtime contract encoded by a supported manifest version.
    package static func inferArchitecture(data: Data) throws -> ArchConfig {
        let root: [String: Any]
        do {
            guard let object = try JSONSerialization.jsonObject(with: data)
                    as? [String: Any] else {
                throw ModelError.indexCorrupt(detail: "manifest.json is not a JSON object")
            }
            root = object
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        let version = (root["versionMajor"] as? NSNumber)?.intValue
        let modelFamily = (root["arch"] as? [String: Any])?["modelFamily"] as? String
        switch (version, modelFamily) {
        case (GTurboFormatV2.versionMajor, "qwen3_5_moe_text"):
            return .qwen36MoeText
        case (GTurboFormatV3.versionMajor, "qwen4_exp_text"):
            return .qwen38FlashNextText
        default:
            return .gemma4_26B_A4B
        }
    }

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig? = nil,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    package static func decode(data: Data,
                               expecting: ArchConfig? = nil) throws -> Manifest {
        let manifest: Manifest
        do {
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let version = (root?["versionMajor"] as? NSNumber)?.intValue
            let arch = root?["arch"] as? [String: Any]
            if version == GTurboFormatV2.versionMajor,
               arch?["modelFamily"] as? String == "qwen3_5_moe_text" {
                manifest = try normalize(GTurboManifestV2Codec.decode(data))
            } else if version == GTurboFormatV3.versionMajor,
                      arch?["modelFamily"] as? String == "qwen4_exp_text" {
                manifest = try normalize(GTurboManifestV3Codec.decode(data))
            } else {
                let wire = try GTurboManifestCodec.decodeUnchecked(data)
                guard wire.magic == GTurboFormatV1.magic else {
                    throw ModelError.notAGTurboDirectory
                }
                guard wire.versionMajor == GTurboFormatV1.versionMajor,
                      wire.versionMinor >= 0 else {
                    throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                        minor: wire.versionMinor)
                }
                for key in wire.flags.keys where !GTurboFormatV1.knownFlags.contains(key) {
                    throw ModelError.unknownFlag(name: key)
                }
                if wire.expertStride % GTurboFormatV1.alignmentBytes != 0 {
                    throw ModelError.expertStrideNotPageAligned(
                        stride: wire.expertStride,
                        pageSize: Int(GTurboFormatV1.alignmentBytes))
                }
                try GTurboManifestCodec.validate(wire)
                manifest = Manifest(wire: wire)
            }
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        let canonicalExpected = try inferArchitecture(data: data)
        if let expecting, canonicalExpected.modelFamily != expecting.modelFamily {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: String(describing: expecting.modelFamily),
                actual: String(describing: canonicalExpected.modelFamily))
        }
        try validate(manifest, against: expecting ?? canonicalExpected)
        return manifest
    }

    private static func normalize(_ wire: GTurboManifestV2) throws -> Manifest {
        func slot(_ role: String) throws -> ManifestQuantSlot {
            guard let value = wire.quant.roles[role] else {
                throw ModelError.indexCorrupt(
                    detail: "manifest.quant.roles.\(role) is required")
            }
            return ManifestQuantSlot(
                weightBits: value.weightBits,
                scheme: value.scheme,
                scaleType: value.scaleType,
                biasType: value.biasType,
                groupSize: value.groupSize)
        }
        let arch = wire.arch
        let layerMask = arch.layerKinds.map { $0 == "fullAttention" ? 1 : 0 }
        return Manifest(
            magic: wire.magic,
            versionMajor: wire.versionMajor,
            versionMinor: wire.versionMinor,
            flags: wire.flags,
            modelID: wire.modelID,
            sourceSnapshotHash: wire.sourceSnapshotHash,
            arch: ManifestArch(
                hiddenSize: arch.hiddenSize,
                ffnIntermediate: arch.sharedExpertIntermediateSize,
                moeIntermediateSize: arch.routedExpertIntermediateSize,
                numHeads: arch.fullAttention.queryHeads,
                numKVHeads: arch.fullAttention.keyValueHeads,
                numFullKVHeads: arch.fullAttention.keyValueHeads,
                headDim: arch.fullAttention.headDim,
                fullHeadDim: arch.fullAttention.headDim,
                vocabSize: arch.vocabSize,
                slidingWindow: 0,
                finalLogitSoftcap: 0,
                ropeTheta: arch.finalRopeTheta,
                fullRopeTheta: arch.fullAttention.ropeTheta,
                partialRotaryFactor: arch.fullAttention.partialRotaryFactor,
                numLayers: arch.numLayers,
                numExperts: arch.numRoutedExperts,
                topKExperts: arch.topKExperts,
                tieWordEmbeddings: arch.tieWordEmbeddings,
                attentionKEqV: false,
                hiddenActivation: arch.routedExpertActivation,
                fullAttentionLayerMask: layerMask),
            quant: ManifestQuant(
                embedding: try slot("embedding"),
                attention: try slot("attention"),
                router: try slot("router"),
                sharedExpert: try slot("sharedExpert"),
                sharedExpertGate: wire.quant.roles["sharedExpertGate"].map {
                    ManifestQuantSlot(
                        weightBits: $0.weightBits,
                        scheme: $0.scheme,
                        scaleType: $0.scaleType,
                        biasType: $0.biasType,
                        groupSize: $0.groupSize)
                },
                routedExpert: try slot("routedExpert"),
                lmHead: nil),
            mtp: nil,
            files: wire.files.mapValues {
                ManifestFileEntry(size: $0.size, sha256: $0.sha256)
            },
            expertsPerLayer: wire.expertsPerLayer,
            numLayers: wire.numLayers,
            expertStride: wire.expertStride)
    }

    private static func normalize(_ wire: GTurboManifestV3) throws -> Manifest {
        func slot(_ role: String) throws -> ManifestQuantSlot {
            guard let value = wire.quant.roles[role] else {
                throw ModelError.indexCorrupt(
                    detail: "manifest.quant.roles.\(role) is required")
            }
            return ManifestQuantSlot(
                weightBits: value.weightBits,
                scheme: value.scheme,
                scaleType: value.scaleType,
                biasType: value.biasType,
                groupSize: value.groupSize)
        }
        let arch = wire.arch
        let layerMask = arch.layerKinds.map { $0 == "sparseAttention" ? 1 : 0 }
        return Manifest(
            magic: wire.magic,
            versionMajor: wire.versionMajor,
            versionMinor: wire.versionMinor,
            flags: wire.flags,
            modelID: wire.modelID,
            sourceSnapshotHash: wire.sourceSnapshotHash,
            arch: ManifestArch(
                hiddenSize: arch.hiddenSize,
                ffnIntermediate: arch.sharedExpertIntermediateSize,
                moeIntermediateSize: arch.routedExpertIntermediateSize,
                numHeads: arch.sparseAttention.queryHeads,
                numKVHeads: arch.sparseAttention.keyValueHeads,
                numFullKVHeads: arch.sparseAttention.keyValueHeads,
                headDim: arch.sparseAttention.headDim,
                fullHeadDim: arch.sparseAttention.headDim,
                vocabSize: arch.vocabSize,
                slidingWindow: 0,
                finalLogitSoftcap: 0,
                ropeTheta: arch.sparseAttention.ropeTheta,
                fullRopeTheta: arch.sparseAttention.ropeTheta,
                partialRotaryFactor: arch.sparseAttention.partialRotaryFactor,
                numLayers: arch.numLayers,
                numExperts: arch.numRoutedExperts,
                topKExperts: arch.topKExperts,
                tieWordEmbeddings: arch.tieWordEmbeddings,
                attentionKEqV: false,
                hiddenActivation: arch.routedExpertActivation,
                fullAttentionLayerMask: layerMask),
            quant: ManifestQuant(
                embedding: try slot("embedding"),
                attention: try slot("attention"),
                router: try slot("router"),
                sharedExpert: try slot("sharedExpert"),
                sharedExpertGate: try slot("sharedExpertGate"),
                routedExpert: try slot("routedExpert"),
                lmHead: wire.quant.roles["lmHead"].map {
                    ManifestQuantSlot(
                        weightBits: $0.weightBits,
                        scheme: $0.scheme,
                        scaleType: $0.scaleType,
                        biasType: $0.biasType,
                        groupSize: $0.groupSize)
                }),
            mtp: wire.mtp.map {
                ManifestMTP(
                    predictLayers: $0.predictLayers,
                    tensorPrefix: $0.tensorPrefix,
                    usesDedicatedEmbeddings: $0.usesDedicatedEmbeddings,
                    depthMax: $0.depthMax,
                    contract: $0.contract.map {
                        ManifestMTPContract(
                            baseHiddenVariant: $0.baseHiddenVariant,
                            concatOrder: $0.concatOrder,
                            hiddenVariant: $0.hiddenVariant,
                            mtpPositionMode: $0.mtpPositionMode,
                            mtpQuantGroupSize: $0.mtpQuantGroupSize,
                            mtpQuantMode: $0.mtpQuantMode)
                    },
                    tensorQuantization: $0.tensorQuantization.mapValues {
                        ManifestMTPQuantization(
                            bits: $0.bits, groupSize: $0.groupSize)
                    })
            },
            files: wire.files.mapValues {
                ManifestFileEntry(size: $0.size, sha256: $0.sha256)
            },
            expertsPerLayer: wire.expertsPerLayer,
            numLayers: wire.numLayers,
            expertStride: wire.expertStride)
    }

    package static func decodeDocument(data: Data) throws -> ManifestDocument {
        do {
            switch try GTurboManifestVersionedCodec.decode(data) {
            case let .v1(wire):
                return .v1(Manifest(wire: wire))
            case let .v2(wire):
                return .v2(ManifestV2(wire: wire))
            case let .v3(wire):
                return .v3(ManifestV3(wire: wire))
            }
        } catch let error as ModelError {
            throw error
        } catch let error as GTurboFormatError {
            if case .invalid(field: "manifest.magic", reason: "expected GTURBO") = error {
                throw ModelError.notAGTurboDirectory
            }
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig) throws {
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant, expected: expected)
        } else if expected.numLayers == ArchConfig.gemma4_26B_A4B.numLayers,
                  expected.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant,
                                      expected: ArchConfig) throws {
        let expectedGroupSize = expected.modelFamily == .qwen38FlashNextText
            ? Quantization.qwen38GroupSize
            : Quantization.groupSize
        let embedding = quant.embedding
        let embeddingIsQ4 = embedding.weightBits == 4
            && embedding.groupSize == expectedGroupSize
        let embeddingIsQ8Qwen = expected.modelFamily == .qwen38FlashNextText
            && embedding.weightBits == 8
            && embedding.groupSize == 64
        guard (embeddingIsQ4 || embeddingIsQ8Qwen),
              embedding.scheme.lowercased() == "affine",
              embedding.scaleType.lowercased() == "bf16",
              embedding.biasType.lowercased() == "bf16" else {
            throw ModelError.indexCorrupt(detail: "unsupported quantization for embedding")
        }
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("attention", quant.attention, [4]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [4]),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == expectedGroupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
        if expected.modelFamily == .qwen36MoeText {
            let isRawBF16Router = quant.router.weightBits == 16
                && quant.router.scheme.lowercased() == "none"
                && quant.router.scaleType.lowercased() == "none"
                && quant.router.biasType.lowercased() == "none"
                && quant.router.groupSize == 1
            let isLegacyAffineRouter = quant.router.weightBits == 8
                && quant.router.scheme.lowercased() == "affine"
                && quant.router.scaleType.lowercased() == "bf16"
                && quant.router.biasType.lowercased() == "bf16"
                && quant.router.groupSize == expectedGroupSize
            guard isRawBF16Router || isLegacyAffineRouter else {
                throw ModelError.indexCorrupt(
                    detail: "unsupported quantization for router")
            }
        } else if expected.modelFamily == .qwen38FlashNextText {
            let isRawBF16Router = quant.router.weightBits == 16
                && quant.router.scheme.lowercased() == "none"
                && quant.router.scaleType.lowercased() == "none"
                && quant.router.biasType.lowercased() == "none"
                && quant.router.groupSize == 1
            let isLegacyAffineRouter = quant.router.weightBits == 8
                && quant.router.scheme.lowercased() == "affine"
                && quant.router.scaleType.lowercased() == "bf16"
                && quant.router.biasType.lowercased() == "bf16"
                && quant.router.groupSize == expectedGroupSize
            guard isRawBF16Router || isLegacyAffineRouter else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for router")
            }
        } else {
            guard quant.router.weightBits == 8,
                  quant.router.scheme.lowercased() == "affine",
                  quant.router.scaleType.lowercased() == "bf16",
                  quant.router.biasType.lowercased() == "bf16",
                  quant.router.groupSize == expectedGroupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for router")
            }
        }
        if expected.modelFamily == .qwen38FlashNextText {
            guard let gate = quant.sharedExpertGate,
                  [4, 8].contains(gate.weightBits),
                  gate.scheme.lowercased() == "affine",
                  gate.scaleType.lowercased() == "bf16",
                  gate.biasType.lowercased() == "bf16",
                  gate.groupSize == expectedGroupSize else {
                throw ModelError.indexCorrupt(
                    detail: "unsupported quantization for sharedExpertGate")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)
    }
}

private extension ManifestFileEntry {
    init(wire: GTurboManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

private extension ManifestArch {
    init(wire: GTurboManifestArchV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  ffnIntermediate: wire.ffnIntermediate,
                  moeIntermediateSize: wire.moeIntermediateSize,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  finalLogitSoftcap: wire.finalLogitSoftcap,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  numLayers: wire.numLayers,
                  numExperts: wire.numExperts,
                  topKExperts: wire.topKExperts,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  hiddenActivation: wire.hiddenActivation,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask)
    }
}

private extension ManifestQuantSlot {
    init(wire: GTurboManifestQuantSlotV1) {
        self.init(weightBits: wire.weightBits, scheme: wire.scheme,
                  scaleType: wire.scaleType, biasType: wire.biasType,
                  groupSize: wire.groupSize)
    }
}

private extension ManifestQuant {
    init(wire: GTurboManifestQuantV1) {
        self.init(embedding: ManifestQuantSlot(wire: wire.embedding),
                  attention: ManifestQuantSlot(wire: wire.attention),
                  router: ManifestQuantSlot(wire: wire.router),
                  sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
                  sharedExpertGate: nil,
                  routedExpert: ManifestQuantSlot(wire: wire.routedExpert),
                  lmHead: nil)
    }
}

private extension Manifest {
    init(wire: GTurboManifestV1) {
        self.init(magic: wire.magic,
                  versionMajor: wire.versionMajor,
                  versionMinor: wire.versionMinor,
                  flags: wire.flags,
                  modelID: wire.modelID,
                  sourceSnapshotHash: wire.sourceSnapshotHash,
                  arch: ManifestArch(wire: wire.arch),
                  quant: wire.quant.map(ManifestQuant.init(wire:)),
                  mtp: nil,
                  files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
                  expertsPerLayer: wire.expertsPerLayer,
                  numLayers: wire.numLayers,
                  expertStride: wire.expertStride)
    }
}
