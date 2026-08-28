import Testing
import Foundation
import Darwin
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct ManifestReaderTests {

    @Test func rejectsManifestFIFOWithoutBlocking() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = dir.appendingPathComponent("manifest.json")
        try FileManager.default.removeItem(at: manifest)
        #expect(mkfifo(manifest.path, 0o600) == 0)

        #expect(throws: ModelError.self) {
            try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(_ overrides: [String: Any] = [:],
                                 flags: [String: Bool] = ["streamingPresent": true,
                                                          "turboQuantKV": false,
                                                          "aneSharedExpert": false],
                                 archOverrides: [String: Any] = [:],
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .gemma4Toy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        for (k, v) in archOverrides { archDict[k] = v }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = ["size": 16384, "sha256": String(repeating: "0", count: 64)]
            }
        }

        var root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    static func quant(sharedExpertBits: Int = 4,
                      routerBits: Int = 8) -> [String: Any] {
        func slot(_ bits: Int) -> [String: Any] {
            [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": Quantization.groupSize,
            ]
        }
        return [
            "embedding": slot(4),
            "attention": slot(4),
            "router": slot(routerBits),
            "sharedExpert": slot(sharedExpertBits),
            "routedExpert": slot(4),
        ]
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    @Test func dispatchesV2ManifestWithoutUsingV1ArchValidation() throws {
        let manifest = GTurboManifestV2(
            flags: ["streamingPresent": true, "untiedHead": true],
            modelID: "qwen36/model", sourceSnapshotHash: "snapshot",
            arch: GTurboManifestV2Arch(
                modelFamily: "qwen3_5_moe_text", hiddenSize: 2_048,
                vocabSize: 248_320, numLayers: 4,
                layerKinds: ["gatedDeltaNet", "gatedDeltaNet", "gatedDeltaNet", "fullAttention"],
                numRoutedExperts: 256, topKExperts: 8,
                routedExpertIntermediateSize: 512,
                sharedExpertIntermediateSize: 2_048,
                routerActivation: "softmax", routedExpertActivation: "silu",
                sharedExpertActivation: "silu", sharedExpertGateActivation: "sigmoid",
                tieWordEmbeddings: false,
                fullAttention: GTurboManifestV2FullAttention(
                    queryHeads: 16, keyValueHeads: 2, headDim: 256,
                    ropeTheta: 1_000_000, partialRotaryFactor: 0.25),
                gatedDeltaNet: GTurboManifestV2GatedDeltaNet(
                    keyHeads: 16, valueHeads: 32, keyHeadDim: 128,
                    valueHeadDim: 128, convolutionKernel: 4, stateDType: "FP32"),
                finalRopeTheta: 1_000_000),
            quant: GTurboManifestQuantV2(roles: [
                "embedding": GTurboManifestQuantSlotV2(
                    weightBits: 4, scheme: "affine", scaleType: "BF16",
                    biasType: "BF16", groupSize: 64),
            ]),
            files: [
                "model_weights.bin": GTurboManifestFileV1(size: 1, sha256: String(repeating: "0", count: 64)),
            ],
            expertsPerLayer: 256, numLayers: 4, expertStride: 16_384)
        let data = try GTurboManifestV2Codec.encode(manifest)

        guard case let .v2(decoded) = try ManifestReader.decodeDocument(data: data) else {
            Issue.record("expected v2 manifest dispatch")
            return
        }
        #expect(decoded.wire.arch.modelFamily == "qwen3_5_moe_text")
        #expect(decoded.wire.arch.gatedDeltaNet.stateDType == "FP32")
    }

    @Test func v3InspectionAndRuntimeNormalizationSucceed() throws {
        let slot = GTurboManifestQuantSlotV2(
            weightBits: 4, scheme: "affine", scaleType: "BF16",
            biasType: "BF16", groupSize: 32)
        let manifest = GTurboManifestV3(
            flags: [
                "streamingPresent": true,
                "untiedHead": true,
                "ngramStreamingPresent": true,
            ],
            modelID: "qwen38/flash-next",
            sourceSnapshotHash: "de597762aa61387c89590a46582222a261ce0387",
            arch: GTurboManifestV3Arch(
                modelFamily: "qwen4_exp_text", hiddenSize: 2_560,
                vocabSize: 248_320, numLayers: 48,
                layerKinds: (0..<48).map {
                    ($0 + 1) % 4 == 0 ? "sparseAttention" : "gatedDeltaNet"
                },
                numRoutedExperts: 512, topKExperts: 10,
                routedExpertIntermediateSize: 640,
                sharedExpertIntermediateSize: 640,
                routerActivation: "sigmoid", routedExpertActivation: "silu",
                sharedExpertActivation: "silu",
                sharedExpertGateActivation: "sigmoid",
                tieWordEmbeddings: false,
                sparseAttention: GTurboManifestV3SparseAttention(
                    queryHeads: 24, keyValueHeads: 2, headDim: 256,
                    ropeTheta: 10_000_000, partialRotaryFactor: 0.25,
                    indexerHeads: 4, indexerKeyValueHeads: 1,
                    indexerHeadDim: 128, indexerCompressRatio: 4,
                    indexerBudget: 2_048),
                gatedDeltaNet: GTurboManifestV2GatedDeltaNet(
                    keyHeads: 16, valueHeads: 48, keyHeadDim: 128,
                    valueHeadDim: 128, convolutionKernel: 4,
                    stateDType: "FP32"),
                hyperConnection: GTurboManifestV3HyperConnection(
                    streamCount: 4, lowRankSize: 320),
                ple: GTurboManifestV3PLE(
                    layerIDs: [2], embeddingSize: 2_560,
                    convolutionKernel: 4, ngramSize: 3, headsPerNgram: 8,
                    vocabSizeBase: 20_000_000, splitParts: 128,
                    vocabSizeDivisor: 128,
                    layoutFile: "packed_ngrams/layout.json")),
            quant: GTurboManifestQuantV2(roles: Dictionary(
                uniqueKeysWithValues: GTurboFormatV3.knownQuantRoles.map {
                    ($0, slot)
                })),
            files: [
                "model_weights.bin": GTurboManifestFileV1(
                    size: 16_384, sha256: String(repeating: "0", count: 64)),
                "packed_experts/layout.json": GTurboManifestFileV1(
                    size: 1, sha256: String(repeating: "0", count: 64)),
                "packed_ngrams/layout.json": GTurboManifestFileV1(
                    size: 1, sha256: String(repeating: "0", count: 64)),
            ],
            expertsPerLayer: 512, numLayers: 48,
            expertStride: 16_384)
        let data = try GTurboManifestV3Codec.encode(manifest)

        guard case let .v3(decoded) = try ManifestReader.decodeDocument(data: data) else {
            Issue.record("expected v3 manifest dispatch")
            return
        }
        #expect(decoded.wire.arch.modelFamily == "qwen4_exp_text")
        #expect(decoded.wire.arch.hyperConnection.streamCount == 4)
        let normalized = try ManifestReader.decode(
            data: data, expecting: .qwen38FlashNextText)
        #expect(normalized.versionMajor == 3)
        #expect(normalized.arch.hiddenSize == 2_560)
        #expect(normalized.arch.fullAttentionLayerMask[3] == 1)
        #expect(normalized.quant?.attention.groupSize == 32)
        #expect {
            try ManifestReader.decode(data: data, expecting: .gemma4_26B_A4B)
        } throws: { error in
            guard case ModelError.archMismatch(field: "modelFamily", _, _) = error else {
                return false
            }
            return true
        }
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4Toy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: toy,
                                        maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotAGTurboDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notAGTurboDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "newFangledOption": true])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "turboQuantKV": true,
                                                           "aneSharedExpert": false])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func defersPackedLayerFilenamesToLayoutValidation() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/experts_a.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/experts_b.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(manifest.files["packed_experts/experts_a.bin"] != nil)
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_01.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }

    @Test func packedNgramLayoutCrossValidatesManifestFiles() throws {
        let page = GTurboFormatV1.alignmentBytes
        let wire = GTurboPackedNgramsLayoutV1(
            layer: 2, splitParts: 2, groupSize: 32,
            shards: (0..<2).map { shard in
                GTurboNgramShardV1(
                    shard: shard,
                    file: "shard_\(String(format: "%03d", shard)).bin",
                    fileSize: 3 * page,
                    weight: GTurboNgramComponentV1(
                        offset: 0, size: 128,
                        dtype: "U32", shape: [4, 8], bits: 4),
                    scales: GTurboNgramComponentV1(
                        offset: page, size: 16,
                        dtype: "BF16", shape: [4, 2], bits: nil),
                    biases: GTurboNgramComponentV1(
                        offset: 2 * page, size: 16,
                        dtype: "BF16", shape: [4, 2], bits: nil))
            })
        let files = Dictionary(uniqueKeysWithValues: [
            ("packed_ngrams/layout.json", ManifestFileEntry(
                size: 1, sha256: String(repeating: "0", count: 64))),
        ] + wire.shards.map {
            ("packed_ngrams/\($0.file)", ManifestFileEntry(
                size: $0.fileSize, sha256: String(repeating: "0", count: 64)))
        })
        let manifest = Manifest(
            magic: GTurboFormatV1.magic, versionMajor: 3, versionMinor: 0,
            flags: [:], modelID: "qwen38/test", sourceSnapshotHash: nil,
            arch: ManifestArch(
                hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
                numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
                headDim: 16, fullHeadDim: 16, vocabSize: 1_024,
                slidingWindow: 0, finalLogitSoftcap: 0,
                ropeTheta: 10_000, fullRopeTheta: 10_000,
                partialRotaryFactor: 0.25, numLayers: 2,
                numExperts: 8, topKExperts: 2,
                tieWordEmbeddings: false, attentionKEqV: false,
                hiddenActivation: "silu", fullAttentionLayerMask: [0, 1]),
            quant: nil, files: files, expertsPerLayer: 8,
            numLayers: 2, expertStride: page)
        let data = try GTurboPackedNgramsLayoutCodec.encode(wire)

        let decoded = try PackedNgramsLayoutReader.decode(
            data: data, manifest: manifest)

        #expect(decoded.splitParts == 2)
        #expect(decoded.shard(1).file == "shard_001.bin")
    }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 2 layers (both full), hidden 64,
    /// vocab 1024, 8 experts. Numbers are intentionally toy.
    static func gemma4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 256,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 1_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh"
        )
    }
}
