import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite
struct RangeCopyPlannerTests {
    @Test func qwenConfigAcceptsCanonicalExpertsPerTokenKey() throws {
        let snapshotDirectory = temporaryRoot("qwen-config")
        defer { try? FileManager.default.removeItem(atPath: snapshotDirectory) }
        _ = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0x5157_36,
            modelFamily: "qwen3_5_moe_text")
        let configPath = (snapshotDirectory as NSString).appendingPathComponent("config.json")
        let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
        var config = try #require(
            JSONSerialization.jsonObject(with: configData) as? [String: Any])
        var textConfig = try #require(config["text_config"] as? [String: Any])
        textConfig["num_experts_per_tok"] = textConfig.removeValue(forKey: "top_k_experts")
        config["text_config"] = textConfig
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: configPath))

        let arch = try ArchInfo.load(configPath: configPath)

        #expect(arch.topKExperts == 2)
    }

    @Test func qwen38ConfigNormalizesArchitectureAndDimensions() throws {
        for rawModelFamily in ["qwen4_exp", "qwen4_exp_text"] {
            let snapshotDirectory = temporaryRoot("qwen38-config-\(rawModelFamily)")
            defer { try? FileManager.default.removeItem(atPath: snapshotDirectory) }
            try FileManager.default.createDirectory(
                atPath: snapshotDirectory,
                withIntermediateDirectories: true)
            let configPath = (snapshotDirectory as NSString).appendingPathComponent("config.json")
            let textConfig: [String: Any] = [
                "model_type": rawModelFamily,
                "hidden_size": 2560,
                "shared_expert_intermediate_size": 640,
                "moe_intermediate_size": 640,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248_320,
                "num_hidden_layers": 48,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_compress_ratio": 4,
                "indexer_budget": 2048,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "ngram_vocab_size_base": 20_000_000,
                "split_ngram_parts": 128,
                "make_ngram_vocab_size_divisible_by": 128,
                "mamba_ssm_dtype": "float32",
                "rope_parameters": [
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10_000_000
                ]
            ]
            let config: [String: Any] = [
                "model_type": "qwen4_exp",
                "text_config": textConfig
            ]
            try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
                .write(to: URL(fileURLWithPath: configPath))

            let arch = try ArchInfo.load(configPath: configPath)

            #expect(arch.modelFamily == "qwen4_exp_text")
            #expect(arch.hiddenSize == 2560)
            #expect(arch.intermediateSize == 640)
            #expect(arch.moeIntermediateSize == 640)
            #expect(arch.numLayers == 48)
            #expect(arch.numExperts == 512)
            #expect(arch.topKExperts == 10)
            #expect(arch.linearNumValueHeads == 48)
            #expect(arch.fullAttentionLayerMask.count == 48)
            #expect(arch.fullAttentionLayerMask[3] == 1)
            #expect(arch.qwen38?.indexerHeads == 4)
            #expect(arch.qwen38?.hyperConnectionCount == 4)
            #expect(arch.qwen38?.pleLayerIDs == [2])
            #expect(arch.qwen38?.ngramVocabSizeBase == 20_000_000)
            #expect(arch.qwen38?.stateDType == "FP32")

            let plan = RepackPlan(
                arch: arch,
                baseMode: "affine",
                baseGroupSize: 32,
                bitsOverrideCount: 0,
                resident: ResidentFilePlan(
                    path: "model_weights.bin",
                    entries: [],
                    stringTable: [],
                    stringTableOffsets: [],
                    indexSize: 16_384,
                    residentSize: 0),
                layers: [],
                ngramShards: [],
                matchedModelID: nil,
                excludedMultimodalTensorNames: [])
            let zeroSHA = String(repeating: "0", count: 64)
            let manifestData = try GTurboJSON.encodeManifest(
                plan: plan,
                modelID: "Vontra/Qwen3.8-Flash-Next-MLX-4bit",
                sourceSnapshotHash: "de597762aa61387c89590a46582222a261ce0387",
                files: [
                    ("model_weights.bin", .init(size: 16_384, sha256: zeroSHA)),
                    ("packed_ngrams/layout.json", .init(size: 1, sha256: zeroSHA)),
                ],
                expertsPerLayer: 512,
                numLayers: 48,
                expertStride: 16_384,
                bitWidths: GTurboJSON.QuantBitWidths(
                    embedding: 4,
                    attention: 4,
                    router: 8,
                    sharedExpert: 4,
                    routedExpert: 4))
            let manifest = try GTurboManifestV3Codec.decode(manifestData)
            #expect(manifest.versionMajor == 3)
            #expect(manifest.arch.modelFamily == "qwen4_exp_text")
            #expect(manifest.arch.sparseAttention.indexerBudget == 2_048)
            #expect(manifest.arch.hyperConnection.lowRankSize == 320)
            #expect(manifest.arch.ple.layoutFile == "packed_ngrams/layout.json")

            guard rawModelFamily == "qwen4_exp_text" else { continue }
            let rows: UInt64 = 2_500_012
            let ngramTensors = (0..<128).flatMap { shard -> [SourceTensor] in
                let base = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard)"
                return [
                    SourceTensor(
                        name: "\(base).weight", shardPath: "source.safetensors",
                        dtype: .u32, shape: [rows, 20],
                        absoluteOffset: 0, sizeBytes: rows * 20 * 4),
                    SourceTensor(
                        name: "\(base).scales", shardPath: "source.safetensors",
                        dtype: .bf16, shape: [rows, 5],
                        absoluteOffset: 0, sizeBytes: rows * 5 * 2),
                    SourceTensor(
                        name: "\(base).biases", shardPath: "source.safetensors",
                        dtype: .bf16, shape: [rows, 5],
                        absoluteOffset: 0, sizeBytes: rows * 5 * 2),
                ]
            }
            let metadata = IndexLoader.SourceMetadata(
                indexPath: "model.safetensors.index.json",
                configPath: "config.json",
                indexSha256Hex: String(repeating: "0", count: 64),
                weightMap: Dictionary(uniqueKeysWithValues: ngramTensors.map {
                    ($0.name, $0.shardPath)
                }),
                baseBits: 4, baseGroupSize: 32, baseMode: "affine",
                bitsOverrides: [:], shardFilenames: ["source.safetensors"])
            let output = temporaryRoot("qwen38-ngram-plan")
            defer { try? FileManager.default.removeItem(atPath: output) }
            let completeHeader = Safetensors.Header(
                path: "source.safetensors", payloadBaseOffset: 0,
                tensors: ngramTensors)

            let ngramPlan = try RepackPlanner.plan(
                meta: metadata, arch: arch,
                shardHeaders: [completeHeader], outputDir: output)

            #expect(ngramPlan.ngramShards.count == 128)
            #expect(!ngramPlan.resident.entries.contains {
                $0.name.contains(".ngram_embedding.shard_")
            })
            #expect(ngramPlan.ngramShards.allSatisfy {
                $0.scalesOffset % GTurboFormatV1.alignmentBytes == 0
                    && $0.biasesOffset % GTurboFormatV1.alignmentBytes == 0
                    && $0.fileSize % GTurboFormatV1.alignmentBytes == 0
            })
            let incompleteHeader = Safetensors.Header(
                path: "source.safetensors", payloadBaseOffset: 0,
                tensors: Array(ngramTensors.dropLast()))
            #expect(throws: RepackError.self) {
                try RepackPlanner.plan(
                    meta: metadata, arch: arch,
                    shardHeaders: [incompleteHeader], outputDir: output)
            }
        }
    }

    @Test func qwenExpertNamesUseTextOnlyLayout() {
        #expect(RepackPlanner.classify(
            "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight",
            numLayers: 40,
            modelFamily: "qwen3_5_moe_text") == .routedExpert(role: "gate", layer: 3))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.3.mlp.switch_mlp.up_proj.scales",
            numLayers: 40,
            modelFamily: "qwen3_5_moe_text") == .routedExpert(role: "up", layer: 3))
        #expect(RepackPlanner.classify(
            "language_model.mtp.layers.0.embed_tokens.weight",
            numLayers: 40,
            modelFamily: "qwen3_5_moe_text") == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "vision_tower.encoder.layers.0.weight",
            numLayers: 40,
            modelFamily: "qwen3_5_moe_text") == .excludedMultimodal)
    }

    @Test func qwen38ExpertNamesUseTextOnlyLayout() {
        #expect(RepackPlanner.classify(
            "language_model.model.layers.47.mlp.switch_mlp.down_proj.biases",
            numLayers: 48,
            modelFamily: "qwen4_exp_text") == .routedExpert(role: "down", layer: 47))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text") == .ngramShard(shard: 0))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text") == .ngramShard(shard: 127))
        #expect(RepackPlanner.classify(
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text") == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "vision_tower.blocks.0.attn.qkv.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text") == .excludedMultimodal)
    }

    @Test func qwenSyntheticSnapshotPlansResidentAndExpertFiles() throws {
        let snapshotDirectory = temporaryRoot("qwen-snapshot")
        let output = temporaryRoot("qwen-output")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: output)
        }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0x5157_36,
            modelFamily: "qwen3_5_moe_text")
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let arch = try ArchInfo.load(
            configPath: (snapshotDirectory as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: output)

        #expect(arch.modelFamily == "qwen3_5_moe_text")
        #expect(arch.linearNumValueHeads == 32)
        #expect(arch.linearKeyHeadDim == 128)
        #expect(plan.layers.count == 2)
        #expect(plan.layers.allSatisfy { $0.expertsPerLayer == 2 })
        #expect(plan.resident.entries.contains { $0.name == "language_model.lm_head.weight" })
        let residentNames = Set(plan.resident.entries.map(\.name))
        let requiredQwenNames = [
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_z.weight",
            "language_model.model.layers.0.linear_attn.in_proj_b.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.0.linear_attn.A_log",
            "language_model.model.layers.0.linear_attn.dt_bias",
            "language_model.model.layers.0.linear_attn.norm.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            "language_model.model.layers.1.self_attn.q_proj.weight",
            "language_model.model.layers.1.self_attn.k_proj.weight",
            "language_model.model.layers.1.self_attn.v_proj.weight",
            "language_model.model.layers.1.self_attn.o_proj.weight",
            "language_model.model.layers.0.shared_expert_gate.weight",
            "language_model.model.layers.1.shared_expert_gate.weight"
        ]
        #expect(requiredQwenNames.allSatisfy { residentNames.contains($0) })
        #expect(!plan.resident.entries.contains {
            $0.name.contains("mtp") || $0.name.contains("vision")
        })
        let manifestData = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "mlx-community/Qwen3.6-35B-A3B-4bit",
            sourceSnapshotHash: "sha256:test",
            files: [],
            expertsPerLayer: 2,
            numLayers: 2,
            expertStride: plan.layers[0].expertStride,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4,
                attention: 4,
                router: 8,
                sharedExpert: 8,
                routedExpert: 4))
        let manifestRoot = try #require(
            JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        #expect(manifestRoot["versionMajor"] as? Int == 2)
        #expect((manifestRoot["arch"] as? [String: Any])?["modelFamily"] as? String
                == "qwen3_5_moe_text")
    }

    @Test func canonicalFingerprintDoesNotDependOnAbsoluteOutputRoot() throws {
        let snapshotDirectory = temporaryRoot("snapshot")
        let firstOutput = temporaryRoot("first")
        let secondOutput = temporaryRoot("second")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: firstOutput)
            try? FileManager.default.removeItem(atPath: secondOutput)
        }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0x1020_3040)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let arch = try ArchInfo.load(
            configPath: (snapshotDirectory as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let firstPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: firstOutput)
        let secondPlan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: secondOutput)

        let first = try RangeCopyPlanner.plan(
            repackPlan: firstPlan,
            rangeChunkBytes: 4096)
        let second = try RangeCopyPlanner.plan(
            repackPlan: secondPlan,
            rangeChunkBytes: 4096)

        #expect(first.canonicalFingerprint == second.canonicalFingerprint)
        #expect(first.coalescedCopies.map(\.id) == second.coalescedCopies.map(\.id))
    }

    @Test func overlappingDestinationIntervalsAreRejected() throws {
        let root = temporaryRoot("overlap")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("file.bin")
        let copies = [
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 0,
                size: 10,
                destinationPath: output,
                destinationOffset: 0),
            RangeCopy(
                shardID: "source.bin",
                sourceOffset: 20,
                size: 10,
                destinationPath: output,
                destinationOffset: 9),
        ]

        #expect(throws: RepackError.self) {
            try RangeCopyPlanner.validateDestinationIntervals(
                copies,
                outputRoot: root)
        }
    }

    @Test func normalizedRelativePathRejectsEscape() throws {
        let root = temporaryRoot("escape")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let outside = (root as NSString).deletingLastPathComponent
            + "/outside.bin"

        #expect(throws: RepackError.self) {
            _ = try RangeCopyPlanner.normalizedRelativePath(
                outside,
                root: root)
        }
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-range-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
    @Test func visionPlanIsBoundToTextManifestButNotAbsoluteOutputRoot() throws {
        let firstRoot = temporaryRoot("vision-first")
        let secondRoot = temporaryRoot("vision-second")
        defer {
            try? FileManager.default.removeItem(atPath: firstRoot)
            try? FileManager.default.removeItem(atPath: secondRoot)
        }
        let source = SourceTensor(
            name: "vision.weight",
            shardPath: "model-00001.safetensors",
            dtype: .bf16,
            shape: [2],
            absoluteOffset: 128,
            sizeBytes: 4)
        let plan = VisionPackPlan(
            entries: [.init(
                source: source,
                executionPosition: 0,
                fileOffset: 0,
                quantSpec: nil,
                groupSize: 64)],
            weightsFileSize: 16_384,
            sourcePayloadBytes: 4)
        let binding = String(repeating: "a", count: 64)
        let first = try RangeCopyPlanner.plan(
            visionPackPlan: plan,
            outputDirectory: firstRoot,
            rangeChunkBytes: 4096,
            textManifestSha256: binding)
        let second = try RangeCopyPlanner.plan(
            visionPackPlan: plan,
            outputDirectory: secondRoot,
            rangeChunkBytes: 4096,
            textManifestSha256: binding)
        let changedBinding = try RangeCopyPlanner.plan(
            visionPackPlan: plan,
            outputDirectory: secondRoot,
            rangeChunkBytes: 4096,
            textManifestSha256: String(repeating: "b", count: 64))

        #expect(first.canonicalFingerprint == second.canonicalFingerprint)
        #expect(first.canonicalFingerprint != changedBinding.canonicalFingerprint)
        #expect(first.remoteBytesToDownload == 4)
        #expect(first.expectedOutputs == [RemoteExpectedOutput(
            relativePath: "vision_weights.bin",
            size: 16_384)])
    }

}
