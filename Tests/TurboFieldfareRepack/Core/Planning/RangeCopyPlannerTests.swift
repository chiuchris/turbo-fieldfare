import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite
struct RangeCopyPlannerTests {
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
}
