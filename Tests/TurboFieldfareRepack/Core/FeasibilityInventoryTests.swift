import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite(.serialized)
struct FeasibilityInventoryTests {
    @Test func scanClassifiesTensorRolesAndTotalsBytes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("f0a-inventory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let names = [
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
            "vision_tower.weight",
            "unclassified.weight"
        ]
        let index = ["weight_map": Dictionary(uniqueKeysWithValues: names.map {
            ($0, "model.safetensors")
        })]
        let config = ["quantization": ["bits": 4, "group_size": 64, "mode": "affine"]]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        try JSONSerialization.data(withJSONObject: config)
            .write(to: root.appendingPathComponent("config.json"))

        let header: [String: Any] = Dictionary(uniqueKeysWithValues: names.enumerated().map {
            ($1, ["dtype": "U32", "shape": [1], "data_offsets": [$0 * 4, ($0 + 1) * 4]])
        })
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var shard = Data()
        var headerSize = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerSize) { shard.append(contentsOf: $0) }
        shard.append(headerData)
        shard.append(Data(repeating: 0, count: names.count * 4))
        try shard.write(to: root.appendingPathComponent("model.safetensors"))

        let inventory = try FeasibilityInventory.scan(snapshotDirectory: root.path)

        #expect(inventory.tensors.count == 4)
        #expect(inventory.summary.textResidentBytes == 4)
        #expect(inventory.summary.routedExpertBytes == 4)
        #expect(inventory.summary.omittedMultimodalBytes == 4)
        #expect(inventory.summary.unsupportedBytes == 4)
        #expect(inventory.summary.totalBytes == 16)
    }
}