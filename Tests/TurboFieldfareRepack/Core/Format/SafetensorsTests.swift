import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct SafetensorsTests {
    @Test
    func parsesI64TensorHeader() throws {
        let header = try Self.header(
            dtype: "I64",
            shape: [3],
            dataOffsets: [0, 24])
        let parsed = try Safetensors.parseHeaderBytes(
            path: "fixture.safetensors",
            fileSize: UInt64(8 + header.count + 24),
            headerBytes: header)

        let tensor = try #require(parsed.tensors.first)
        #expect(tensor.dtype == .i64)
        #expect(tensor.shape == [3])
        #expect(tensor.sizeBytes == 24)
    }

    @Test
    func rejectsI64ShapeByteMismatch() throws {
        let header = try Self.header(
            dtype: "I64",
            shape: [3],
            dataOffsets: [0, 16])

        #expect(throws: Error.self) {
            try Safetensors.parseHeaderBytes(
                path: "fixture.safetensors",
                fileSize: UInt64(8 + header.count + 16),
                headerBytes: header)
        }
    }

    private static func header(
        dtype: String,
        shape: [Int],
        dataOffsets: [Int]
    ) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "ple_embedding.layer_multipliers": [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": dataOffsets,
            ],
        ])
    }
}
