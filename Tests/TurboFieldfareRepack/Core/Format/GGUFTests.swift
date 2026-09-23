import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct GGUFTests {
    @Test
    func parsesQwenMetadataAndTensorRanges() throws {
        let data = Fixture.file(
            tensors: [Fixture.Tensor(name: "blk.0.attn_q.weight", shape: [32], type: .q4_0)],
            payloads: [Data(repeating: 0, count: 18)])
        let document = try GGUFDocument.parse(data)

        #expect(document.architecture == "qwen35moe")
        #expect(document.alignment == 32)
        #expect(document.tensors.count == 1)
        #expect(document.tensors[0].absoluteOffset == document.dataOffset)
        #expect(document.tensors[0].byteCount == 18)
    }

    @Test
    func mapsQwenTensorNamesToRuntimeNames() {
        #expect(GGUFNameMapper.canonical("token_embd.weight") ==
                "language_model.model.embed_tokens.weight")
        #expect(GGUFNameMapper.canonical("output_norm.weight") ==
                "language_model.model.norm.weight")
        #expect(GGUFNameMapper.canonical("blk.7.ffn_gate_inp.weight") ==
                "language_model.model.layers.7.mlp.gate.weight")
        #expect(GGUFNameMapper.canonical("blk.7.ffn_up_shexp.weight") ==
                "language_model.model.layers.7.mlp.shared_expert.up_proj.weight")
        #expect(GGUFNameMapper.canonical("blk.7.unknown.weight") == nil)
    }

    @Test
    func decodesQ4AndQ2Blocks() throws {
        var q4 = Data()
        q4.append(contentsOf: [0, 60])
        q4.append(contentsOf: Data(repeating: 0x88, count: 16))
        let q4Values = try GGUFDecoder.decode(q4, type: .q4_0, elementCount: 32)
        #expect(q4Values.allSatisfy { $0 == 0 })

        var q2 = Data(repeating: 0, count: 84)
        q2[80] = 0
        q2[81] = 60
        let q2Values = try GGUFDecoder.decode(q2, type: .q2K, elementCount: 256)
        #expect(q2Values.allSatisfy { $0 == 0 })
        #expect(throws: GGUFError.payloadSizeMismatch) {
            try GGUFDecoder.decode(Data(repeating: 0, count: 1),
                                   type: .q4_0, elementCount: 32)
        }
    }

    @Test
    func rejectsOutOfBoundsTensor() {
        let data = Fixture.file(
            tensors: [Fixture.Tensor(name: "blk.0.attn_q.weight", shape: [32], type: .q4_0)],
            payloads: [])

        #expect(throws: GGUFError.self) {
            try GGUFDocument.parse(data)
        }
    }

    @Test
    func rejectsUnsupportedArchitecture() {
        let data = Fixture.file(
            architecture: "llama",
            tensors: [],
            payloads: [])

        #expect(throws: GGUFError.incompatibleArchitecture("llama")) {
            try GGUFDocument.parse(data)
        }
    }

    private enum Fixture {
        struct Tensor {
            let name: String
            let shape: [UInt64]
            let type: GGUFTensorType
        }

        static func file(architecture: String = "qwen35moe",
                         tensors: [Tensor],
                         payloads: [Data]) -> Data {
            var data = Data("GGUF".utf8)
            data.appendUInt32(3)
            data.appendUInt64(UInt64(tensors.count))
            data.appendUInt64(2)
            data.appendString("general.architecture")
            data.appendUInt32(8)
            data.appendString(architecture)
            data.appendString("general.alignment")
            data.appendUInt32(4)
            data.appendUInt32(32)
            for tensor in tensors {
                data.appendString(tensor.name)
                data.appendUInt32(UInt32(tensor.shape.count))
                for dimension in tensor.shape {
                    data.appendUInt64(dimension)
                }
                data.appendUInt32(tensor.type.rawValue)
                data.appendUInt64(0)
            }
            while data.count % 32 != 0 { data.append(0) }
            for payload in payloads { data.append(payload) }
            return data
        }
    }
}

private extension Data {
    mutating func appendUInt32(_ value: UInt32) {
        append(UInt8(truncatingIfNeeded: value))
        append(UInt8(truncatingIfNeeded: value >> 8))
        append(UInt8(truncatingIfNeeded: value >> 16))
        append(UInt8(truncatingIfNeeded: value >> 24))
    }

    mutating func appendUInt64(_ value: UInt64) {
        appendUInt32(UInt32(truncatingIfNeeded: value))
        appendUInt32(UInt32(truncatingIfNeeded: value >> 32))
    }

    mutating func appendString(_ value: String) {
        let bytes = Data(value.utf8)
        appendUInt64(UInt64(bytes.count))
        append(bytes)
    }
}
