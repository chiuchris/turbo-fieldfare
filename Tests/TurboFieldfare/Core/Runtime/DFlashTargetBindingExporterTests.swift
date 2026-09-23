import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite struct DFlashTargetBindingExporterTests {
    @Test func exportsPackedBindingsWithSafetensorsMetadataAndOffsets() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffer = try #require(device.makeBuffer(length: 32, options: .storageModeShared))
        for byte in 0..<32 {
            buffer.contents().storeBytes(of: UInt8(byte), toByteOffset: byte, as: UInt8.self)
        }

        let embedding = makeView(buffer: buffer, offset: 0)
        let lmHead = makeView(buffer: buffer, offset: 16)
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash-bindings-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: output) }

        try DFlashTargetBindingExporter.export(
            embedding: embedding,
            lmHead: lmHead,
            vocabSize: 2,
            hiddenSize: 4,
            to: output
        )

        let file = try Data(contentsOf: output)
        let headerLength = file.prefix(8).enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << (UInt64($1.offset) * 8))
        }
        let headerStart = 8
        let dataStart = headerStart + Int(headerLength)
        let headerData = file.subdata(in: headerStart..<dataStart)
        let header = try #require(
            JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        )
        let embeddingWeight = try #require(header["target.embed_tokens.weight"] as? [String: Any])
        let embeddingScales = try #require(header["target.embed_tokens.scales"] as? [String: Any])
        let lmHeadWeight = try #require(header["target.lm_head.weight"] as? [String: Any])
        let metadata = try #require(header["__metadata__"] as? [String: String])

        #expect(embeddingWeight["dtype"] as? String == "U8")
        #expect(embeddingWeight["shape"] as? [Int] == [8])
        #expect(embeddingScales["dtype"] as? String == "BF16")
        #expect(embeddingScales["shape"] as? [Int] == [2])
        #expect(lmHeadWeight["data_offsets"] as? [Int] == [16, 24])
        #expect(metadata["target.embed_tokens.logical_shape"] == "2,4")
        #expect(metadata["target.lm_head.quantization_bits"] == "4")
        #expect(metadata["target.lm_head.quantization_group_size"] == "64")
        #expect(file.subdata(in: dataStart..<file.count) == Data((0..<32).map(UInt8.init)))
    }

    @Test func rejectsBindingShapeThatDoesNotMatchTargetDimensions() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let buffer = try #require(device.makeBuffer(length: 32, options: .storageModeShared))
        let invalidEmbedding = makeView(buffer: buffer, offset: 0, vocabSize: 3)
        let lmHead = makeView(buffer: buffer, offset: 16)
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash-bindings-invalid-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: output) }

        #expect(throws: (any Error).self) {
            try DFlashTargetBindingExporter.export(
                embedding: invalidEmbedding,
                lmHead: lmHead,
                vocabSize: 2,
                hiddenSize: 4,
                to: output
            )
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    private func makeView(
        buffer: MTLBuffer,
        offset: UInt64,
        vocabSize: UInt32 = 2
    ) -> TensorView {
        TensorView(
            buffer: buffer,
            offset: offset,
            length: 8,
            scaleOffset: offset + 8,
            scaleLength: 4,
            biasOffset: offset + 12,
            biasLength: 4,
            shape: (vocabSize, 4, 1, 1),
            dtype: 0,
            quantization: TensorQuantizationDescriptor(bits: 4, groupSize: 64)
        )
    }
}
