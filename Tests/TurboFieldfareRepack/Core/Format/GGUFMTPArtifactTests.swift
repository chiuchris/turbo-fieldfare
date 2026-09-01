import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct GGUFMTPArtifactTests {
    @Test
    func discoversQwenMtpMetadataAndTensors() throws {
        let data = Fixture.valid()
        let artifact = try GGUFMTPArtifact.parse(data)

        #expect(artifact.architecture == "qwen4exp")
        #expect(artifact.nextnPredictLayers == 1)
        #expect(artifact.alignment == 64)
        #expect(artifact.dataOffset % 64 == 0)
        #expect(artifact.mtpTensors.map(\.name) == [
            "blk.48.nextn.input_norm.weight",
            "blk.48.nextn.output.weight",
        ])
        #expect(artifact.tensors[1].dimensions == [2, 2])
        #expect(artifact.tensors[1].offset == 4)
        #expect(artifact.tensors[1].byteCount == 16)
        #expect(artifact.tensors[1].absoluteOffset == artifact.dataOffset + 4)
    }

    @Test
    func computesScalarTensorByteCount() throws {
        let artifact = try GGUFMTPArtifact.parse(Fixture.valid())

        #expect(artifact.tensors[0].byteCount == 4)
        #expect(artifact.tensors[0].absoluteOffset == artifact.dataOffset)
    }

    @Test
    func loadsHeaderFromFileAndValidatesFullFileSize() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try Fixture.valid(payloadLength: 20).write(to: url)

        let artifact = try GGUFMTPArtifact.load(from: url)

        #expect(artifact.mtpTensors.count == 2)
        #expect(artifact.tensors[1].byteCount == 16)
    }

    @Test
    func rejectsHeaderThatExceedsConfiguredBound() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try Fixture.valid().write(to: url)

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.load(from: url, maxHeaderBytes: 16)
        }
    }

    @Test
    func readsExactMtpTensorPayload() throws {
        let original = Fixture.valid()
        let artifact = try GGUFMTPArtifact.parse(original)
        let expected = Data([0x11, 0x22, 0x33, 0x44])
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + expected.count), with: expected)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let payload = try artifact.readPayload(
            for: artifact.mtpTensors[0], from: url)

        #expect(payload == expected)
    }

    @Test
    func rejectsPayloadReadForNonMtpTensor() throws {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.0.attn_norm.weight", dimensions: [1], offset: 0),
            .init(name: "blk.48.nextn.input", dimensions: [1], offset: 4),
        ])
        let artifact = try GGUFMTPArtifact.parse(data)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        #expect(throws: GGUFMTPArtifactError.invalidTensor(
            "payload is not a validated MTP tensor")) {
            try artifact.readPayload(for: artifact.tensors[0], from: url)
        }
    }

    @Test
    func rejectsPayloadAboveAllocationCap() throws {
        let data = Fixture.valid()
        let artifact = try GGUFMTPArtifact.parse(data)
        let tensor = artifact.mtpTensors[0]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        #expect(throws: GGUFMTPArtifactError.tensorPayloadTooLarge(
            name: tensor.name, byteCount: 4)) {
            try artifact.readPayload(for: tensor, from: url, maxBytes: 3)
        }
    }

    @Test
    func rejectsShortPayloadReadAfterDirectoryValidation() throws {
        let original = Fixture.valid()
        let artifact = try GGUFMTPArtifact.parse(original)
        let truncated = Data(original.prefix(Int(artifact.dataOffset) + 3))
        let tensor = artifact.mtpTensors[0]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try truncated.write(to: url)

        #expect(throws: GGUFMTPArtifactError.tensorPayloadTruncated(
            name: tensor.name, expected: 4, actual: 3)) {
            try artifact.readPayload(for: tensor, from: url)
        }
    }

    @Test
    func convertsF32PayloadToNativeBF16() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.f32", dimensions: [2], offset: 0),
        ], payloadLength: 8)
        let artifact = try GGUFMTPArtifact.parse(original)
        let source = Data([
            0x00, 0x00, 0xC0, 0x3F,
            0x00, 0x00, 0x00, 0x40,
        ])
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeBF16Payload(
            for: artifact.mtpTensors[0], from: url)

        #expect(converted == Data([0xC0, 0x3F, 0x00, 0x40]))
    }

    @Test
    func convertsF16PayloadToNativeBF16() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.f16", dimensions: [2], type: 1, offset: 0),
        ], payloadLength: 4)
        let artifact = try GGUFMTPArtifact.parse(original)
        let source = Data([0x00, 0x3C, 0x00, 0xC0])
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeBF16Payload(
            for: artifact.mtpTensors[0], from: url)

        #expect(converted == Data([0x80, 0x3F, 0x00, 0xC0]))
    }

    @Test
    func preservesBF16Payload() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.bf16", dimensions: [2], type: 30, offset: 0),
        ], payloadLength: 4)
        let artifact = try GGUFMTPArtifact.parse(original)
        let expected = Data([0x80, 0x3F, 0x00, 0xC0])
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + expected.count), with: expected)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeBF16Payload(
            for: artifact.mtpTensors[0], from: url)

        #expect(converted == expected)
    }

    @Test
    func convertsQ4KPayloadToNativeBF16() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.q4_k", dimensions: [256],
                  type: 12, offset: 0),
        ], payloadLength: 144)
        let artifact = try GGUFMTPArtifact.parse(original)
        var source = Data([0x00, 0x3C, 0x00, 0x3C])
        source.append(contentsOf: [
            1, 2, 3, 4, 5, 6, 7, 8,
            0x99, 0xAA, 0xBB, 0xCC,
        ])
        source.append(contentsOf: repeatElement(UInt8(0x53), count: 128))
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeBF16Payload(
            for: artifact.mtpTensors[0], from: url)

        let expectedBytes: [[UInt8]] = [
            [0x00, 0xC0], [0x80, 0x40], [0x00, 0x40], [0x40, 0x41],
            [0x90, 0x41], [0x20, 0x42], [0xB0, 0x41], [0x40, 0x42],
        ]
        var expected = Data()
        for bytes in expectedBytes {
            for _ in 0..<32 {
                expected.append(contentsOf: bytes)
            }
        }
        #expect(converted.count == 512)
        #expect(converted == expected)
    }

    @Test
    func rejectsNonFiniteQ4KMetadata() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.q4_k", dimensions: [256],
                  type: 12, offset: 0),
        ], payloadLength: 144)
        let artifact = try GGUFMTPArtifact.parse(original)
        var source = Data(repeating: 0, count: 144)
        source[0] = 0x00
        source[1] = 0x7C
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        #expect(throws: Error.self) {
            try artifact.readNativeBF16Payload(
                for: artifact.mtpTensors[0], from: url)
        }
    }

    @Test
    func convertsScalarPayloadToNativeAffineInt4() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.affine", dimensions: [64], offset: 0),
        ], payloadLength: 256)
        let artifact = try GGUFMTPArtifact.parse(original)
        let source = Self.f32Payload((0..<64).map { Float($0 % 16) })
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeInt4AffinePayload(
            for: artifact.mtpTensors[0], from: url)

        let packedPattern: [UInt8] = [
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
        ]
        #expect(converted.elementCount == 64)
        #expect(converted.packedWeights == Data((0..<4).flatMap { _ in packedPattern }))
        #expect(converted.scales == Data([0x80, 0x3F]))
        #expect(converted.biases == Data([0x00, 0x00]))
    }

    @Test
    func preservesConstantGroupInNativeAffinePayload() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.constant", dimensions: [64], offset: 0),
        ], payloadLength: 256)
        let artifact = try GGUFMTPArtifact.parse(original)
        let source = Self.f32Payload([Float](repeating: 0.42, count: 64))
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeInt4AffinePayload(
            for: artifact.mtpTensors[0], from: url)

        #expect(converted.packedWeights == Data(repeating: 0, count: 32))
        #expect(converted.scales == Data([0x80, 0x3F]))
        #expect(converted.biases == Data([0xD7, 0x3E]))
    }

    @Test
    func convertsQ4KPayloadToNativeAffineInt4() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.q4_k", dimensions: [256],
                  type: 12, offset: 0),
        ], payloadLength: 144)
        let artifact = try GGUFMTPArtifact.parse(original)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try original.write(to: url)

        let converted = try artifact.readNativeInt4AffinePayload(
            for: artifact.mtpTensors[0], from: url)

        #expect(converted.elementCount == 256)
        #expect(converted.packedWeights == Data(repeating: 0, count: 128))
        #expect(converted.scales == Data([
            0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F,
        ]))
        #expect(converted.biases == Data(repeating: 0x00, count: 8))
    }

    @Test
    func rejectsNonGroupAlignedNativeAffinePayload() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.odd", dimensions: [63], offset: 0),
        ], payloadLength: 252)
        let artifact = try GGUFMTPArtifact.parse(original)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try original.write(to: url)

        #expect(throws: Error.self) {
            try artifact.readNativeInt4AffinePayload(
                for: artifact.mtpTensors[0], from: url)
        }
    }

    @Test
    func convertsAllMtpTensorsInDirectoryOrder() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.first", dimensions: [64], offset: 0),
            .init(name: "blk.48.nextn.second", dimensions: [64], offset: 256),
        ], payloadLength: 512)
        let artifact = try GGUFMTPArtifact.parse(original)
        let first = Self.f32Payload([Float](repeating: 1, count: 64))
        let second = Self.f32Payload([Float](repeating: 2, count: 64))
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + first.count), with: first)
        data.replaceSubrange(
            (start + first.count)..<(start + first.count + second.count),
            with: second)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let converted = try artifact.readNativeInt4AffinePayloads(from: url)

        #expect(converted.map(\.tensor.name) == [
            "blk.48.nextn.first", "blk.48.nextn.second",
        ])
        #expect(converted.map(\.payload.elementCount) == [64, 64])
        #expect(converted.map(\.payload.packedWeights) == [
            Data(repeating: 0, count: 32), Data(repeating: 0, count: 32),
        ])
        #expect(converted.map(\.payload.biases) == [
            Data([0x80, 0x3F]), Data([0x00, 0x40]),
        ])
    }

    @Test
    func rejectsBatchConversionWithoutMtpTensors() {
        let artifact = GGUFMTPArtifact(
            architecture: "qwen4exp", nextnPredictLayers: 1,
            alignment: 64, dataOffset: 64, tensors: [])

        #expect(throws: Error.self) {
            try artifact.readNativeInt4AffinePayloads(
                from: URL(fileURLWithPath: "/dev/null"))
        }
    }

    @Test
    func propagatesBatchConversionAlignmentFailure() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.odd", dimensions: [63], offset: 0),
        ], payloadLength: 252)
        let artifact = try GGUFMTPArtifact.parse(original)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try original.write(to: url)

        #expect(throws: Error.self) {
            try artifact.readNativeInt4AffinePayloads(from: url)
        }
    }

    @Test
    func loadsNativeMTPDraftBlockWithStableRoles() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.input_norm.weight", dimensions: [64], offset: 0),
            .init(name: "blk.48.nextn.output.weight", dimensions: [64], offset: 256),
        ], payloadLength: 512)
        let artifact = try GGUFMTPArtifact.parse(original)
        let first = Self.f32Payload([Float](repeating: 1, count: 64))
        let second = Self.f32Payload([Float](repeating: 2, count: 64))
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + first.count), with: first)
        data.replaceSubrange(
            (start + first.count)..<(start + first.count + second.count),
            with: second)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        let block = try artifact.readNativeMTPDraftBlock(from: url)

        #expect(block.predictLayers == 1)
        #expect(block.tensors.map(\.role) == [
            "input_norm.weight", "output.weight",
        ])
        #expect(block.tensors.map(\.tensor) == artifact.mtpTensors)
        #expect(block.tensors.map(\.payload.biases) == [
            Data([0x80, 0x3F]), Data([0x00, 0x40]),
        ])
    }

    @Test
    func rejectsMTPDraftBlockWithEmptyRole() throws {
        let original = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.", dimensions: [64], offset: 0),
        ], payloadLength: 256)
        let artifact = try GGUFMTPArtifact.parse(original)
        let source = Self.f32Payload([Float](repeating: 1, count: 64))
        var data = original
        let start = Int(artifact.dataOffset)
        data.replaceSubrange(start..<(start + source.count), with: source)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        #expect(throws: Error.self) {
            try artifact.readNativeMTPDraftBlock(from: url)
        }
    }

    @Test
    func computesQuantizedTensorByteCount() throws {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.q4_k", dimensions: [256],
                  type: 12, offset: 0),
        ], payloadLength: 144)

        let artifact = try GGUFMTPArtifact.parse(data)

        #expect(artifact.mtpTensors[0].byteCount == 144)
        #expect(artifact.mtpTensors[0].absoluteOffset == artifact.dataOffset)
    }

    @Test
    func rejectsUnsupportedTensorType() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.unknown", dimensions: [1],
                  type: 99, offset: 0),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsTensorShapeOverflow() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.overflow", dimensions: [UInt64.max, 2],
                  offset: 0),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsTensorRangeOverflow() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.overflow", dimensions: [1],
                  offset: UInt64.max),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsTensorPayloadOutsideFile() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.missing", dimensions: [4],
                  offset: 0),
        ], payloadLength: 4)

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsOverlappingTensorPayloads() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.48.nextn.first", dimensions: [2], offset: 0),
            .init(name: "blk.48.nextn.second", dimensions: [2], offset: 4),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsMissingNextnMetadata() {
        let data = Fixture.valid(metadata: [
            .string("general.architecture", "qwen4exp"),
            .uint32("general.alignment", 64),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsUnsupportedVersion() {
        let data = Fixture.valid(version: 2)

        #expect(throws: GGUFMTPArtifactError.unsupportedVersion) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsTruncatedInput() {
        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(Data("GGUF".utf8))
        }
    }

    @Test
    func rejectsWrongArchitecture() {
        let data = Fixture.valid(metadata: [
            .string("general.architecture", "llama"),
            .uint32("qwen4exp.nextn_predict_layers", 1),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsDirectoryWithoutMtpTensors() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.0.attn_norm.weight", dimensions: [2560], offset: 0),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    @Test
    func rejectsUnexpectedMtpLayerPrefix() {
        let data = Fixture.valid(tensors: [
            .init(name: "blk.47.nextn.input_norm.weight", dimensions: [2560], offset: 0),
        ])

        #expect(throws: Error.self) {
            try GGUFMTPArtifact.parse(data)
        }
    }

    private static func f32Payload(_ values: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(values.count * 4)
        for value in values {
            let bits = value.bitPattern
            for shift in stride(from: 0, through: 24, by: 8) {
                data.append(UInt8((bits >> UInt32(shift)) & 0xFF))
            }
        }
        return data
    }

    private struct Fixture {
        struct Tensor {
            let name: String
            let dimensions: [UInt64]
            let type: UInt32
            let offset: UInt64

            init(
                name: String,
                dimensions: [UInt64],
                type: UInt32 = 0,
                offset: UInt64
            ) {
                self.name = name
                self.dimensions = dimensions
                self.type = type
                self.offset = offset
            }
        }

        enum Metadata {
            case string(String, String)
            case uint32(String, UInt32)
            case strings(String, [String])
        }

        static func valid(
            version: UInt32 = 3,
            metadata: [Metadata]? = nil,
            tensors: [Tensor] = [
                .init(name: "blk.48.nextn.input_norm.weight",
                      dimensions: [1], offset: 0),
                .init(name: "blk.48.nextn.output.weight",
                      dimensions: [2, 2], offset: 4),
            ],
            payloadLength: Int = 128
        ) -> Data {
            let metadata = metadata ?? [
                .string("general.architecture", "qwen4exp"),
                .uint32("qwen4exp.nextn_predict_layers", 1),
                .uint32("general.alignment", 64),
                .strings("general.tags", ["qwen3.8", "mtp"]),
            ]
            var builder = Builder()
            builder.appendBytes(Data("GGUF".utf8))
            builder.appendUInt32(version)
            builder.appendUInt64(UInt64(tensors.count))
            builder.appendUInt64(UInt64(metadata.count))
            for entry in metadata {
                builder.appendMetadata(entry)
            }
            for tensor in tensors {
                builder.appendString(tensor.name)
                builder.appendUInt32(UInt32(tensor.dimensions.count))
                for dimension in tensor.dimensions {
                    builder.appendUInt64(dimension)
                }
                builder.appendUInt32(tensor.type)
                builder.appendUInt64(tensor.offset)
            }
            while builder.data.count % 64 != 0 {
                builder.data.append(0)
            }
            builder.data.append(Data(repeating: 0, count: payloadLength))
            return builder.data
        }
    }

    private struct Builder {
        var data = Data()

        mutating func appendMetadata(_ metadata: Fixture.Metadata) {
            switch metadata {
            case .string(let key, let value):
                appendString(key)
                appendUInt32(8)
                appendString(value)
            case .uint32(let key, let value):
                appendString(key)
                appendUInt32(4)
                appendUInt32(value)
            case .strings(let key, let values):
                appendString(key)
                appendUInt32(9)
                appendUInt32(8)
                appendUInt64(UInt64(values.count))
                for value in values {
                    appendString(value)
                }
            }
        }

        mutating func appendString(_ value: String) {
            let bytes = Data(value.utf8)
            appendUInt64(UInt64(bytes.count))
            appendBytes(bytes)
        }

        mutating func appendUInt32(_ value: UInt32) {
            for shift in stride(from: 0, through: 24, by: 8) {
                data.append(UInt8((value >> UInt32(shift)) & 0xff))
            }
        }

        mutating func appendUInt64(_ value: UInt64) {
            for shift in stride(from: 0, through: 56, by: 8) {
                data.append(UInt8((value >> UInt64(shift)) & 0xff))
            }
        }

        mutating func appendBytes(_ bytes: Data) {
            data.append(contentsOf: bytes)
        }
    }
}
