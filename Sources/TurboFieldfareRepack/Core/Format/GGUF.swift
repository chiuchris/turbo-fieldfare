import Foundation

struct GGUFTensor: Sendable, Hashable {
    let name: String
    let shape: [UInt64]
    let type: GGUFTensorType
    let offset: UInt64
    let byteCount: UInt64
    let absoluteOffset: UInt64
}

enum GGUFTensorType: UInt32, Sendable, Hashable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q2K = 10
    case q3K = 11
    case q4K = 12
    case q5K = 13
    case q6K = 14
    case q8K = 15
    case bf16 = 30

    var blockElements: UInt64 {
        switch self {
        case .f32, .f16, .bf16: 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0: 32
        case .q2K, .q3K, .q4K, .q5K, .q6K, .q8K: 256
        }
    }

    var blockBytes: UInt64 {
        switch self {
        case .f32: 4
        case .f16, .bf16: 2
        case .q4_0: 18
        case .q4_1: 20
        case .q5_0: 22
        case .q5_1: 24
        case .q8_0: 34
        case .q2K: 84
        case .q3K: 110
        case .q4K: 144
        case .q5K: 176
        case .q6K: 210
        case .q8K: 292
        }
    }
}

struct GGUFDocument: Sendable {
    let architecture: String
    let alignment: UInt64
    let dataOffset: UInt64
    let tensors: [GGUFTensor]
    let fileSize: UInt64

    static func load(from url: URL,
                     maxHeaderBytes: Int = 64 * 1024 * 1024) throws -> GGUFDocument {
        guard maxHeaderBytes > 0 else { throw GGUFError.limitExceeded("header") }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attributes[.size] as? NSNumber,
              number.int64Value >= 0 else {
            throw GGUFError.invalidFileSize
        }
        let size = number.uint64Value
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let header = try handle.read(upToCount: maxHeaderBytes) ?? Data()
        return try parse(header, fileSize: size)
    }

    static func parse(_ data: Data, fileSize: UInt64? = nil) throws -> GGUFDocument {
        let sourceSize = fileSize ?? UInt64(data.count)
        guard UInt64(data.count) <= sourceSize else { throw GGUFError.invalidFileSize }
        var reader = GGUFByteReader(data: data)
        guard try reader.readBytes(count: 4) == Data("GGUF".utf8) else {
            throw GGUFError.invalidMagic
        }
        guard try reader.readUInt32() == 3 else { throw GGUFError.unsupportedVersion }
        let tensorCount = try reader.readCount(limit: 1_000_000)
        let metadataCount = try reader.readCount(limit: 1_000_000)
        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(metadataCount)
        for _ in 0..<metadataCount {
            let key = try reader.readString()
            guard metadata[key] == nil else {
                throw GGUFError.invalidMetadata("duplicate key \(key)")
            }
            metadata[key] = try reader.readValue()
        }

        guard let architecture = metadata["general.architecture"]?.stringValue,
              ["qwen35moe", "qwen3moe", "qwen3_5_moe"].contains(architecture) else {
            throw GGUFError.incompatibleArchitecture(
                metadata["general.architecture"]?.stringValue ?? "missing")
        }
        let alignment = metadata["general.alignment"]?.positiveUInt64 ?? 32
        guard alignment <= 1 << 20, alignment & (alignment - 1) == 0 else {
            throw GGUFError.invalidMetadata("general.alignment must be a power of two")
        }

        var tensors: [GGUFTensor] = []
        tensors.reserveCapacity(tensorCount)
        var names = Set<String>()
        for _ in 0..<tensorCount {
            let name = try reader.readString()
            guard names.insert(name).inserted else {
                throw GGUFError.invalidTensor("duplicate tensor \(name)")
            }
            let dimensionCount = try reader.readUInt32()
            guard dimensionCount <= 8 else {
                throw GGUFError.limitExceeded("dimension count \(dimensionCount)")
            }
            var shape: [UInt64] = []
            shape.reserveCapacity(Int(dimensionCount))
            for _ in 0..<dimensionCount {
                let dimension = try reader.readUInt64()
                guard dimension > 0 else {
                    throw GGUFError.invalidTensor("zero dimension for \(name)")
                }
                shape.append(dimension)
            }
            guard let type = GGUFTensorType(rawValue: try reader.readUInt32()) else {
                throw GGUFError.unsupportedTensorType(name: name)
            }
            let offset = try reader.readUInt64()
            let elementCount = try shape.reduce(UInt64(1)) { partial, dimension in
                let (value, overflow) = partial.multipliedReportingOverflow(by: dimension)
                guard !overflow else { throw GGUFError.tensorShapeOverflow(name) }
                return value
            }
            guard elementCount % type.blockElements == 0 else {
                throw GGUFError.invalidTensor("shape is not block-aligned for \(name)")
            }
            let byteCount = try (elementCount / type.blockElements)
                .multipliedReportingChecked(type.blockBytes,
                                            error: GGUFError.tensorShapeOverflow(name))
            tensors.append(GGUFTensor(name: name, shape: shape, type: type,
                                      offset: offset, byteCount: byteCount,
                                      absoluteOffset: 0))
        }

        let dataOffset = try alignedOffset(UInt64(reader.offset), alignment: alignment)
        guard dataOffset <= sourceSize else { throw GGUFError.invalidDataOffset(dataOffset) }
        var resolved: [GGUFTensor] = []
        resolved.reserveCapacity(tensors.count)
        var ranges: [(UInt64, UInt64, String)] = []
        for tensor in tensors {
            let absolute = try dataOffset.addingReportingChecked(
                tensor.offset, error: GGUFError.tensorRangeOutOfBounds(name: tensor.name))
            let end = try absolute.addingReportingChecked(
                tensor.byteCount, error: GGUFError.tensorRangeOutOfBounds(name: tensor.name))
            guard end <= sourceSize else {
                throw GGUFError.tensorRangeOutOfBounds(name: tensor.name)
            }
            resolved.append(GGUFTensor(name: tensor.name, shape: tensor.shape,
                                       type: tensor.type, offset: tensor.offset,
                                       byteCount: tensor.byteCount,
                                       absoluteOffset: absolute))
            ranges.append((absolute, end, tensor.name))
        }
        let sorted = ranges.sorted { $0.0 < $1.0 }
        for pair in zip(sorted, sorted.dropFirst()) where pair.1.0 < pair.0.1 {
            throw GGUFError.overlappingTensors(first: pair.0.2, second: pair.1.2)
        }
        return GGUFDocument(architecture: architecture, alignment: alignment,
                            dataOffset: dataOffset, tensors: resolved,
                            fileSize: sourceSize)
    }

    func readPayload(for tensor: GGUFTensor, from url: URL,
                     maxBytes: UInt64 = 1 << 32) throws -> Data {
        guard tensors.contains(tensor) else { throw GGUFError.unknownTensor(tensor.name) }
        guard tensor.byteCount <= maxBytes else {
            throw GGUFError.payloadTooLarge(name: tensor.name, bytes: tensor.byteCount)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: tensor.absoluteOffset)
        let data = try handle.read(upToCount: Int(tensor.byteCount)) ?? Data()
        guard UInt64(data.count) == tensor.byteCount else {
            throw GGUFError.payloadTruncated(name: tensor.name,
                                             expected: tensor.byteCount,
                                             actual: UInt64(data.count))
        }
        return data
    }
}

enum GGUFNameMapper {
    static func canonical(_ name: String) -> String? {
        if name == "token_embd.weight" {
            return "language_model.model.embed_tokens.weight"
        }
        if name == "output.weight" {
            return "language_model.lm_head.weight"
        }
        if name == "output_norm.weight" {
            return "language_model.model.norm.weight"
        }
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 4, parts[0] == "blk", let layer = Int(parts[1]) else {
            return nil
        }
        let prefix = "language_model.model.layers.\(layer)"
        switch parts.dropFirst(2).joined(separator: ".") {
        case "attn_norm.weight": return prefix + ".input_layernorm.weight"
        case "ffn_norm.weight": return prefix + ".post_attention_layernorm.weight"
        case "attn_q.weight": return prefix + ".self_attn.q_proj.weight"
        case "attn_k.weight": return prefix + ".self_attn.k_proj.weight"
        case "attn_v.weight": return prefix + ".self_attn.v_proj.weight"
        case "attn_output.weight": return prefix + ".self_attn.o_proj.weight"
        case "ffn_gate_inp.weight": return prefix + ".mlp.gate.weight"
        case "ffn_gate_exps.weight":
            return prefix + ".mlp.switch_mlp.gate_proj.weight"
        case "ffn_up_exps.weight":
            return prefix + ".mlp.switch_mlp.up_proj.weight"
        case "ffn_down_exps.weight":
            return prefix + ".mlp.switch_mlp.down_proj.weight"
        case "ffn_gate_shexp.weight":
            return prefix + ".mlp.shared_expert.gate_proj.weight"
        case "ffn_up_shexp.weight":
            return prefix + ".mlp.shared_expert.up_proj.weight"
        case "ffn_down_shexp.weight":
            return prefix + ".mlp.shared_expert.down_proj.weight"
        default: return nil
        }
    }
}

enum GGUFDecoder {
    static func decode(_ payload: Data, type: GGUFTensorType,
                       elementCount: Int) throws -> [Float] {
        guard elementCount >= 0 else { throw GGUFError.invalidTensor("negative count") }
        let expectedBlocks = (elementCount + Int(type.blockElements) - 1)
            / Int(type.blockElements)
        guard UInt64(payload.count) == UInt64(expectedBlocks) * type.blockBytes else {
            throw GGUFError.payloadSizeMismatch
        }
        switch type {
        case .f32: return decodeF32(payload, count: elementCount)
        case .f16: return decodeF16(payload, count: elementCount)
        case .bf16: return decodeBF16(payload, count: elementCount)
        case .q4_0: return decodeQ4_0(payload, count: elementCount)
        case .q8_0: return decodeQ8_0(payload, count: elementCount)
        case .q2K: return decodeQ2K(payload, count: elementCount)
        case .q4K: return decodeQ4K(payload, count: elementCount)
        default: throw GGUFError.unsupportedDecode(type: type)
        }
    }

    private static func decodeF32(_ data: Data, count: Int) -> [Float] {
        (0..<count).map { Float(bitPattern: data.u32(at: $0 * 4)) }
    }

    private static func decodeF16(_ data: Data, count: Int) -> [Float] {
        (0..<count).map { Float(Float16(bitPattern: data.u16(at: $0 * 2))) }
    }

    private static func decodeBF16(_ data: Data, count: Int) -> [Float] {
        (0..<count).map { Float(bitPattern: UInt32(data.u16(at: $0 * 2)) << 16) }
    }

    private static func decodeQ4_0(_ data: Data, count: Int) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(count)
        for block in 0..<(count / 32) {
            let base = block * 18
            let scale = Float(Float16(bitPattern: data.u16(at: base)))
            for index in 0..<16 {
                let packed = data[base + 2 + index]
                result.append(scale * Float(Int(packed & 0x0F) - 8))
            }
            for index in 0..<16 {
                let packed = data[base + 2 + index]
                result.append(scale * Float(Int(packed >> 4) - 8))
            }
        }
        return result
    }

    private static func decodeQ8_0(_ data: Data, count: Int) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(count)
        for block in 0..<(count / 32) {
            let base = block * 34
            let scale = Float(Float16(bitPattern: data.u16(at: base)))
            for index in 0..<32 {
                result.append(scale * Float(Int(Int8(bitPattern: data[base + 2 + index]))))
            }
        }
        return result
    }

    private static func decodeQ2K(_ data: Data, count: Int) -> [Float] {
        var result = Array(repeating: Float(0), count: count)
        for block in 0..<(count / 256) {
            let base = block * 84
            let scaleBase = base
            let quantBase = base + 16
            let d = Float(Float16(bitPattern: data.u16(at: base + 80)))
            let dMin = Float(Float16(bitPattern: data.u16(at: base + 82)))
            for group in 0..<16 {
                let scale = d * Float(data[scaleBase + group] & 0x0F)
                let minimum = dMin * Float(data[scaleBase + group] >> 4)
                let plane = group % 4
                let row = group / 4
                for index in 0..<16 {
                    let packed = data[quantBase + row * 16 + index]
                    let quant = Float((packed >> UInt8(plane * 2)) & 0x03)
                    result[block * 256 + group * 16 + index] = scale * quant - minimum
                }
            }
        }
        return result
    }

    private static func decodeQ4K(_ data: Data, count: Int) -> [Float] {
        var result = Array(repeating: Float(0), count: count)
        for block in 0..<(count / 256) {
            let base = block * 144
            let scale = Float(Float16(bitPattern: data.u16(at: base)))
            let minimumScale = Float(Float16(bitPattern: data.u16(at: base + 2)))
            let scalesBase = base + 4
            let quantBase = base + 16
            for group in 0..<8 {
                let packedScale: UInt8
                let packedMinimum: UInt8
                if group < 4 {
                    packedScale = data[scalesBase + group] & 0x3F
                    packedMinimum = data[scalesBase + group + 4] & 0x3F
                } else {
                    packedScale = (data[scalesBase + group + 4] & 0x0F)
                        | ((data[scalesBase + group - 4] >> 6) << 4)
                    packedMinimum = (data[scalesBase + group + 4] >> 4)
                        | ((data[scalesBase + group] >> 6) << 4)
                }
                let groupScale = scale * Float(packedScale)
                let groupMinimum = minimumScale * Float(packedMinimum)
                for index in 0..<16 {
                    let packed = data[quantBase + group * 16 + index]
                    result[block * 256 + group * 32 + index] =
                        groupScale * Float(packed & 0x0F) - groupMinimum
                    result[block * 256 + group * 32 + index + 16] =
                        groupScale * Float(packed >> 4) - groupMinimum
                }
            }
        }
        return result
    }
}

enum GGUFError: Error, CustomStringConvertible, Equatable {
    case invalidMagic
    case unsupportedVersion
    case truncated(Int)
    case limitExceeded(String)
    case invalidMetadata(String)
    case incompatibleArchitecture(String)
    case invalidTensor(String)
    case unsupportedTensorType(name: String)
    case unsupportedDecode(type: GGUFTensorType)
    case tensorShapeOverflow(String)
    case tensorRangeOutOfBounds(name: String)
    case overlappingTensors(first: String, second: String)
    case invalidDataOffset(UInt64)
    case invalidFileSize
    case unknownTensor(String)
    case payloadTooLarge(name: String, bytes: UInt64)
    case payloadTruncated(name: String, expected: UInt64, actual: UInt64)
    case payloadSizeMismatch

    var description: String {
        switch self {
        case .invalidMagic: return "invalid GGUF magic"
        case .unsupportedVersion: return "unsupported GGUF version"
        case .truncated(let offset): return "truncated GGUF at byte \(offset)"
        case .limitExceeded(let detail): return "GGUF limit exceeded: \(detail)"
        case .invalidMetadata(let detail): return "invalid GGUF metadata: \(detail)"
        case .incompatibleArchitecture(let value): return "unsupported GGUF architecture \(value)"
        case .invalidTensor(let detail): return "invalid GGUF tensor: \(detail)"
        case .unsupportedTensorType(let name): return "unsupported GGUF tensor type for \(name)"
        case .unsupportedDecode(let type): return "GGUF decoder does not support \(type)"
        case .tensorShapeOverflow(let name): return "GGUF tensor shape overflows for \(name)"
        case .tensorRangeOutOfBounds(let name): return "GGUF tensor range is out of bounds for \(name)"
        case .overlappingTensors(let first, let second): return "overlapping GGUF tensors: \(first), \(second)"
        case .invalidDataOffset(let offset): return "GGUF data offset \(offset) exceeds file size"
        case .invalidFileSize: return "invalid GGUF file size"
        case .unknownTensor(let name): return "unknown GGUF tensor \(name)"
        case .payloadTooLarge(let name, let bytes): return "GGUF payload too large for \(name): \(bytes)"
        case .payloadTruncated(let name, let expected, let actual): return "GGUF payload truncated for \(name): \(actual)/\(expected)"
        case .payloadSizeMismatch: return "GGUF payload size does not match tensor shape"
        }
    }
}

private enum GGUFValue {
    case unsigned(UInt64)
    case signed(Int64)
    case float(Double)
    case boolean(Bool)
    case string(String)
    case array([GGUFValue])

    var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    var positiveUInt64: UInt64? {
        switch self {
        case .unsigned(let value) where value > 0: return value
        case .signed(let value) where value > 0: return UInt64(value)
        default: return nil
        }
    }
}

private struct GGUFByteReader {
    let data: Data
    var offset = 0

    mutating func readBytes(count: Int) throws -> Data {
        guard count >= 0, offset <= data.count,
              count <= data.count - offset else {
            throw GGUFError.truncated(offset)
        }
        let result = data.subdata(in: offset..<(offset + count))
        offset += count
        return result
    }

    mutating func readUInt32() throws -> UInt32 {
        let bytes = try readBytes(count: 4)
        return bytes.u32(at: 0)
    }

    mutating func readUInt64() throws -> UInt64 {
        let bytes = try readBytes(count: 8)
        return bytes.u64(at: 0)
    }

    mutating func readCount(limit: Int) throws -> Int {
        let value = try readUInt64()
        guard value <= UInt64(limit) else {
            throw GGUFError.limitExceeded("count \(value) > \(limit)")
        }
        return Int(value)
    }

    mutating func readString() throws -> String {
        let count = try readUInt64()
        guard count <= 16 * 1024 * 1024, count <= UInt64(Int.max) else {
            throw GGUFError.limitExceeded("string length \(count)")
        }
        guard let string = String(data: try readBytes(count: Int(count)), encoding: .utf8) else {
            throw GGUFError.invalidMetadata("string is not UTF-8")
        }
        return string
    }

    mutating func readValue() throws -> GGUFValue {
        try readValue(type: try readUInt32())
    }

    mutating func readValue(type: UInt32) throws -> GGUFValue {
        switch type {
        case 0: return .unsigned(UInt64(try readBytes(count: 1)[0]))
        case 1: return .signed(Int64(Int8(bitPattern: try readBytes(count: 1)[0])))
        case 2: return .unsigned(UInt64(try readBytes(count: 2).u16(at: 0)))
        case 3: return .signed(Int64(Int16(bitPattern: try readBytes(count: 2).u16(at: 0))))
        case 4: return .unsigned(UInt64(try readUInt32()))
        case 5: return .signed(Int64(Int32(bitPattern: try readUInt32())))
        case 6: return .float(Double(Float(bitPattern: try readUInt32())))
        case 7: return .boolean(try readBytes(count: 1)[0] != 0)
        case 8: return .string(try readString())
        case 9:
            let elementType = try readUInt32()
            let count = try readCount(limit: 1_000_000)
            return .array(try (0..<count).map { _ in try readValue(type: elementType) })
        case 10: return .unsigned(try readUInt64())
        case 11: return .signed(Int64(bitPattern: try readUInt64()))
        case 12: return .float(Double(Float(bitPattern: try readUInt32())))
        default: throw GGUFError.invalidMetadata("unknown value type \(type)")
        }
    }
}

private func alignedOffset(_ offset: UInt64, alignment: UInt64) throws -> UInt64 {
    let remainder = offset % alignment
    guard remainder != 0 else { return offset }
    return try offset.addingReportingChecked(
        alignment - remainder, error: GGUFError.invalidDataOffset(offset))
}

private extension UInt64 {
    func addingReportingChecked(_ other: UInt64, error: GGUFError) throws -> UInt64 {
        let (value, overflow) = addingReportingOverflow(other)
        guard !overflow else { throw error }
        return value
    }

    func multipliedReportingChecked(_ other: UInt64, error: GGUFError) throws -> UInt64 {
        let (value, overflow) = multipliedReportingOverflow(by: other)
        guard !overflow else { throw error }
        return value
    }
}

private extension Data {
    func u16(at offset: Int) -> UInt16 {
        UInt16(self[offset]) | UInt16(self[offset + 1]) << 8
    }

    func u32(at offset: Int) -> UInt32 {
        UInt32(u16(at: offset)) | UInt32(u16(at: offset + 2)) << 16
    }

    func u64(at offset: Int) -> UInt64 {
        UInt64(u32(at: offset)) | UInt64(u32(at: offset + 4)) << 32
    }
}
