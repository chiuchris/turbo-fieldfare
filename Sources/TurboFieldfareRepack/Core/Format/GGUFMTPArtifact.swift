import Foundation

struct GGUFMTPArtifact: Equatable, Sendable {
    static let defaultMaxHeaderBytes = 64 * 1024 * 1024
    static let defaultMaxPayloadBytes: UInt64 = 512 * 1024 * 1024
    private static let mtpTensorPrefix = "blk.48.nextn."

    struct Tensor: Equatable, Sendable {
        let name: String
        let dimensions: [UInt64]
        let type: UInt32
        let offset: UInt64
        let byteCount: UInt64?
        let absoluteOffset: UInt64?
    }

    struct NativeInt4AffinePayload: Equatable, Sendable {
        let packedWeights: Data
        let scales: Data
        let biases: Data
        let elementCount: Int
    }

    struct NativeInt4AffineTensor: Equatable, Sendable {
        let tensor: Tensor
        let payload: NativeInt4AffinePayload
    }

    struct NativeMTPDraftTensor: Equatable, Sendable {
        let role: String
        let tensor: Tensor
        let payload: NativeInt4AffinePayload
    }

    struct NativeMTPDraftBlock: Equatable, Sendable {
        let predictLayers: Int
        let tensors: [NativeMTPDraftTensor]
    }

    let architecture: String
    let nextnPredictLayers: Int
    let alignment: UInt64
    let dataOffset: UInt64
    let tensors: [Tensor]

    var mtpTensors: [Tensor] {
        tensors.filter { $0.name.hasPrefix(Self.mtpTensorPrefix) }
    }

    func readPayload(
        for tensor: Tensor,
        from url: URL,
        maxBytes: UInt64 = defaultMaxPayloadBytes
    ) throws -> Data {
        guard let validatedTensor = tensors.first(where: { $0 == tensor }),
              let byteCount = validatedTensor.byteCount,
              let absoluteOffset = validatedTensor.absoluteOffset else {
            throw GGUFMTPArtifactError.invalidTensor(
                "payload is not a validated MTP tensor")
        }
        guard maxBytes > 0, byteCount <= maxBytes else {
            throw GGUFMTPArtifactError.tensorPayloadTooLarge(
                name: validatedTensor.name, byteCount: byteCount)
        }
        guard byteCount <= UInt64(Int.max) else {
            throw GGUFMTPArtifactError.tensorPayloadTooLarge(
                name: validatedTensor.name, byteCount: byteCount)
        }

        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: absoluteOffset)
            let data = try handle.read(upToCount: Int(byteCount)) ?? Data()
            guard UInt64(data.count) == byteCount else {
                throw GGUFMTPArtifactError.tensorPayloadTruncated(
                    name: validatedTensor.name,
                    expected: byteCount,
                    actual: UInt64(data.count))
            }
            return data
        } catch let error as GGUFMTPArtifactError {
            throw error
        } catch {
            throw GGUFMTPArtifactError.fileReadFailed(
                name: validatedTensor.name)
        }
    }

    func readNativeBF16Payload(
        for tensor: Tensor,
        from url: URL,
        maxBytes: UInt64 = defaultMaxPayloadBytes
    ) throws -> Data {
        let payload = try readPayload(for: tensor, from: url, maxBytes: maxBytes)
        guard let validatedTensor = tensors.first(where: { $0 == tensor }) else {
            throw GGUFMTPArtifactError.invalidTensor(
                "payload is not a validated MTP tensor")
        }

        func appendBF16(_ bits: UInt16, to data: inout Data) {
            data.append(UInt8(truncatingIfNeeded: bits))
            data.append(UInt8(truncatingIfNeeded: bits >> 8))
        }

        func bf16Bits(fromFloatBits bits: UInt32) -> UInt16 {
            let lsb = (bits >> 16) & 1
            let roundingBias = 0x7FFF &+ lsb
            let rounded = bits &+ roundingBias
            return UInt16(truncatingIfNeeded: rounded >> 16)
        }

        switch validatedTensor.type {
        case 30:
            return payload
        case 1:
            guard payload.count % 2 == 0 else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "F16 payload width is invalid for \(validatedTensor.name)")
            }
            var converted = Data()
            converted.reserveCapacity(payload.count)
            for index in stride(from: 0, to: payload.count, by: 2) {
                let bits = UInt16(payload[index])
                    | UInt16(payload[index + 1]) << 8
                let value = Float(Float16(bitPattern: bits))
                appendBF16(bf16Bits(fromFloatBits: value.bitPattern), to: &converted)
            }
            return converted
        case 0:
            guard payload.count % 4 == 0 else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "F32 payload width is invalid for \(validatedTensor.name)")
            }
            var converted = Data()
            converted.reserveCapacity(payload.count / 2)
            for index in stride(from: 0, to: payload.count, by: 4) {
                let bits = UInt32(payload[index])
                    | UInt32(payload[index + 1]) << 8
                    | UInt32(payload[index + 2]) << 16
                    | UInt32(payload[index + 3]) << 24
                appendBF16(bf16Bits(fromFloatBits: bits), to: &converted)
            }
            return converted
        case 12:
            let blockByteCount = 144
            let blockElementCount = 256
            guard let expectedByteCount = validatedTensor.byteCount,
                  UInt64(payload.count) == expectedByteCount,
                  payload.count % blockByteCount == 0 else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "Q4_K payload width is invalid for \(validatedTensor.name)")
            }
            var converted = Data()
            converted.reserveCapacity(
                payload.count / blockByteCount * blockElementCount * 2)
            for blockOffset in stride(from: 0, to: payload.count, by: blockByteCount) {
                let dBits = UInt16(payload[blockOffset])
                    | UInt16(payload[blockOffset + 1]) << 8
                let dminBits = UInt16(payload[blockOffset + 2])
                    | UInt16(payload[blockOffset + 3]) << 8
                let d = Float(Float16(bitPattern: dBits))
                let dmin = Float(Float16(bitPattern: dminBits))
                guard d.isFinite, dmin.isFinite else {
                    throw GGUFMTPArtifactError.invalidTensor(
                        "Q4_K scale metadata is non-finite for \(validatedTensor.name)")
                }

                let scalesOffset = blockOffset + 4
                let quantizedOffset = blockOffset + 16
                for group in 0..<8 {
                    let scaleCode: UInt8
                    let minimumCode: UInt8
                    if group < 4 {
                        scaleCode = payload[scalesOffset + group] & 0x3F
                        minimumCode = payload[scalesOffset + group + 4] & 0x3F
                    } else {
                        let packed = payload[scalesOffset + group + 4]
                        scaleCode = (packed & 0x0F)
                            | ((payload[scalesOffset + group - 4] >> 6) << 4)
                        minimumCode = (packed >> 4)
                            | ((payload[scalesOffset + group] >> 6) << 4)
                    }
                    let scale = d * Float(scaleCode)
                    let minimum = dmin * Float(minimumCode)
                    let pairOffset = (group / 2) * 32
                    let highNibble = group % 2 == 1
                    for value in 0..<32 {
                        let packed = payload[quantizedOffset + pairOffset + value]
                        let code = highNibble ? packed >> 4 : packed & 0x0F
                        appendBF16(
                            bf16Bits(fromFloatBits: ((scale * Float(code)) - minimum).bitPattern),
                            to: &converted)
                    }
                }
            }
            return converted
        default:
            throw GGUFMTPArtifactError.invalidTensor(
                "native BF16 conversion does not support type \(validatedTensor.type) "
                    + "for \(validatedTensor.name)")
        }
    }

    func readNativeInt4AffinePayload(
        for tensor: Tensor,
        from url: URL,
        maxBytes: UInt64 = defaultMaxPayloadBytes
    ) throws -> NativeInt4AffinePayload {
        let bf16Payload = try readNativeBF16Payload(
            for: tensor, from: url, maxBytes: maxBytes)
        guard let validatedTensor = tensors.first(where: { $0 == tensor }) else {
            throw GGUFMTPArtifactError.invalidTensor(
                "payload is not a validated MTP tensor")
        }
        var elementCount: UInt64 = 1
        for dimension in validatedTensor.dimensions {
            let (product, overflow) = elementCount.multipliedReportingOverflow(
                by: dimension)
            guard !overflow else {
                throw GGUFMTPArtifactError.tensorShapeOverflow(
                    validatedTensor.name)
            }
            elementCount = product
        }
        guard elementCount <= UInt64(Int.max),
              elementCount % 64 == 0,
              elementCount <= UInt64(Int.max / 2),
              UInt64(bf16Payload.count) == elementCount * 2 else {
            throw GGUFMTPArtifactError.invalidTensor(
                "native affine payload requires 64-value groups for "
                    + validatedTensor.name)
        }

        let count = Int(elementCount)
        let groupCount = count / 64
        var packedWeights = Data(repeating: 0, count: count / 2)
        var scales = Data()
        var biases = Data()
        scales.reserveCapacity(groupCount * 2)
        biases.reserveCapacity(groupCount * 2)

        func bf16Bits(at index: Int) -> UInt16 {
            UInt16(bf16Payload[index * 2])
                | UInt16(bf16Payload[index * 2 + 1]) << 8
        }

        func bf16ToFloat(_ bits: UInt16) -> Float {
            Float(bitPattern: UInt32(bits) << 16)
        }

        func appendBF16(_ bits: UInt16, to data: inout Data) {
            data.append(UInt8(truncatingIfNeeded: bits))
            data.append(UInt8(truncatingIfNeeded: bits >> 8))
        }

        func encodeBF16(_ value: Float) -> UInt16 {
            let bits = value.bitPattern
            let lsb = (bits >> 16) & 1
            let rounded = bits &+ (0x7FFF &+ lsb)
            return UInt16(truncatingIfNeeded: rounded >> 16)
        }

        for group in 0..<groupCount {
            let start = group * 64
            var minimum = Float.infinity
            var maximum = -Float.infinity
            for index in start..<(start + 64) {
                let value = bf16ToFloat(bf16Bits(at: index))
                guard value.isFinite else {
                    throw GGUFMTPArtifactError.invalidTensor(
                        "native affine payload contains non-finite value for "
                            + validatedTensor.name)
                }
                minimum = min(minimum, value)
                maximum = max(maximum, value)
            }
            let scaleValue = minimum == maximum
                ? Float(1) : (maximum - minimum) / 15
            let scaleBits = encodeBF16(scaleValue)
            let biasBits = encodeBF16(minimum)
            appendBF16(scaleBits, to: &scales)
            appendBF16(biasBits, to: &biases)
            let scale = bf16ToFloat(scaleBits)
            let bias = bf16ToFloat(biasBits)
            let inverseScale = scale == 0 ? Float(0) : 1 / scale

            for offset in 0..<64 {
                let value = bf16ToFloat(bf16Bits(at: start + offset))
                var code = Int(((value - bias) * inverseScale).rounded())
                code = max(0, min(15, code))
                let byteIndex = start / 2 + offset / 2
                let nibble = UInt8(code) & 0x0F
                if offset % 2 == 0 {
                    packedWeights[byteIndex] =
                        (packedWeights[byteIndex] & 0xF0) | nibble
                } else {
                    packedWeights[byteIndex] =
                        (packedWeights[byteIndex] & 0x0F) | (nibble << 4)
                }
            }
        }

        return NativeInt4AffinePayload(
            packedWeights: packedWeights,
            scales: scales,
            biases: biases,
            elementCount: count)
    }

    func readNativeInt4AffinePayloads(
        from url: URL,
        maxBytes: UInt64 = defaultMaxPayloadBytes
    ) throws -> [NativeInt4AffineTensor] {
        let tensors = mtpTensors
        guard !tensors.isEmpty else {
            throw GGUFMTPArtifactError.missingMtpTensors(Self.mtpTensorPrefix)
        }
        return try tensors.map { tensor in
            NativeInt4AffineTensor(
                tensor: tensor,
                payload: try readNativeInt4AffinePayload(
                    for: tensor, from: url, maxBytes: maxBytes))
        }
    }

    func readNativeMTPDraftBlock(
        from url: URL,
        maxBytes: UInt64 = defaultMaxPayloadBytes
    ) throws -> NativeMTPDraftBlock {
        let converted = try readNativeInt4AffinePayloads(
            from: url, maxBytes: maxBytes)
        var roles = Set<String>()
        let tensors = try converted.map { item in
            let prefix = Self.mtpTensorPrefix
            guard item.tensor.name.hasPrefix(prefix) else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "unexpected MTP tensor \(item.tensor.name)")
            }
            let role = String(item.tensor.name.dropFirst(prefix.count))
            guard !role.isEmpty, roles.insert(role).inserted else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "invalid MTP tensor role \(role)")
            }
            return NativeMTPDraftTensor(
                role: role, tensor: item.tensor, payload: item.payload)
        }
        return NativeMTPDraftBlock(
            predictLayers: nextnPredictLayers, tensors: tensors)
    }

    static func load(
        from url: URL,
        maxHeaderBytes: Int = defaultMaxHeaderBytes
    ) throws -> GGUFMTPArtifact {
        guard maxHeaderBytes > 0 else {
            throw GGUFMTPArtifactError.limitExceeded(
                "header length must be positive")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attributes[.size] as? NSNumber,
              size.int64Value >= 0 else {
            throw GGUFMTPArtifactError.invalidFileSize
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let header = try handle.read(upToCount: maxHeaderBytes) ?? Data()
        return try parse(header, fileSize: size.uint64Value)
    }

    static func parse(
        _ data: Data,
        fileSize: UInt64? = nil
    ) throws -> GGUFMTPArtifact {
        let sourceFileSize = fileSize ?? UInt64(data.count)
        guard UInt64(data.count) <= sourceFileSize else {
            throw GGUFMTPArtifactError.invalidFileSize
        }
        var reader = GGUFReader(data: data)
        let magic = try reader.readBytes(count: 4)
        guard magic == Data("GGUF".utf8) else {
            throw GGUFMTPArtifactError.invalidMagic
        }
        guard try reader.readUInt32() == 3 else {
            throw GGUFMTPArtifactError.unsupportedVersion
        }

        let tensorCount = try reader.readCount(limit: 1_000_000)
        let metadataCount = try reader.readCount(limit: 1_000_000)
        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(metadataCount)
        for _ in 0..<metadataCount {
            let key = try reader.readString()
            guard metadata[key] == nil else {
                throw GGUFMTPArtifactError.invalidMetadata("duplicate key \(key)")
            }
            metadata[key] = try reader.readValue()
        }

        let architecture = try requiredString(
            metadata["general.architecture"], key: "general.architecture")
        guard architecture == "qwen4exp" else {
            throw GGUFMTPArtifactError.incompatibleArchitecture(architecture)
        }
        let nextnKey = "\(architecture).nextn_predict_layers"
        let nextnPredictLayers = try requiredPositiveInt(metadata[nextnKey], key: nextnKey)
        let alignment = try optionalAlignment(metadata["general.alignment"])
        var tensors: [Tensor] = []
        tensors.reserveCapacity(tensorCount)
        var names = Set<String>()
        for _ in 0..<tensorCount {
            let name = try reader.readString()
            guard names.insert(name).inserted else {
                throw GGUFMTPArtifactError.invalidTensor("duplicate tensor \(name)")
            }
            let dimensionCountValue = try reader.readUInt32()
            guard dimensionCountValue <= 8 else {
                throw GGUFMTPArtifactError.limitExceeded(
                    "dimension count \(dimensionCountValue) > 8")
            }
            let dimensionCount = Int(dimensionCountValue)
            var dimensions: [UInt64] = []
            dimensions.reserveCapacity(dimensionCount)
            for _ in 0..<dimensionCount {
                dimensions.append(try reader.readUInt64())
            }
            let type = try reader.readUInt32()
            let offset = try reader.readUInt64()
            let byteCount: UInt64?
            if name.contains(".nextn.") {
                byteCount = try tensorByteCount(
                    type: type, dimensions: dimensions, name: name)
            } else {
                byteCount = nil
            }
            tensors.append(Tensor(name: name, dimensions: dimensions,
                                  type: type, offset: offset,
                                  byteCount: byteCount, absoluteOffset: nil))
        }

        let dataOffset = try alignedOffset(UInt64(reader.offset), alignment: alignment)
        guard dataOffset <= sourceFileSize else {
            throw GGUFMTPArtifactError.invalidDataOffset(dataOffset)
        }
        let expectedPrefix = mtpTensorPrefix
        let mtpTensors = tensors.filter { $0.name.hasPrefix(expectedPrefix) }
        guard !mtpTensors.isEmpty else {
            throw GGUFMTPArtifactError.missingMtpTensors(expectedPrefix)
        }
        for tensor in tensors where tensor.name.contains(".nextn.") {
            guard tensor.name.hasPrefix(expectedPrefix) else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "unexpected MTP tensor \(tensor.name)")
            }
        }

        var resolvedTensors: [Tensor] = []
        resolvedTensors.reserveCapacity(tensors.count)
        var ranges: [(start: UInt64, end: UInt64, name: String)] = []
        for tensor in tensors {
            guard let byteCount = tensor.byteCount else {
                resolvedTensors.append(tensor)
                continue
            }
            let (absoluteOffset, offsetOverflow) = dataOffset.addingReportingOverflow(
                tensor.offset)
            let (end, endOverflow) = absoluteOffset.addingReportingOverflow(byteCount)
            guard !offsetOverflow, !endOverflow, end <= sourceFileSize else {
                throw GGUFMTPArtifactError.tensorRangeOutOfBounds(
                    name: tensor.name, offset: absoluteOffset, byteCount: byteCount)
            }
            resolvedTensors.append(Tensor(
                name: tensor.name, dimensions: tensor.dimensions, type: tensor.type,
                offset: tensor.offset, byteCount: byteCount,
                absoluteOffset: absoluteOffset))
            ranges.append((absoluteOffset, end, tensor.name))
        }
        let sortedRanges = ranges.sorted { $0.start < $1.start }
        for pair in zip(sortedRanges, sortedRanges.dropFirst())
            where pair.1.start < pair.0.end {
            throw GGUFMTPArtifactError.overlappingTensors(
                first: pair.0.name, second: pair.1.name)
        }

        return GGUFMTPArtifact(
            architecture: architecture,
            nextnPredictLayers: nextnPredictLayers,
            alignment: alignment,
            dataOffset: dataOffset,
            tensors: resolvedTensors)
    }

    private static func tensorByteCount(
        type: UInt32,
        dimensions: [UInt64],
        name: String
    ) throws -> UInt64 {
        guard let layout = tensorLayout(type: type) else {
            throw GGUFMTPArtifactError.unsupportedTensorType(type: type, name: name)
        }
        var elementCount: UInt64 = 1
        for dimension in dimensions {
            guard dimension > 0 else {
                throw GGUFMTPArtifactError.invalidTensor(
                    "zero dimension for \(name)")
            }
            let (product, overflow) = elementCount.multipliedReportingOverflow(
                by: dimension)
            guard !overflow else {
                throw GGUFMTPArtifactError.tensorShapeOverflow(name)
            }
            elementCount = product
        }
        guard elementCount % layout.blockElements == 0 else {
            throw GGUFMTPArtifactError.invalidTensor(
                "shape is not divisible by block size for \(name)")
        }
        let blocks = elementCount / layout.blockElements
        let (byteCount, overflow) = blocks.multipliedReportingOverflow(
            by: layout.blockBytes)
        guard !overflow else {
            throw GGUFMTPArtifactError.tensorShapeOverflow(name)
        }
        return byteCount
    }

    private static func tensorLayout(
        type: UInt32
    ) -> (blockElements: UInt64, blockBytes: UInt64)? {
        switch type {
        case 0: return (1, 4)       // F32
        case 1, 30: return (1, 2)   // F16, BF16
        case 2: return (32, 18)     // Q4_0
        case 3: return (32, 20)     // Q4_1
        case 6: return (32, 22)     // Q5_0
        case 7: return (32, 24)     // Q5_1
        case 8: return (32, 34)     // Q8_0
        case 9: return (32, 36)     // Q8_1
        case 10: return (256, 84)   // Q2_K
        case 11: return (256, 110)  // Q3_K
        case 12: return (256, 144)  // Q4_K
        case 13: return (256, 176)  // Q5_K
        case 14: return (256, 210)  // Q6_K
        case 15: return (256, 292)  // Q8_K
        case 16: return (256, 66)   // IQ2_XXS
        case 17: return (256, 74)   // IQ2_XS
        case 18: return (256, 98)   // IQ3_XXS
        case 19: return (256, 50)   // IQ1_S
        case 20: return (32, 18)    // IQ4_NL
        case 21: return (256, 110)  // IQ3_S
        case 22: return (256, 82)   // IQ2_S
        case 23: return (256, 136)  // IQ4_XS
        case 24: return (1, 1)      // I8
        case 25: return (1, 2)      // I16
        case 26: return (1, 4)      // I32
        case 27: return (1, 8)      // I64
        case 28: return (256, 56)   // IQ1_M
        default: return nil
        }
    }

    private static func requiredString(
        _ value: GGUFValue?, key: String
    ) throws -> String {
        guard case .string(let string) = value else {
            throw GGUFMTPArtifactError.invalidMetadata(
                "missing or non-string \(key)")
        }
        return string
    }

    private static func requiredPositiveInt(
        _ value: GGUFValue?, key: String
    ) throws -> Int {
        let number: UInt64
        switch value {
        case .unsigned(let value): number = value
        case .signed(let value) where value >= 0: number = UInt64(value)
        default:
            throw GGUFMTPArtifactError.invalidMetadata(
                "missing or non-integer \(key)")
        }
        guard number > 0, number <= UInt64(Int.max) else {
            throw GGUFMTPArtifactError.invalidMetadata(
                "invalid positive integer \(key)")
        }
        return Int(number)
    }

    private static func optionalAlignment(_ value: GGUFValue?) throws -> UInt64 {
        guard let value else { return 32 }
        let alignment: UInt64
        switch value {
        case .unsigned(let value): alignment = value
        case .signed(let value) where value > 0: alignment = UInt64(value)
        default:
            throw GGUFMTPArtifactError.invalidMetadata(
                "general.alignment must be a positive integer")
        }
        guard alignment > 0, alignment <= 1 << 20,
              alignment & (alignment - 1) == 0 else {
            throw GGUFMTPArtifactError.invalidMetadata(
                "general.alignment must be a power of two")
        }
        return alignment
    }

    private static func alignedOffset(
        _ offset: UInt64, alignment: UInt64
    ) throws -> UInt64 {
        let remainder = offset % alignment
        guard remainder != 0 else { return offset }
        let padding = alignment - remainder
        let (result, overflow) = offset.addingReportingOverflow(padding)
        guard !overflow else {
            throw GGUFMTPArtifactError.invalidDataOffset(offset)
        }
        return result
    }
}

enum GGUFMTPArtifactError: Error, CustomStringConvertible, Equatable {
    case invalidMagic
    case unsupportedVersion
    case truncated(offset: Int)
    case limitExceeded(String)
    case invalidMetadata(String)
    case incompatibleArchitecture(String)
    case invalidTensor(String)
    case missingMtpTensors(String)
    case invalidDataOffset(UInt64)
    case invalidFileSize
    case unsupportedTensorType(type: UInt32, name: String)
    case tensorShapeOverflow(String)
    case tensorRangeOutOfBounds(name: String, offset: UInt64, byteCount: UInt64)
    case overlappingTensors(first: String, second: String)
    case tensorPayloadTooLarge(name: String, byteCount: UInt64)
    case tensorPayloadTruncated(name: String, expected: UInt64, actual: UInt64)
    case fileReadFailed(name: String)

    var description: String {
        switch self {
        case .invalidMagic: return "invalid GGUF magic"
        case .unsupportedVersion: return "unsupported GGUF version"
        case .truncated(let offset): return "truncated GGUF at byte \(offset)"
        case .limitExceeded(let detail): return "GGUF limit exceeded: \(detail)"
        case .invalidMetadata(let detail): return "invalid GGUF metadata: \(detail)"
        case .incompatibleArchitecture(let value):
            return "unsupported GGUF architecture \(value)"
        case .invalidTensor(let detail): return "invalid GGUF tensor: \(detail)"
        case .missingMtpTensors(let prefix):
            return "no MTP tensors with prefix \(prefix)"
        case .invalidDataOffset(let offset):
            return "GGUF data offset \(offset) exceeds file size"
        case .invalidFileSize:
            return "invalid GGUF file size"
        case .unsupportedTensorType(let type, let name):
            return "unsupported GGUF tensor type \(type) for \(name)"
        case .tensorShapeOverflow(let name):
            return "GGUF tensor shape overflows for \(name)"
        case .tensorRangeOutOfBounds(let name, let offset, let byteCount):
            return "GGUF tensor range is out of bounds for \(name): \(offset)+\(byteCount)"
        case .overlappingTensors(let first, let second):
            return "overlapping GGUF tensor ranges: \(first), \(second)"
        case .tensorPayloadTooLarge(let name, let byteCount):
            return "GGUF tensor payload is too large for \(name): \(byteCount)"
        case .tensorPayloadTruncated(let name, let expected, let actual):
            return "GGUF tensor payload is truncated for \(name): \(actual)/\(expected)"
        case .fileReadFailed(let name):
            return "failed to read GGUF tensor payload for \(name)"
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
}

private struct GGUFReader {
    private static let maxStringBytes = 16 * 1024 * 1024
    private static let maxArrayElements = 1_000_000

    let data: Data
    var offset = 0

    mutating func readBytes(count: Int) throws -> Data {
        guard count >= 0, offset <= data.count,
              count <= data.count - offset else {
            throw GGUFMTPArtifactError.truncated(offset: offset)
        }
        let result = data.subdata(in: offset..<(offset + count))
        offset += count
        return result
    }

    mutating func readUInt32() throws -> UInt32 {
        let bytes = try readBytes(count: 4)
        return bytes.enumerated().reduce(UInt32(0)) { result, item in
            result | UInt32(item.element) << UInt32(item.offset * 8)
        }
    }

    mutating func readUInt64() throws -> UInt64 {
        let bytes = try readBytes(count: 8)
        return bytes.enumerated().reduce(UInt64(0)) { result, item in
            result | UInt64(item.element) << UInt64(item.offset * 8)
        }
    }

    mutating func readCount(limit: Int) throws -> Int {
        let value = try readUInt64()
        guard value <= UInt64(limit) else {
            throw GGUFMTPArtifactError.limitExceeded("count \(value) > \(limit)")
        }
        return Int(value)
    }

    mutating func readString() throws -> String {
        let count = try readUInt64()
        guard count <= UInt64(Self.maxStringBytes),
              count <= UInt64(Int.max) else {
            throw GGUFMTPArtifactError.limitExceeded("string length \(count)")
        }
        let bytes = try readBytes(count: Int(count))
        guard let string = String(data: bytes, encoding: .utf8) else {
            throw GGUFMTPArtifactError.invalidMetadata("string is not UTF-8")
        }
        return string
    }

    mutating func readValue() throws -> GGUFValue {
        let type = try readUInt32()
        switch type {
        case 0: return .unsigned(UInt64(try readBytes(count: 1)[0]))
        case 1: return .signed(Int64(Int8(bitPattern: try readBytes(count: 1)[0])))
        case 2: return .unsigned(UInt64(try readFixedUInt16()))
        case 3: return .signed(Int64(Int16(bitPattern: try readFixedUInt16())))
        case 4: return .unsigned(UInt64(try readUInt32()))
        case 5: return .signed(Int64(Int32(bitPattern: try readUInt32())))
        case 6:
            let bits = try readUInt32()
            return .float(Double(Float(bitPattern: bits)))
        case 7: return .boolean(try readBytes(count: 1)[0] != 0)
        case 8: return .string(try readString())
        case 9:
            let elementType = try readUInt32()
            let count = try readCount(limit: Self.maxArrayElements)
            var values: [GGUFValue] = []
            values.reserveCapacity(count)
            for _ in 0..<count {
                values.append(try readValue(type: elementType))
            }
            return .array(values)
        case 10: return .unsigned(try readUInt64())
        case 11: return .signed(Int64(bitPattern: try readUInt64()))
        case 12:
            return .float(Double(bitPattern: try readUInt64()))
        default:
            throw GGUFMTPArtifactError.invalidMetadata("unsupported value type \(type)")
        }
    }

    private mutating func readValue(type: UInt32) throws -> GGUFValue {
        switch type {
        case 0: return .unsigned(UInt64(try readBytes(count: 1)[0]))
        case 1: return .signed(Int64(Int8(bitPattern: try readBytes(count: 1)[0])))
        case 2: return .unsigned(UInt64(try readFixedUInt16()))
        case 3: return .signed(Int64(Int16(bitPattern: try readFixedUInt16())))
        case 4: return .unsigned(UInt64(try readUInt32()))
        case 5: return .signed(Int64(Int32(bitPattern: try readUInt32())))
        case 6: return .float(Double(Float(bitPattern: try readUInt32())))
        case 7: return .boolean(try readBytes(count: 1)[0] != 0)
        case 8: return .string(try readString())
        case 10: return .unsigned(try readUInt64())
        case 11: return .signed(Int64(bitPattern: try readUInt64()))
        case 12: return .float(Double(bitPattern: try readUInt64()))
        default:
            throw GGUFMTPArtifactError.invalidMetadata(
                "unsupported array element type \(type)")
        }
    }

    private mutating func readFixedUInt16() throws -> UInt16 {
        let bytes = try readBytes(count: 2)
        return UInt16(bytes[0]) | UInt16(bytes[1]) << 8
    }
}
