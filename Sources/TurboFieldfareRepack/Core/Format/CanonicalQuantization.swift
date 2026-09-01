import Foundation

/// Converts supported MLX affine packed weights to the runtime's canonical
/// Q4/group-32 representation without materializing a whole tensor.
enum CanonicalQuantization {
    static let target = QuantSpec(bits: 4, groupSize: 32)

    struct Layout: Equatable {
        let rowCount: Int
        let inputWidth: Int
        let sourceWordsPerRow: Int
        let sourceGroupsPerRow: Int
        let outputWordsPerRow: Int
        let outputGroupsPerRow: Int
    }

    static func layout(shape: [UInt64], source: QuantSpec) throws -> Layout {
        guard [4, 5, 8].contains(source.bits), source.groupSize > 0 else {
            throw RepackError.configurationInvalid(
                detail: "unsupported affine quantization \(source.bits)-bit/group-\(source.groupSize)")
        }
        guard shape.count >= 2 else {
            throw RepackError.configurationInvalid(
                detail: "packed quantized tensor must have rank at least two")
        }

        let rowCount = try checkedProduct(shape.dropLast(), label: "row count")
        let sourceWordsPerRow = try checkedInt(shape.last!, label: "packed width")
        let packedBits = sourceWordsPerRow.multipliedReportingOverflow(by: 32)
        guard !packedBits.overflow, packedBits.partialValue % source.bits == 0 else {
            throw RepackError.configurationInvalid(
                detail: "packed width does not contain a whole number of source values")
        }
        let inputWidth = packedBits.partialValue / source.bits
        guard inputWidth % source.groupSize == 0,
              inputWidth % target.groupSize == 0 else {
            throw RepackError.configurationInvalid(
                detail: "packed width \(inputWidth) is not divisible by source/runtime group size")
        }
        return Layout(
            rowCount: rowCount,
            inputWidth: inputWidth,
            sourceWordsPerRow: sourceWordsPerRow,
            sourceGroupsPerRow: inputWidth / source.groupSize,
            outputWordsPerRow: inputWidth / 8,
            outputGroupsPerRow: inputWidth / target.groupSize)
    }

    static func outputWeightBytes(shape: [UInt64], source: QuantSpec) throws -> UInt64 {
        let layout = try layout(shape: shape, source: source)
        return UInt64(layout.rowCount * layout.outputWordsPerRow * MemoryLayout<UInt32>.size)
    }

    static func outputCompanionBytes(shape: [UInt64], source: QuantSpec) throws -> UInt64 {
        let layout = try layout(shape: shape, source: source)
        return UInt64(layout.rowCount * layout.outputGroupsPerRow * MemoryLayout<UInt16>.size)
    }

    static func writeConverted(
        weight: UnsafeRawBufferPointer,
        shape: [UInt64],
        scales: UnsafeRawBufferPointer,
        biases: UnsafeRawBufferPointer,
        source: QuantSpec,
        destinationFd: Int32,
        destinationPath: String,
        weightOffset: UInt64,
        scaleOffset: UInt64,
        biasOffset: UInt64,
        audit: RepackAudit
    ) throws {
        let layout = try layout(shape: shape, source: source)
        let sourceWeightBytes = layout.rowCount * layout.sourceWordsPerRow * MemoryLayout<UInt32>.size
        let sourceCompanionBytes = layout.rowCount * layout.sourceGroupsPerRow * MemoryLayout<UInt16>.size
        guard weight.count >= sourceWeightBytes,
              scales.count >= sourceCompanionBytes,
              biases.count >= sourceCompanionBytes else {
            throw RepackError.configurationInvalid(
                detail: "quantized source buffers are shorter than their declared shape")
        }

        let outputWeightRowBytes = layout.outputWordsPerRow * MemoryLayout<UInt32>.size
        let outputCompanionRowBytes = layout.outputGroupsPerRow * MemoryLayout<UInt16>.size
        var outputScales = [UInt8](repeating: 0, count: outputCompanionRowBytes)
        var outputBiases = [UInt8](repeating: 0, count: outputCompanionRowBytes)
        var values = [Float](repeating: 0, count: target.groupSize)
        let mask = UInt64((1 << source.bits) - 1)

        for row in 0..<layout.rowCount {
            var outputWeight = [UInt8](repeating: UInt8(0), count: outputWeightRowBytes)
            for outputGroup in 0..<layout.outputGroupsPerRow {
                let groupStart = outputGroup * target.groupSize
                var minimum = Float.infinity
                var maximum = -Float.infinity
                for index in 0..<target.groupSize {
                    let column = groupStart + index
                    let bitOffset = column * source.bits
                    let wordIndex = row * layout.sourceWordsPerRow + bitOffset / 32
                    let shift = bitOffset % 32
                    let word = UInt64(weight.loadUnaligned(
                        fromByteOffset: wordIndex * MemoryLayout<UInt32>.size,
                        as: UInt32.self).littleEndian)
                    let nextWord: UInt64 = shift + source.bits > 32
                        ? UInt64(weight.loadUnaligned(
                            fromByteOffset: (wordIndex + 1) * MemoryLayout<UInt32>.size,
                            as: UInt32.self).littleEndian)
                        : 0
                    let packed = word | (nextWord << 32)
                    let quantized = (packed >> UInt64(shift)) & mask
                    let sourceGroup = column / source.groupSize
                    let companionIndex = row * layout.sourceGroupsPerRow + sourceGroup
                    let scale = readBF16(scales, index: companionIndex)
                    let bias = readBF16(biases, index: companionIndex)
                    let value = scale * Float(quantized) + bias
                    guard value.isFinite else {
                        throw RepackError.configurationInvalid(
                            detail: "non-finite dequantized value in row \(row)")
                    }
                    values[index] = value
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                }

                let scale = maximum == minimum ? Float(1) : (maximum - minimum) / 15
                let bias = maximum == minimum ? minimum : minimum
                writeBF16(scale, to: &outputScales, index: outputGroup)
                writeBF16(bias, to: &outputBiases, index: outputGroup)
                for index in 0..<target.groupSize {
                    let quantized: UInt32
                    if maximum == minimum {
                        quantized = 0
                    } else {
                        quantized = UInt32(max(0, min(15, Int(((values[index] - bias) / scale).rounded()))))
                    }
                    let outputIndex = groupStart + index
                    let wordIndex = outputIndex / 8
                    let shift = (outputIndex % 8) * 4
                    outputWeight[wordIndex * 4 + shift / 8] |= UInt8(quantized << UInt32(shift % 8))
                }
            }

            let outputWeightStart = weightOffset + UInt64(row * outputWeightRowBytes)
            let outputScaleStart = scaleOffset + UInt64(row * outputCompanionRowBytes)
            let outputBiasStart = biasOffset + UInt64(row * outputCompanionRowBytes)
            try write(outputWeight, to: destinationFd, path: destinationPath,
                       offset: outputWeightStart, audit: audit)
            try write(outputScales, to: destinationFd, path: destinationPath,
                       offset: outputScaleStart, audit: audit)
            try write(outputBiases, to: destinationFd, path: destinationPath,
                       offset: outputBiasStart, audit: audit)
            audit.recordRead(bytes: outputWeightRowBytes + 2 * sourceCompanionBytes / layout.rowCount)
        }
    }

    private static func checkedProduct<S: Sequence>(_ values: S, label: String) throws -> Int
    where S.Element == UInt64 {
        try values.reduce(1) { partial, value in
            let factor = try checkedInt(value, label: label)
            let (product, overflow) = partial.multipliedReportingOverflow(by: factor)
            guard !overflow else {
                throw RepackError.configurationInvalid(detail: "\(label) overflows Int")
            }
            return product
        }
    }

    private static func checkedInt(_ value: UInt64, label: String) throws -> Int {
        guard value <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(detail: "\(label) overflows Int")
        }
        return Int(value)
    }

    private static func readBF16(_ buffer: UnsafeRawBufferPointer, index: Int) -> Float {
        let bits = buffer.loadUnaligned(
            fromByteOffset: index * MemoryLayout<UInt16>.size,
            as: UInt16.self).littleEndian
        return Float(bitPattern: UInt32(bits) << 16)
    }

    private static func writeBF16(_ value: Float, to bytes: inout [UInt8], index: Int) {
        let rounded = UInt16((value.bitPattern &+ 0x8000) >> 16).littleEndian
        withUnsafeBytes(of: rounded) { raw in
            bytes.replaceSubrange((index * 2)..<(index * 2 + 2), with: raw)
        }
    }

    private static func write(
        _ bytes: [UInt8],
        to fd: Int32,
        path: String,
        offset: UInt64,
        audit: RepackAudit
    ) throws {
        try bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.pwriteAll(fd: fd, path: path, buf: base,
                                count: bytes.count, offset: offset)
        }
        audit.recordTile(bytes: bytes.count)
        audit.recordWrite(bytes: bytes.count)
    }
}
