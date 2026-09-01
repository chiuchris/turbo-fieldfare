import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct CanonicalQuantizationTests {
    @Test
    func computesCanonicalGeometryForEightBitSource() throws {
        let layout = try CanonicalQuantization.layout(
            shape: [2, 16], source: QuantSpec(bits: 8, groupSize: 64))

        #expect(layout.rowCount == 2)
        #expect(layout.inputWidth == 64)
        #expect(layout.sourceGroupsPerRow == 1)
        #expect(layout.outputWordsPerRow == 8)
        #expect(layout.outputGroupsPerRow == 2)
        #expect(try CanonicalQuantization.outputWeightBytes(
            shape: [2, 16], source: QuantSpec(bits: 8, groupSize: 64)) == 64)
        #expect(try CanonicalQuantization.outputCompanionBytes(
            shape: [2, 16], source: QuantSpec(bits: 8, groupSize: 64)) == 8)
    }

    @Test
    func convertsPackedSourceIntoQ4RowsAndBF16Companions() throws {
        let shape: [UInt64] = [2, 16]
        let source = QuantSpec(bits: 8, groupSize: 64)
        var packed = [UInt32](repeating: 0, count: 32)
        for index in 0..<128 {
            let value = index < 64 ? index % 16 : 15 - (index % 16)
            packed[index / 4] |= UInt32(value) << UInt32((index % 4) * 8)
        }
        let scales = bf16Data([1, 1])
        let biases = bf16Data([0, 0])
        let outputBytes = 64 + 8 + 8
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let output = Data(repeating: 0, count: outputBytes)
        try output.write(to: url)
        let fd = try Posix.openExistingRW(url.path)
        defer { close(fd) }
        let audit = RepackAudit()

        try packed.withUnsafeBytes { weightRaw in
            try scales.withUnsafeBytes { scalesRaw in
                try biases.withUnsafeBytes { biasesRaw in
                    try CanonicalQuantization.writeConverted(
                        weight: weightRaw,
                        shape: shape,
                        scales: scalesRaw,
                        biases: biasesRaw,
                        source: source,
                        destinationFd: fd,
                        destinationPath: url.path,
                        weightOffset: 0,
                        scaleOffset: 64,
                        biasOffset: 72,
                        audit: audit)
                }
            }
        }

        let result = try Data(contentsOf: url)
        let words = result.prefix(64).withUnsafeBytes { raw in
            (0..<16).map { raw.loadUnaligned(
                fromByteOffset: $0 * 4, as: UInt32.self).littleEndian }
        }
        #expect(words[0] == 0x76543210)
        #expect(words[1] == 0xFEDCBA98)
        #expect(words[8] == 0x89ABCDEF)
        #expect(words[9] == 0x01234567)
        #expect(audit.outputBytesWritten == UInt64(outputBytes))
    }

    private func bf16Data(_ values: [Float]) -> Data {
        var result = Data()
        for value in values {
            var bits = UInt16((value.bitPattern &+ 0x8000) >> 16).littleEndian
            withUnsafeBytes(of: &bits) { result.append(contentsOf: $0) }
        }
        return result
    }
}
