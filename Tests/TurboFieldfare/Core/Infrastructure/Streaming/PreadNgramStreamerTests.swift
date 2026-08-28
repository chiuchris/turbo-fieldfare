import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct PreadNgramStreamerTests {
    @Test func readsGroup32RowsAcrossShardBoundary() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let shards = try [0, 1].map { shardIndex in
            let file = "shard_\(shardIndex).bin"
            let data = shardData(seeds: shardIndex == 0 ? [1, 3] : [5, 7])
            try data.write(to: directory.appendingPathComponent(file))
            return shardEntry(index: shardIndex, file: file, fileSize: UInt64(data.count))
        }
        let layout = PackedNgramsLayout(
            layer: 1, splitParts: 2, groupSize: 32, shards: shards)
        let streamer = try PreadNgramStreamer(directoryURL: directory, layout: layout)

        let rows = try streamer.read(addresses: [1, 2])

        #expect(rows.count == 2)
        #expect(rows[0].count == 64)
        #expect(rows[0][0] == 13)
        #expect(rows[0][1] == 14)
        #expect(rows[0][32] == 26)
        #expect(rows[1][0] == 15)
        #expect(rows[1][1] == 16)
        #expect(rows[1][32] == 30)
        #expect(throws: StreamerError.self) {
            try streamer.readRow(address: -1)
        }
        #expect(throws: StreamerError.self) {
            try streamer.readRow(address: 4)
        }
    }

    @Test func rejectsTruncatedShardBeforeReading() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = "shard_0.bin"
        try Data(repeating: 0, count: 79).write(
            to: directory.appendingPathComponent(file))
        let layout = PackedNgramsLayout(
            layer: 1,
            splitParts: 1,
            groupSize: 32,
            shards: [shardEntry(index: 0, file: file, fileSize: 80)])

        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(directoryURL: directory, layout: layout)
        }
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pread-ngram-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func shardEntry(index: Int, file: String, fileSize: UInt64) -> NgramShardEntry {
        NgramShardEntry(
            shard: index,
            file: file,
            fileSize: fileSize,
            weight: NgramComponentEntry(
                offset: 0, size: 64, dtype: "U32", shape: [2, 8], bits: 4),
            scales: NgramComponentEntry(
                offset: 64, size: 8, dtype: "BF16", shape: [2, 2], bits: nil),
            biases: NgramComponentEntry(
                offset: 72, size: 8, dtype: "BF16", shape: [2, 2], bits: nil))
    }

    private func shardData(seeds: [UInt8]) -> Data {
        var data = Data()
        for seed in seeds {
            data.append(contentsOf: repeatElement(seed | ((seed + 1) << 4), count: 32))
        }
        for _ in seeds {
            appendUInt16(Quantization.bf16Bits(1), to: &data)
            appendUInt16(Quantization.bf16Bits(2), to: &data)
        }
        for _ in seeds {
            appendUInt16(Quantization.bf16Bits(10), to: &data)
            appendUInt16(Quantization.bf16Bits(20), to: &data)
        }
        return data
    }

    private func appendUInt16(_ value: UInt16, to data: inout Data) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
    }
}
