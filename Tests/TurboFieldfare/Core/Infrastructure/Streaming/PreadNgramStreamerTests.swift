import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct PreadNgramStreamerTests {
    @Test func readsGroup32RowsAcrossShardBoundary() async throws {
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
        let asyncRows = try await streamer.readAsync(addresses: [1, 2])
        let singleWorkerRows = try await streamer.readAsync(
            addresses: [1, 2], maxConcurrentReads: 1)

        #expect(asyncRows == rows)
        #expect(singleWorkerRows == rows)
        #expect(rows.count == 2)
        #expect(rows[0].count == 64)
        #expect(rows[0][0] == 13)
        #expect(rows[0][1] == 14)
        #expect(rows[0][32] == 26)
        #expect(rows[1][0] == 15)
        #expect(rows[1][1] == 16)
        #expect(rows[1][32] == 30)

        let duplicateAddresses: [Int64] = [1, 2, 1, 0, 3, 2]
        let duplicateRows = try streamer.read(addresses: duplicateAddresses)
        for workerLimit in [1, 4] {
            let concurrentRows = try await streamer.readAsync(
                addresses: duplicateAddresses,
                maxConcurrentReads: workerLimit)
            #expect(concurrentRows == duplicateRows)
        }

        await #expect(throws: StreamerError.self) {
            try await streamer.readAsync(addresses: [1], maxConcurrentReads: 0)
        }
        #expect(throws: StreamerError.self) {
            try streamer.readRow(address: -1)
        }
        #expect(throws: StreamerError.self) {
            try streamer.readRow(address: 4)
        }
        await #expect(throws: StreamerError.self) {
            try await streamer.readAsync(addresses: [-1])
        }
    }

    @Test func cachesSmallGathersAndBypassesLargeGathers() async throws {
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
        let streamer = try PreadNgramStreamer(
            directoryURL: directory,
            layout: layout,
            rowCacheBytes: 512,
            rowCacheMaxUniqueRows: 2)

        let addresses: [Int64] = [1, 2]
        let expected = try streamer.read(addresses: addresses)
        let firstRows = try await streamer.readAsync(
            addresses: addresses, maxConcurrentReads: 4)
        #expect(firstRows == expected)
        #expect(streamer.cacheDiagnostics.misses == 2)
        #expect(streamer.cacheDiagnostics.currentBytes == 512)
        #expect(streamer.cacheDiagnostics.peakBytes == 512)

        let secondRows = try await streamer.readAsync(
            addresses: [2, 1], maxConcurrentReads: 1)
        #expect(secondRows == [expected[1], expected[0]])
        #expect(streamer.cacheDiagnostics.hits == 2)
        #expect(streamer.cacheDiagnostics.misses == 2)

        let evictingRows = try await streamer.readAsync(
            addresses: [0, 3], maxConcurrentReads: 4)
        let evictingReference = try streamer.read(addresses: [0, 3])
        #expect(evictingRows == evictingReference)
        #expect(streamer.cacheDiagnostics.evictions == 2)
        #expect(streamer.cacheDiagnostics.currentBytes == 512)

        let largeAddresses: [Int64] = [0, 1, 2]
        let largeRows = try await streamer.readAsync(
            addresses: largeAddresses, maxConcurrentReads: 4)
        let largeReference = try streamer.read(addresses: largeAddresses)
        #expect(largeRows == largeReference)
        #expect(streamer.cacheDiagnostics.bypasses == 1)
        #expect(streamer.cacheDiagnostics.currentBytes == 512)

        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(
                directoryURL: directory, layout: layout, rowCacheBytes: -1)
        }
        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(
                directoryURL: directory, layout: layout, rowCacheMaxUniqueRows: 0)
        }
        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(
                directoryURL: directory, layout: layout, rowProfileMaxRows: -1)
        }
    }

    @Test func profilesRowsWithBoundedRankedDiagnostics() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = "shard_0.bin"
        let data = shardData(seeds: [1, 3, 5, 7])
        try data.write(to: directory.appendingPathComponent(file))
        let layout = PackedNgramsLayout(
            layer: 1,
            splitParts: 1,
            groupSize: 32,
            shards: [profileShardEntry(
                index: 0, file: file, fileSize: UInt64(data.count))])
        let streamer = try PreadNgramStreamer(
            directoryURL: directory,
            layout: layout,
            rowCacheBytes: 512,
            rowCacheMaxUniqueRows: 2,
            rowProfileMaxRows: 2)

        _ = try await streamer.readAsync(addresses: [1, 2], maxConcurrentReads: 1)
        _ = try await streamer.readAsync(addresses: [2, 1], maxConcurrentReads: 1)

        let profile = streamer.rowProfileSnapshot
        #expect(profile.count == 2)
        #expect(profile.map(\.address) == [1, 2])
        #expect(profile.allSatisfy { $0.requests == 2 })
        #expect(profile.allSatisfy { $0.cacheHits == 1 })
        #expect(profile.allSatisfy { $0.cacheMisses == 1 })
        #expect(profile.allSatisfy { $0.readNanos > 0 })

        _ = try await streamer.readAsync(addresses: [0, 3], maxConcurrentReads: 1)
        #expect(streamer.rowProfileSnapshot.count == 2)
    }

    @Test func pinsRowsOutsideTheOrdinaryLRUBudget() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = "shard_0.bin"
        let data = shardData(seeds: [1, 3, 5, 7])
        try data.write(to: directory.appendingPathComponent(file))
        let layout = PackedNgramsLayout(
            layer: 1,
            splitParts: 1,
            groupSize: 32,
            shards: [profileShardEntry(
                index: 0, file: file, fileSize: UInt64(data.count))])
        let streamer = try PreadNgramStreamer(
            directoryURL: directory,
            layout: layout,
            rowCacheBytes: 256,
            rowCacheMaxUniqueRows: 2,
            pinnedRows: [0],
            pinnedRowByteBudget: 256)

        _ = try await streamer.readAsync(addresses: [1, 2], maxConcurrentReads: 1)
        _ = try await streamer.readAsync(addresses: [0], maxConcurrentReads: 1)

        #expect(streamer.cacheDiagnostics.pinnedRowCount == 1)
        #expect(streamer.cacheDiagnostics.pinnedBytes == 256)
        #expect(streamer.cacheDiagnostics.currentBytes == 512)
        #expect(streamer.cacheDiagnostics.hits == 1)
        #expect(streamer.cacheDiagnostics.misses == 2)
        #expect(streamer.cacheDiagnostics.evictions == 1)

        let largeAddresses: [Int64] = [0, 1, 2]
        let largeReference = try streamer.read(addresses: largeAddresses)
        let largeRows = try await streamer.readAsync(
            addresses: largeAddresses, maxConcurrentReads: 3)
        #expect(largeRows == largeReference)
        #expect(streamer.cacheDiagnostics.bypasses == 1)
        #expect(streamer.cacheDiagnostics.hits == 3)
        #expect(streamer.cacheDiagnostics.currentBytes == 512)
    }

    @Test func rejectsInvalidPinnedRowsAndBudget() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = "shard_0.bin"
        let data = shardData(seeds: [1, 3, 5, 7])
        try data.write(to: directory.appendingPathComponent(file))
        let layout = PackedNgramsLayout(
            layer: 1,
            splitParts: 1,
            groupSize: 32,
            shards: [profileShardEntry(
                index: 0, file: file, fileSize: UInt64(data.count))])

        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(
                directoryURL: directory,
                layout: layout,
                pinnedRows: [0, 0],
                pinnedRowByteBudget: 512)
        }
        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(
                directoryURL: directory,
                layout: layout,
                pinnedRows: [4],
                pinnedRowByteBudget: 256)
        }
        #expect(throws: StreamerError.self) {
            try PreadNgramStreamer(
                directoryURL: directory,
                layout: layout,
                pinnedRows: [0, 1],
                pinnedRowByteBudget: 256)
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

    private func profileShardEntry(index: Int, file: String, fileSize: UInt64) -> NgramShardEntry {
        NgramShardEntry(
            shard: index,
            file: file,
            fileSize: fileSize,
            weight: NgramComponentEntry(
                offset: 0, size: 128, dtype: "U32", shape: [4, 8], bits: 4),
            scales: NgramComponentEntry(
                offset: 128, size: 16, dtype: "BF16", shape: [4, 2], bits: nil),
            biases: NgramComponentEntry(
                offset: 144, size: 16, dtype: "BF16", shape: [4, 2], bits: nil))
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
