import Testing
import Foundation
import Darwin
import Metal
@testable import TurboFieldfare

@Suite struct PagedResidentCacheTests {
    @Test func loadsPayloadScalesAndBiasesIntoOneBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let fileURL = try makeWeightsFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let fileDescriptor = open(fileURL.path, O_RDONLY)
        defer { close(fileDescriptor) }
        let cache = try PagedResidentCache(
            fileDescriptor: fileDescriptor,
            residentFileOffset: 16,
            residentSize: 64,
            capacityBytes: 16,
            device: device)
        let entry = ResidentIndexEntry(
            name: "tensor",
            dtype: 1,
            fileOffset: 20,
            sizeBytes: 4,
            shape: (4, 1, 1, 1),
            scaleOffset: 36,
            scaleSize: 2,
            biasOffset: 46,
            biasSize: 1)

        let view = try cache.tensor(
            name: entry.name,
            entry: entry,
            relativeOffset: 4,
            scaleOffset: 20,
            biasOffset: 30)
        let values = (0..<7).map { view.buffer.contents().load(
            fromByteOffset: Int(view.offset) + $0, as: UInt8.self) }
        #expect(values == [4, 5, 6, 7, 20, 21, 30])
        #expect(view.scaleOffset == 4)
        #expect(view.biasOffset == 6)
        #expect(view.scaleLength == 2)
        #expect(view.biasLength == 1)

        let diagnostics = cache.diagnostics
        #expect(diagnostics.currentBytes == 7)
        #expect(diagnostics.peakBytes == 7)
        #expect(diagnostics.hitCount == 0)
        #expect(diagnostics.missCount == 1)
    }

    @Test func evictsLeastRecentlyUsedTensorAtCapacity() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let fileURL = try makeWeightsFile()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let fileDescriptor = open(fileURL.path, O_RDONLY)
        defer { close(fileDescriptor) }
        let cache = try PagedResidentCache(
            fileDescriptor: fileDescriptor,
            residentFileOffset: 16,
            residentSize: 64,
            capacityBytes: 4,
            device: device)
        let first = makeEntry(name: "first", relativeOffset: 0)
        let second = makeEntry(name: "second", relativeOffset: 8)

        _ = try cache.tensor(name: first.name, entry: first,
                             relativeOffset: 0, scaleOffset: 40, biasOffset: 42)
        _ = try cache.tensor(name: second.name, entry: second,
                             relativeOffset: 8, scaleOffset: 40, biasOffset: 42)
        _ = try cache.tensor(name: first.name, entry: first,
                             relativeOffset: 0, scaleOffset: 40, biasOffset: 42)

        let diagnostics = cache.diagnostics
        #expect(diagnostics.currentBytes == 4)
        #expect(diagnostics.hitCount == 0)
        #expect(diagnostics.missCount == 3)
        #expect(diagnostics.evictionCount == 2)
    }

    private func makeEntry(name: String, relativeOffset: UInt64) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: name,
            dtype: 1,
            fileOffset: 16 + relativeOffset,
            sizeBytes: 2,
            shape: (2, 1, 1, 1),
            scaleOffset: 56,
            scaleSize: 1,
            biasOffset: 58,
            biasSize: 1)
    }

    private func makeWeightsFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-resident-\(UUID().uuidString).bin")
        var data = Data(repeating: 0, count: 16 + 64)
        for index in 0..<64 {
            data[16 + index] = UInt8(index)
        }
        try data.write(to: url)
        return url
    }
}
