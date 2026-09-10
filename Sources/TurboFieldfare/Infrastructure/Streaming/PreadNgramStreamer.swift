import Darwin
import Foundation

public struct NgramCacheDiagnostics: Codable, Sendable, Equatable {
    public let hits: Int
    public let misses: Int
    public let evictions: Int
    public let bypasses: Int
    public let currentBytes: Int
    public let peakBytes: Int
    public let pinnedRowCount: Int
    public let pinnedBytes: Int
    public let cacheCapacityBytes: Int
    public let pinnedRowByteBudget: Int

    public init(hits: Int,
                misses: Int,
                evictions: Int,
                bypasses: Int,
                currentBytes: Int,
                peakBytes: Int,
                pinnedRowCount: Int = 0,
                pinnedBytes: Int = 0,
                cacheCapacityBytes: Int = 0,
                pinnedRowByteBudget: Int = 0) {
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.bypasses = bypasses
        self.currentBytes = currentBytes
        self.peakBytes = peakBytes
        self.pinnedRowCount = pinnedRowCount
        self.pinnedBytes = pinnedBytes
        self.cacheCapacityBytes = cacheCapacityBytes
        self.pinnedRowByteBudget = pinnedRowByteBudget
    }
}

public struct NgramRowProfileEntry: Codable, Sendable, Equatable {
    public let address: Int64
    public let requests: Int
    public let cacheHits: Int
    public let cacheMisses: Int
    public let readNanos: UInt64

    public init(address: Int64,
                requests: Int,
                cacheHits: Int,
                cacheMisses: Int,
                readNanos: UInt64) {
        self.address = address
        self.requests = requests
        self.cacheHits = cacheHits
        self.cacheMisses = cacheMisses
        self.readNanos = readNanos
    }
}

final class PreadNgramStreamer: @unchecked Sendable {
    let layout: PackedNgramsLayout

    private struct CachedRow {
        let values: [Float]
        let byteCount: Int
        var lastUse: UInt64
    }

    private struct RowProfile {
        var requests = 0
        var cacheHits = 0
        var cacheMisses = 0
        var readNanos: UInt64 = 0
    }

    private let fileDescriptors: [Int32]
    private let rowCacheCapacityBytes: Int
    private let rowCacheMaxUniqueRows: Int
    private let rowProfileMaxRows: Int
    private let pinnedRows: Set<Int64>
    private let pinnedRowByteBudget: Int
    private let cacheLock = NSLock()
    private var rowCache: [Int64: CachedRow] = [:]
    private var rowProfile: [Int64: RowProfile] = [:]
    private var cacheClock: UInt64 = 0
    private var cacheBytes = 0
    private var cachePeakBytes = 0
    private var cacheHits = 0
    private var cacheMisses = 0
    private var cacheEvictions = 0
    private var cacheBypasses = 0

    init(directoryURL: URL,
         layout: PackedNgramsLayout,
         rowCacheBytes: Int = 8 * 1024 * 1024,
         rowCacheMaxUniqueRows: Int = 64,
         rowProfileMaxRows: Int = 0,
         pinnedRows: [Int64] = [],
         pinnedRowByteBudget: Int = 0) throws {
        guard !layout.shards.isEmpty,
              layout.groupSize > 0,
              layout.rowWidth > 0,
              layout.rowWidth % layout.groupSize == 0,
              rowCacheBytes >= 0,
              rowCacheMaxUniqueRows > 0,
              rowProfileMaxRows >= 0,
              pinnedRowByteBudget >= 0,
              Set(pinnedRows).count == pinnedRows.count,
              pinnedRows.allSatisfy({ $0 >= 0 }),
              pinnedRows.count <= pinnedRowByteBudget / Self.pinnedPayloadBytes(
                  rowWidth: layout.rowWidth) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "invalid packed n-gram row geometry or cache configuration")
        }
        for address in pinnedRows {
            _ = try layout.locate(globalRow: address)
        }

        var opened: [Int32] = []
        opened.reserveCapacity(layout.shards.count)
        do {
            for shard in layout.shards {
                let path = directoryURL.appendingPathComponent(shard.file).path
                let descriptor = open(path, O_RDONLY | O_CLOEXEC)
                guard descriptor >= 0 else {
                    throw StreamerError.openFailed(path: path, errno: errno)
                }
                opened.append(descriptor)

                var stats = stat()
                guard fstat(descriptor, &stats) == 0,
                      (stats.st_mode & S_IFMT) == S_IFREG,
                      stats.st_size >= 0 else {
                    throw StreamerError.openFailed(
                        path: path, errno: errno == 0 ? EINVAL : errno)
                }
                guard UInt64(stats.st_size) == shard.fileSize else {
                    throw StreamerError.sizeMismatch(
                        expected: shard.fileSize, actual: UInt64(stats.st_size))
                }
            }
        } catch {
            for descriptor in opened { close(descriptor) }
            throw error
        }

        self.layout = layout
        self.fileDescriptors = opened
        self.rowCacheCapacityBytes = rowCacheBytes
        self.rowCacheMaxUniqueRows = rowCacheMaxUniqueRows
        self.rowProfileMaxRows = rowProfileMaxRows
        self.pinnedRows = Set(pinnedRows)
        self.pinnedRowByteBudget = pinnedRowByteBudget
        do {
            for address in pinnedRows {
                try preloadPinnedRow(address: address)
            }
        } catch {
            for descriptor in opened { close(descriptor) }
            throw error
        }
    }

    deinit {
        for descriptor in fileDescriptors { close(descriptor) }
    }

    var cacheDiagnostics: NgramCacheDiagnostics {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return NgramCacheDiagnostics(
            hits: cacheHits,
            misses: cacheMisses,
            evictions: cacheEvictions,
            bypasses: cacheBypasses,
            currentBytes: cacheBytes,
            peakBytes: cachePeakBytes,
            pinnedRowCount: pinnedRows.count,
            pinnedBytes: pinnedRows.count * Self.pinnedPayloadBytes(rowWidth: layout.rowWidth),
            cacheCapacityBytes: rowCacheCapacityBytes,
            pinnedRowByteBudget: pinnedRowByteBudget)
    }

    var rowProfileSnapshot: [NgramRowProfileEntry] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return rowProfile
            .map { address, profile in
                NgramRowProfileEntry(
                    address: address,
                    requests: profile.requests,
                    cacheHits: profile.cacheHits,
                    cacheMisses: profile.cacheMisses,
                    readNanos: profile.readNanos)
            }
            .sorted {
                if $0.requests != $1.requests { return $0.requests > $1.requests }
                if $0.cacheHits != $1.cacheHits { return $0.cacheHits > $1.cacheHits }
                if $0.readNanos != $1.readNanos { return $0.readNanos > $1.readNanos }
                return $0.address < $1.address
            }
    }

    func read(addresses: [Int64]) throws -> [[Float]] {
        for address in addresses { recordProfileRequest(address: address) }
        return try addresses.map { try readRow(address: $0) }
    }

    func readAsync(addresses: [Int64], maxConcurrentReads: Int = 8) async throws -> [[Float]] {
        guard maxConcurrentReads > 0 else {
            throw StreamerError.invalidIOSplitConfiguration(
                "max concurrent n-gram reads must be positive")
        }
        guard !addresses.isEmpty else { return [] }
        for address in addresses { recordProfileRequest(address: address) }

        let uniqueAddresses = Set(addresses)
        guard rowCacheCapacityBytes > 0,
              uniqueAddresses.count <= rowCacheMaxUniqueRows else {
            if rowCacheCapacityBytes > 0 {
                recordCacheBypass()
            }
            return try await readRowsConcurrently(
                addresses: addresses, maxConcurrentReads: maxConcurrentReads)
        }

        var rows = Array<[Float]?>(repeating: nil, count: addresses.count)
        var missingAddresses: [Int64] = []
        var missingIndices: [Int64: [Int]] = [:]
        for (index, address) in addresses.enumerated() {
            if let indices = missingIndices[address] {
                missingIndices[address] = indices + [index]
            } else if let cached = cachedRow(address: address) {
                rows[index] = cached
            } else {
                missingAddresses.append(address)
                missingIndices[address] = [index]
            }
        }

        if !missingAddresses.isEmpty {
            let loadedRows = try await readRowsConcurrently(
                addresses: missingAddresses,
                maxConcurrentReads: maxConcurrentReads)
            for (address, row) in zip(missingAddresses, loadedRows) {
                storeCachedRow(address: address, values: row)
                for index in missingIndices[address] ?? [] {
                    rows[index] = row
                }
            }
        }

        return rows.map { $0! }
    }

    func readRow(address: Int64, usePinnedCache: Bool = true) throws -> [Float] {
        if usePinnedCache, pinnedRows.contains(address),
           let cached = cachedRow(address: address) {
            return cached
        }
        let location = try layout.locate(globalRow: address)
        let profileStart = DispatchTime.now().uptimeNanoseconds
        defer {
            recordProfileRead(
                address: address,
                nanos: DispatchTime.now().uptimeNanoseconds - profileStart)
        }
        let shard = layout.shard(location.shardIndex)
        let descriptor = fileDescriptors[location.shardIndex]
        let width = layout.rowWidth
        let packedBytes = width / 2
        let groups = width / layout.groupSize

        let packed = try readBytes(
            descriptor: descriptor,
            component: shard.weight,
            row: location.row,
            rowBytes: packedBytes,
            fileSize: shard.fileSize)
        let scaleBytes = try readBytes(
            descriptor: descriptor,
            component: shard.scales,
            row: location.row,
            rowBytes: groups * MemoryLayout<UInt16>.size,
            fileSize: shard.fileSize)
        let biasBytes = try readBytes(
            descriptor: descriptor,
            component: shard.biases,
            row: location.row,
            rowBytes: groups * MemoryLayout<UInt16>.size,
            fileSize: shard.fileSize)

        let row = Quantization.Int4AffineRow(
            packed: packed,
            scales: decodeUInt16(scaleBytes),
            biases: decodeUInt16(biasBytes))
        return Quantization.dequantizeInt4Affine(
            row, n: width, groupSize: layout.groupSize)
    }

    private func readRowsConcurrently(addresses: [Int64],
                                      maxConcurrentReads: Int) async throws -> [[Float]] {
        try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            var rows = Array(repeating: [Float](), count: addresses.count)
            var nextIndex = 0
            let initialCount = min(maxConcurrentReads, addresses.count)

            for index in 0..<initialCount {
                let address = addresses[index]
                group.addTask { [self] in
                    try Task.checkCancellation()
                    return (index, try self.readRow(address: address))
                }
                nextIndex += 1
            }

            while let (index, row) = try await group.next() {
                rows[index] = row
                guard nextIndex < addresses.count else { continue }
                let index = nextIndex
                let address = addresses[index]
                group.addTask { [self] in
                    try Task.checkCancellation()
                    return (index, try self.readRow(address: address))
                }
                nextIndex += 1
            }
            return rows
        }
    }

    private func cachedRow(address: Int64) -> [Float]? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard var entry = rowCache[address] else {
            cacheMisses += 1
            updateRowProfileLocked(address: address) { $0.cacheMisses += 1 }
            return nil
        }
        cacheClock &+= 1
        entry.lastUse = cacheClock
        rowCache[address] = entry
        cacheHits += 1
        updateRowProfileLocked(address: address) { $0.cacheHits += 1 }
        return entry.values
    }

    private func recordProfileRequest(address: Int64) {
        guard rowProfileMaxRows > 0 else { return }
        cacheLock.lock()
        updateRowProfileLocked(address: address) { $0.requests += 1 }
        cacheLock.unlock()
    }

    private func recordProfileRead(address: Int64, nanos: UInt64) {
        guard rowProfileMaxRows > 0 else { return }
        cacheLock.lock()
        updateRowProfileLocked(address: address) { $0.readNanos &+= nanos }
        cacheLock.unlock()
    }

    private func updateRowProfileLocked(address: Int64,
                                        update: (inout RowProfile) -> Void) {
        guard rowProfileMaxRows > 0 else { return }
        if rowProfile[address] == nil, rowProfile.count >= rowProfileMaxRows,
           let victim = rowProfile.min(by: {
               if $0.value.requests != $1.value.requests {
                   return $0.value.requests < $1.value.requests
               }
               return $0.key < $1.key
           }) {
            rowProfile.removeValue(forKey: victim.key)
        }
        var profile = rowProfile[address] ?? RowProfile()
        update(&profile)
        rowProfile[address] = profile
    }

    private func preloadPinnedRow(address: Int64) throws {
        let values = try readRow(address: address, usePinnedCache: false)
        let byteCount = values.count * MemoryLayout<Float>.stride
        cacheLock.lock()
        cacheClock &+= 1
        rowCache[address] = CachedRow(
            values: values, byteCount: byteCount, lastUse: cacheClock)
        cacheBytes += byteCount
        cachePeakBytes = max(cachePeakBytes, cacheBytes)
        cacheLock.unlock()
    }

    private func storeCachedRow(address: Int64, values: [Float]) {
        let byteCount = values.count * MemoryLayout<Float>.stride
        guard byteCount <= rowCacheCapacityBytes else {
            recordCacheBypass()
            return
        }

        cacheLock.lock()
        defer { cacheLock.unlock() }
        cacheClock &+= 1
        if let existing = rowCache.removeValue(forKey: address) {
            cacheBytes -= existing.byteCount
        }
        while unpinnedCacheBytesLocked + byteCount > rowCacheCapacityBytes,
              let victim = rowCache
                  .filter({ !pinnedRows.contains($0.key) })
                  .min(by: { $0.value.lastUse < $1.value.lastUse }) {
            rowCache.removeValue(forKey: victim.key)
            cacheBytes -= victim.value.byteCount
            cacheEvictions += 1
        }
        rowCache[address] = CachedRow(
            values: values, byteCount: byteCount, lastUse: cacheClock)
        cacheBytes += byteCount
        cachePeakBytes = max(cachePeakBytes, cacheBytes)
    }

    private var unpinnedCacheBytesLocked: Int {
        rowCache.reduce(0) { total, entry in
            total + (pinnedRows.contains(entry.key) ? 0 : entry.value.byteCount)
        }
    }

    private static func pinnedPayloadBytes(rowWidth: Int) -> Int {
        rowWidth * MemoryLayout<Float>.stride
    }

    private func recordCacheBypass() {
        cacheLock.lock()
        cacheBypasses += 1
        cacheLock.unlock()
    }

    private func readBytes(descriptor: Int32,
                           component: NgramComponentEntry,
                           row: UInt64,
                           rowBytes: Int,
                           fileSize: UInt64) throws -> [UInt8] {
        let (rowOffset, multiplyOverflow) = row.multipliedReportingOverflow(
            by: UInt64(rowBytes))
        let (offset, addOverflow) = component.offset.addingReportingOverflow(rowOffset)
        let (end, endOverflow) = offset.addingReportingOverflow(UInt64(rowBytes))
        let (componentEnd, componentOverflow) = component.offset.addingReportingOverflow(component.size)
        guard !multiplyOverflow, !addOverflow, !endOverflow, !componentOverflow,
              end <= componentEnd, end <= fileSize, offset <= UInt64(Int64.max) else {
            throw StreamerError.offsetOutOfRange(offset)
        }

        var bytes = [UInt8](repeating: 0, count: rowBytes)
        var completed = 0
        while completed < rowBytes {
            let count = bytes.withUnsafeMutableBytes { rawBuffer -> Int in
                guard let base = rawBuffer.baseAddress else { return 0 }
                return pread(
                    descriptor,
                    base.advanced(by: completed),
                    rowBytes - completed,
                    off_t(offset + UInt64(completed)))
            }
            guard count > 0 else {
                throw StreamerError.preadFailed(errno: count == 0 ? EIO : errno)
            }
            completed += count
        }
        return bytes
    }

    private func decodeUInt16(_ bytes: [UInt8]) -> [UInt16] {
        stride(from: 0, to: bytes.count, by: 2).map {
            UInt16(bytes[$0]) | (UInt16(bytes[$0 + 1]) << 8)
        }
    }
}
