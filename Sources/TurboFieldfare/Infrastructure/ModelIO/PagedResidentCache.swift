import Foundation
import Darwin
import Metal

/// Tensor-granular LRU for the dense resident region. Entries own copied Metal
/// buffers so the full `model_weights.bin` resident mapping is unnecessary.
final class PagedResidentCache: @unchecked Sendable {
    struct Diagnostics: Sendable, Equatable {
        let currentBytes: UInt64
        let peakBytes: UInt64
        let capacityBytes: UInt64
        let hitCount: UInt64
        let missCount: UInt64
        let evictionCount: UInt64
        let bypassCount: UInt64
    }

    private struct Entry {
        let buffer: MTLBuffer
        let byteCount: UInt64
    }

    private let fileDescriptor: Int32
    private let residentFileOffset: UInt64
    private let residentSize: UInt64
    private let device: MTLDevice
    private let capacityBytes: UInt64
    private var entries: [String: Entry] = [:]
    private var lru: [String] = []
    private let lock = NSLock()
    private var currentBytes: UInt64 = 0
    private var peakBytes: UInt64 = 0
    private var hitCount: UInt64 = 0
    private var missCount: UInt64 = 0
    private var evictionCount: UInt64 = 0
    private var bypassCount: UInt64 = 0

    init(fileDescriptor: Int32,
         residentFileOffset: UInt64,
         residentSize: UInt64,
         capacityBytes: UInt64,
         device: MTLDevice) throws {
        guard capacityBytes > 0 else {
            throw ModelError.indexCorrupt(detail: "dense cache capacity must be positive")
        }
        self.fileDescriptor = fcntl(fileDescriptor, F_DUPFD_CLOEXEC, 0)
        guard self.fileDescriptor >= 0 else {
            throw ModelError.posixFailed(call: "dup(model_weights.bin)", errno: errno)
        }
        self.residentFileOffset = residentFileOffset
        self.residentSize = residentSize
        self.capacityBytes = capacityBytes
        self.device = device
    }

    deinit {
        close(fileDescriptor)
    }

    func tensor(name: String,
                entry: ResidentIndexEntry,
                relativeOffset: UInt64,
                scaleOffset: UInt64,
                biasOffset: UInt64) throws -> TensorView {
        lock.lock()
        defer { lock.unlock() }

        if let cached = entries[name] {
            hitCount += 1
            touch(name)
            let packedScaleOffset = entry.sizeBytes
            let packedBiasOffset = packedScaleOffset + entry.scaleSize
            return makeView(entry: entry, buffer: cached.buffer,
                            scaleOffset: packedScaleOffset,
                            biasOffset: packedBiasOffset)
        }

        missCount += 1
        let scaleLength = entry.scaleSize
        let biasLength = entry.biasSize
        let (scaleEnd, scaleOverflow) = scaleOffset.addingReportingOverflow(scaleLength)
        let (biasEnd, biasOverflow) = biasOffset.addingReportingOverflow(biasLength)
        let (payloadEnd, payloadOverflow) = relativeOffset.addingReportingOverflow(entry.sizeBytes)
        guard !scaleOverflow, !biasOverflow, !payloadOverflow,
              payloadEnd <= residentSize,
              scaleEnd <= residentSize,
              biasEnd <= residentSize else {
            throw ModelError.indexCorrupt(detail: "dense cache tensor exceeds resident payload")
        }

        let scaleBase = entry.sizeBytes
        let (biasBase, biasBaseOverflow) = scaleBase.addingReportingOverflow(scaleLength)
        let (totalBytes, totalOverflow) = biasBase.addingReportingOverflow(biasLength)
        guard !biasBaseOverflow, !totalOverflow,
              totalBytes <= UInt64(Int.max), totalBytes > 0 else {
            throw ModelError.indexCorrupt(detail: "dense cache tensor size is not addressable")
        }

        let buffer: MTLBuffer
        if totalBytes > capacityBytes {
            bypassCount += 1
            buffer = try loadBuffer(
                ranges: [(relativeOffset, entry.sizeBytes),
                         (scaleOffset, scaleLength),
                         (biasOffset, biasLength)],
                length: totalBytes)
        } else {
            buffer = try loadBuffer(
                ranges: [(relativeOffset, entry.sizeBytes),
                         (scaleOffset, scaleLength),
                         (biasOffset, biasLength)],
                length: totalBytes)
            while currentBytes + totalBytes > capacityBytes, let oldest = lru.first {
                lru.removeFirst()
                if let removed = entries.removeValue(forKey: oldest) {
                    currentBytes -= removed.byteCount
                    evictionCount += 1
                }
            }
            entries[name] = Entry(buffer: buffer, byteCount: totalBytes)
            lru.append(name)
            currentBytes += totalBytes
            peakBytes = max(peakBytes, currentBytes)
        }

        return makeView(entry: entry, buffer: buffer,
                        scaleOffset: scaleBase, biasOffset: biasBase)
    }

    var diagnostics: Diagnostics {
        lock.lock()
        defer { lock.unlock() }
        return Diagnostics(currentBytes: currentBytes,
                           peakBytes: peakBytes,
                           capacityBytes: capacityBytes,
                           hitCount: hitCount,
                           missCount: missCount,
                           evictionCount: evictionCount,
                           bypassCount: bypassCount)
    }

    private func touch(_ name: String) {
        lru.removeAll { $0 == name }
        lru.append(name)
    }

    private func makeView(entry: ResidentIndexEntry,
                          buffer: MTLBuffer,
                          scaleOffset: UInt64,
                          biasOffset: UInt64) -> TensorView {
        TensorView(buffer: buffer,
                    offset: 0,
                    length: entry.sizeBytes,
                    scaleOffset: scaleOffset,
                    scaleLength: entry.scaleSize,
                    biasOffset: biasOffset,
                    biasLength: entry.biasSize,
                    shape: entry.shape,
                    dtype: entry.dtype,
                    quantization: entry.quantization)
    }

    private func loadBuffer(ranges: [(UInt64, UInt64)], length: UInt64) throws -> MTLBuffer {
        let byteLength = Int(length)
        var bytes = [UInt8](repeating: 0, count: byteLength)
        var destinationOffset = 0
        for (relativeOffset, rangeLength) in ranges where rangeLength > 0 {
            guard relativeOffset <= residentSize,
                  rangeLength <= residentSize - relativeOffset,
                  destinationOffset <= byteLength - Int(rangeLength) else {
                throw ModelError.indexCorrupt(detail: "dense cache range exceeds resident payload")
            }
            var remaining = Int(rangeLength)
            var fileOffset = residentFileOffset + relativeOffset
            while remaining > 0 {
                let readCount = bytes.withUnsafeMutableBytes { rawBuffer in
                    pread(fileDescriptor,
                          rawBuffer.baseAddress!.advanced(by: destinationOffset),
                          remaining,
                          off_t(fileOffset))
                }
                guard readCount > 0 else {
                    throw ModelError.posixFailed(call: "pread(model_weights.bin)", errno: errno)
                }
                remaining -= readCount
                destinationOffset += readCount
                fileOffset += UInt64(readCount)
            }
        }
        guard let buffer = bytes.withUnsafeBytes({ rawBuffer in
            device.makeBuffer(bytes: rawBuffer.baseAddress!,
                              length: byteLength,
                              options: .storageModeShared)
        }) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buffer
    }
}
