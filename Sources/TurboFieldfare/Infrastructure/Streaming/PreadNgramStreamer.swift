import Darwin
import Foundation

final class PreadNgramStreamer: @unchecked Sendable {
    let layout: PackedNgramsLayout

    private let fileDescriptors: [Int32]

    init(directoryURL: URL, layout: PackedNgramsLayout) throws {
        guard !layout.shards.isEmpty,
              layout.groupSize > 0,
              layout.rowWidth > 0,
              layout.rowWidth % layout.groupSize == 0 else {
            throw StreamerError.invalidIOSplitConfiguration(
                "invalid packed n-gram row geometry")
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
    }

    deinit {
        for descriptor in fileDescriptors { close(descriptor) }
    }

    func read(addresses: [Int64]) throws -> [[Float]] {
        try addresses.map(readRow)
    }

    func readRow(address: Int64) throws -> [Float] {
        let location = try layout.locate(globalRow: address)
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
