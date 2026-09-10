import Foundation

struct RepackOutputSupport {
    static func destinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch suppliedScratch: UnsafeMutableRawBufferPointer? = nil
    ) throws -> String {
        let scratch = suppliedScratch ?? UnsafeMutableRawBufferPointer.allocate(
            byteCount: WriterCore.tileBytes,
            alignment: 16_384)
        defer {
            if suppliedScratch == nil { scratch.deallocate() }
        }

        var digest = OutputDestinationDigest(copy: copy)
        for destination in copy.destinations {
            digest.append(try RangeCopyPlanner.normalizedRelativePath(
                destination.destinationPath,
                root: partialDirectory))
            digest.append(destination.destinationOffset)
            digest.append(destination.sourceOffset - copy.sourceOffset)
            digest.append(destination.size)

            let descriptor = try Posix.openReadNoFollow(destination.destinationPath)
            defer { close(descriptor) }
            var remaining = destination.size
            var offset = destination.destinationOffset
            while remaining > 0 {
                let count = min(Int(remaining), scratch.count)
                try Posix.preadAll(
                    fd: descriptor,
                    path: destination.destinationPath,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: offset)
                digest.append(UnsafeRawBufferPointer(
                    start: scratch.baseAddress,
                    count: count))
                remaining -= UInt64(count)
                offset += UInt64(count)
            }
        }
        return digest.finalize()
    }

    static func incrementalDestinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String
    ) throws -> IncrementalDestinationDigest? {
        try IncrementalDestinationDigest(
            copy: copy,
            partialDirectory: partialDirectory)
    }
}

struct IncrementalDestinationDigest {
    private var digest: OutputDestinationDigest
    private let destinations: [RangeCopy]
    private let partialDirectory: String
    private let copySourceOffset: UInt64
    private var nextIndex = 0
    private var nextOffset: UInt64 = 0
    private var valid = true

    init?(copy: CoalescedRangeCopy,
          partialDirectory: String) throws {
        var previousEnd: UInt64 = 0
        for (index, destination) in copy.destinations.enumerated() {
            let sourceStart = destination.sourceOffset - copy.sourceOffset
            guard index == 0 || sourceStart >= previousEnd else {
                return nil
            }
            previousEnd = sourceStart + destination.size
        }
        self.digest = OutputDestinationDigest(copy: copy)
        self.destinations = copy.destinations
        self.partialDirectory = partialDirectory
        self.copySourceOffset = copy.sourceOffset
        if let first = destinations.first {
            self.nextOffset = first.destinationOffset
        }
    }

    mutating func append(destinationIndex: Int,
                         destinationOffset: UInt64,
                         bytes: UnsafeRawBufferPointer) {
        guard valid, !bytes.isEmpty else { return }
        guard destinationIndex == nextIndex,
              destinationIndex < destinations.count,
              destinationOffset == nextOffset else {
            valid = false
            return
        }
        let destination = destinations[destinationIndex]
        if destinationOffset == destination.destinationOffset {
            do {
                try digest.appendMetadata(
                    destination,
                    relativeSourceOffset: destination.sourceOffset - copySourceOffset,
                    partialDirectory: partialDirectory)
            } catch {
                valid = false
                return
            }
        }
        digest.append(bytes)
        nextOffset += UInt64(bytes.count)
        let end = destination.destinationOffset + destination.size
        if nextOffset == end {
            nextIndex += 1
            if nextIndex < destinations.count {
                nextOffset = destinations[nextIndex].destinationOffset
            }
        } else if nextOffset > end {
            valid = false
        }
    }

    func finalize() -> String? {
        guard valid, nextIndex == destinations.count else { return nil }
        return digest.finalize()
    }
}

private struct OutputDestinationDigest {
    private var stream = Sha256Stream()

    init(copy: CoalescedRangeCopy) {
        append("TurboFieldfare.RemoteRangeDestination.v1")
        append(copy.id)
        append(UInt64(copy.destinations.count))
    }

    mutating func appendMetadata(_ destination: RangeCopy,
                                 relativeSourceOffset: UInt64,
                                 partialDirectory: String) throws {
        append(try RangeCopyPlanner.normalizedRelativePath(
            destination.destinationPath,
            root: partialDirectory))
        append(destination.destinationOffset)
        append(relativeSourceOffset)
        append(destination.size)
    }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    mutating func append(_ bytes: UnsafeRawBufferPointer) {
        stream.update(bytes)
    }

    func finalize() -> String {
        stream.finalizeHexString()
    }
}
