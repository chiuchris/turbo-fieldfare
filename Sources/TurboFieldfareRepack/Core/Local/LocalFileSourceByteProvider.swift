import Darwin
import Foundation

public final class LocalFileSourceByteProvider: SourceByteProvider {
    private let writeTileBytes: Int

    public init(writeTileBytes: Int = WriterCore.tileBytes) {
        self.writeTileBytes = writeTileBytes
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        _ = temporaryPath
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var sourceFDs: [String: Int32] = [:]
        var outputFDs: [String: Int32] = [:]
        defer {
            sourceFDs.values.forEach { close($0) }
            outputFDs.values.forEach { close($0) }
        }
        var copied: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            let sourceFD: Int32
            if let existing = sourceFDs[copy.shardID] {
                sourceFD = existing
            } else {
                sourceFD = try Posix.openReadNoFollow(copy.shardID)
                sourceFDs[copy.shardID] = sourceFD
            }
            var touched = Set<String>()
            for destination in copy.destinations {
                let destinationFD: Int32
                if let existing = outputFDs[destination.destinationPath] {
                    destinationFD = existing
                } else {
                    destinationFD = try Posix.openExistingRW(
                        destination.destinationPath)
                    outputFDs[destination.destinationPath] = destinationFD
                }
                touched.insert(destination.destinationPath)
                try copyBytes(
                    sourceFD: sourceFD,
                    sourcePath: copy.shardID,
                    destinationFD: destinationFD,
                    destinationPath: destination.destinationPath,
                    sourceOffset: destination.sourceOffset,
                    destinationOffset: destination.destinationOffset,
                    size: destination.size,
                    scratch: scratch,
                    audit: audit)
            }

            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try RepackOutputSupport.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            copied += copy.size
            progress(copied)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) {
                    $0 + $1.size
                }))
            try Task.checkCancellation()
        }
    }

    private func copyBytes(
        sourceFD: Int32,
        sourcePath: String,
        destinationFD: Int32,
        destinationPath: String,
        sourceOffset: UInt64,
        destinationOffset: UInt64,
        size: UInt64,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(
                fd: sourceFD,
                path: sourcePath,
                buf: scratch.baseAddress!,
                count: count,
                offset: source)
            try Posix.pwriteAll(
                fd: destinationFD,
                path: destinationPath,
                buf: scratch.baseAddress!,
                count: count,
                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }
}
