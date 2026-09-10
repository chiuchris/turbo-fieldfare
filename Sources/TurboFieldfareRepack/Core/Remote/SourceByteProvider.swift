import Darwin
import Foundation

public protocol SourceByteProvider {
    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws
}

public final class HTTPRangeSourceByteProvider: SourceByteProvider {
    private final class ProgressAccumulator: @unchecked Sendable {
        private let lock = NSLock()
        private var workerBytes: [String: UInt64] = [:]
        private var totalBytes: UInt64 = 0
        private let progress: @Sendable (UInt64) -> Void

        init(progress: @escaping @Sendable (UInt64) -> Void) {
            self.progress = progress
        }

        func update(workerID: String, bytes: UInt64) {
            lock.lock()
            let previous = workerBytes[workerID, default: 0]
            if bytes > previous {
                totalBytes += bytes - previous
                workerBytes[workerID] = bytes
            }
            progress(totalBytes)
            lock.unlock()
        }
    }

    private struct WorkerRetry: Sendable {
        let label: String
        let attempt: Int
        let detail: String
    }

    private struct WorkerMetrics: Sendable {
        let sourceBytesRead: UInt64
        let outputBytesWritten: UInt64
        let intentionalCopyBytes: UInt64
        let byteCopyTiles: UInt64
        let largestScratchBytes: Int
        let remoteBytesDownloaded: UInt64
        let remoteRangeRequests: UInt64
        let remoteRangeRetries: UInt64
        let largestRemoteTransferBytes: Int
        let largestRemotePayloadHeapBytes: Int
        let remoteRangeStreamingSupported: Bool
        let remoteRetries: [WorkerRetry]
    }

    private struct WorkerResult: Sendable {
        let completed: RemoteCompletedRange
        let metrics: WorkerMetrics
    }

    private let remote: HuggingFaceRemoteSource
    private let files: [String: RemoteFileInfo]
    private let writeTileBytes: Int
    private let remoteConcurrency: Int

    public init(remote: HuggingFaceRemoteSource,
                files: [String: RemoteFileInfo],
                writeTileBytes: Int = WriterCore.tileBytes,
                remoteConcurrency: Int = 1) {
        self.remote = remote
        self.files = files
        self.writeTileBytes = writeTileBytes
        self.remoteConcurrency = remoteConcurrency
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
        let pending = copies.filter { !completedRangeIDs.contains($0.id) }
        guard !pending.isEmpty else { return }
        let workerCount = min(remoteConcurrency, pending.count)
        let remote = self.remote
        let files = self.files
        let writeTileBytes = self.writeTileBytes
        var nextIndex = workerCount
        var downloaded: UInt64 = 0
        let progressAccumulator = ProgressAccumulator(progress: progress)

        try await withThrowingTaskGroup(of: WorkerResult.self) { group in
            for copy in pending.prefix(workerCount) {
                group.addTask {
                    try await Self.runWorker(
                        copy: copy,
                        remote: remote,
                        files: files,
                        partialDirectory: partialDirectory,
                        temporaryPath: temporaryPath,
                        writeTileBytes: writeTileBytes,
                        progress: { bytes in
                            progressAccumulator.update(workerID: copy.id, bytes: bytes)
                        })
                }
            }

            while let result = try await group.next() {
                try Task.checkCancellation()
                downloaded += result.metrics.remoteBytesDownloaded
                Self.mergeWorkerMetrics(result.metrics, into: audit)
                progress(downloaded)
                try commit(result.completed)
                if nextIndex < pending.count {
                    let copy = pending[nextIndex]
                    nextIndex += 1
                    group.addTask {
                        try await Self.runWorker(
                            copy: copy,
                            remote: remote,
                            files: files,
                            partialDirectory: partialDirectory,
                            temporaryPath: temporaryPath,
                            writeTileBytes: writeTileBytes,
                            progress: { bytes in
                                progressAccumulator.update(workerID: copy.id, bytes: bytes)
                            })
                    }
                }
            }
        }
    }

    private static func mergeWorkerMetrics(
        _ metrics: WorkerMetrics,
        into audit: RepackAudit
    ) {
        audit.sourceBytesRead += metrics.sourceBytesRead
        audit.outputBytesWritten += metrics.outputBytesWritten
        audit.intentionalCopyBytes += metrics.intentionalCopyBytes
        audit.byteCopyTiles += metrics.byteCopyTiles
        audit.largestScratchBytes = max(
            audit.largestScratchBytes,
            metrics.largestScratchBytes)
        audit.remoteBytesDownloaded += metrics.remoteBytesDownloaded
        audit.remoteRangeRequests += metrics.remoteRangeRequests
        audit.remoteRangeRetries += metrics.remoteRangeRetries
        audit.largestRemoteTransferBytes = max(
            audit.largestRemoteTransferBytes,
            metrics.largestRemoteTransferBytes)
        audit.largestRemotePayloadHeapBytes = max(
            audit.largestRemotePayloadHeapBytes,
            metrics.largestRemotePayloadHeapBytes)
        audit.remoteRangeStreamingSupported =
            audit.remoteRangeStreamingSupported || metrics.remoteRangeStreamingSupported
        audit.remoteRetries.append(contentsOf: metrics.remoteRetries.map {
            RepackAudit.RemoteRetryRecord(
                label: $0.label,
                attempt: $0.attempt,
                detail: $0.detail)
        })
    }

    private static func runWorker(
        copy: CoalescedRangeCopy,
        remote: HuggingFaceRemoteSource,
        files: [String: RemoteFileInfo],
        partialDirectory: String,
        temporaryPath: String,
        writeTileBytes: Int,
        progress: @escaping @Sendable (UInt64) -> Void
    ) async throws -> WorkerResult {
        _ = temporaryPath
        guard let info = files[copy.shardID] else {
            throw RepackError.configurationInvalid(
                detail: "missing remote info for \(copy.shardID)")
        }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        let workerAudit = RepackAudit()
        workerAudit.largestScratchBytes = scratch.count
        let scatterWriter = try StreamingScatterWriter(
            copy: copy,
            partialDirectory: partialDirectory,
            writeTileBytes: writeTileBytes,
            audit: workerAudit)
        defer { scatterWriter.close() }
        let receivedBytes = try await remote.streamRange(
            filename: copy.shardID,
            info: info,
            offset: copy.sourceOffset,
            length: Int(copy.size),
            progress: progress,
            audit: workerAudit,
            receive: { data, relativeOffset in
                try scatterWriter.receive(data, relativeOffset: relativeOffset)
            })
        workerAudit.remoteRangeStreamingSupported = true
        workerAudit.remoteRangeRequests += 1
        workerAudit.remoteBytesDownloaded += receivedBytes
        workerAudit.largestRemoteTransferBytes = max(
            workerAudit.largestRemoteTransferBytes,
            Int(receivedBytes))

        try Task.checkCancellation()
        try scatterWriter.fsyncTouched()
        let digest = try scatterWriter.destinationDigest(scratch: scratch)
        return WorkerResult(
            completed: RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }),
            metrics: WorkerMetrics(
                sourceBytesRead: workerAudit.sourceBytesRead,
                outputBytesWritten: workerAudit.outputBytesWritten,
                intentionalCopyBytes: workerAudit.intentionalCopyBytes,
                byteCopyTiles: workerAudit.byteCopyTiles,
                largestScratchBytes: workerAudit.largestScratchBytes,
                remoteBytesDownloaded: workerAudit.remoteBytesDownloaded,
                remoteRangeRequests: workerAudit.remoteRangeRequests,
                remoteRangeRetries: workerAudit.remoteRangeRetries,
                largestRemoteTransferBytes: workerAudit.largestRemoteTransferBytes,
                largestRemotePayloadHeapBytes: workerAudit.largestRemotePayloadHeapBytes,
                remoteRangeStreamingSupported: workerAudit.remoteRangeStreamingSupported,
                remoteRetries: workerAudit.remoteRetries.map {
                    WorkerRetry(label: $0.label, attempt: $0.attempt, detail: $0.detail)
                }))
    }

    public static func destinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch suppliedScratch: UnsafeMutableRawBufferPointer? = nil
    ) throws -> String {
        try RepackOutputSupport.destinationDigest(
            copy,
            partialDirectory: partialDirectory,
            scratch: suppliedScratch)
    }

    private static func copyBytes(
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

private final class StreamingScatterWriter: @unchecked Sendable {
    private let copy: CoalescedRangeCopy
    private let partialDirectory: String
    private let audit: RepackAudit
    private var outputFDs: [String: Int32] = [:]
    private var touchedPaths: Set<String> = []
    private var incrementalDigest: IncrementalDestinationDigest?
    private let writeTileBytes: Int

    init(copy: CoalescedRangeCopy,
         partialDirectory: String,
         writeTileBytes: Int,
         audit: RepackAudit) throws {
        self.copy = copy
        self.partialDirectory = partialDirectory
        self.writeTileBytes = writeTileBytes
        self.audit = audit
        self.incrementalDigest = try RepackOutputSupport.incrementalDestinationDigest(
            copy,
            partialDirectory: partialDirectory)
        var opened: [String: Int32] = [:]
        do {
            for destination in copy.destinations {
                if opened[destination.destinationPath] == nil {
                    opened[destination.destinationPath] = try Posix.openExistingRW(
                        destination.destinationPath)
                }
            }
            self.outputFDs = opened
        } catch {
            opened.values.forEach { Darwin.close($0) }
            throw error
        }
    }

    func receive(_ data: Data, relativeOffset: UInt64) throws {
        guard !data.isEmpty else { return }
        let chunkStart = relativeOffset
        let chunkEnd = chunkStart + UInt64(data.count)
        for (destinationIndex, destination) in copy.destinations.enumerated() {
            let destinationStart = destination.sourceOffset - copy.sourceOffset
            let destinationEnd = destinationStart + destination.size
            let writeStart = max(chunkStart, destinationStart)
            let writeEnd = min(chunkEnd, destinationEnd)
            guard writeStart < writeEnd else { continue }
            guard let fd = outputFDs[destination.destinationPath] else {
                throw RepackError.fileOpenFailed(
                    path: destination.destinationPath,
                    errno: EBADF)
            }
            let sourceOffset = Int(writeStart - chunkStart)
            let destinationOffset = destination.destinationOffset +
                (writeStart - destinationStart)
            let size = Int(writeEnd - writeStart)
            var remaining = size
            var source = sourceOffset
            var output = destinationOffset
            while remaining > 0 {
                try Task.checkCancellation()
                let count = min(remaining, writeTileBytes)
                try data.withUnsafeBytes { raw in
                    guard let base = raw.baseAddress else { return }
                    try Posix.pwriteAll(
                        fd: fd,
                        path: destination.destinationPath,
                        buf: base.advanced(by: source),
                        count: count,
                        offset: output)
                    incrementalDigest?.append(
                        destinationIndex: destinationIndex,
                        destinationOffset: output,
                        bytes: UnsafeRawBufferPointer(
                            start: base.advanced(by: source),
                            count: count))
                }
                audit.recordTile(bytes: count)
                audit.recordRead(bytes: count)
                audit.recordWrite(bytes: count)
                remaining -= count
                source += count
                output += UInt64(count)
            }
            touchedPaths.insert(destination.destinationPath)
        }
    }

    func destinationDigest(scratch: UnsafeMutableRawBufferPointer) throws -> String {
        if let digest = incrementalDigest?.finalize() {
            return digest
        }
        return try RepackOutputSupport.destinationDigest(
            copy,
            partialDirectory: partialDirectory,
            scratch: scratch)
    }

    func fsyncTouched() throws {
        for path in touchedPaths {
            if let fd = outputFDs[path] {
                try Posix.fsync(fd, path: path)
            }
        }
    }

    func close() {
        outputFDs.values.forEach { Darwin.close($0) }
        outputFDs.removeAll(keepingCapacity: false)
    }
}
