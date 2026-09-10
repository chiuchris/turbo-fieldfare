import Foundation

struct RemoteSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let ngramHeader: Safetensors.Header?
    let mtpHeader: Safetensors.Header?
    let remoteFiles: [String: RemoteFileInfo]
    let resolvedCommit: String
    let metadataDirectory: String
}

struct RemoteHeaderSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let shardHeaders: [Safetensors.Header]
    let ngramHeader: Safetensors.Header?
    let mtpHeader: Safetensors.Header?
    let remoteFiles: [String: RemoteFileInfo]
    let resolvedCommit: String
    let metadataDirectory: String
}

enum RemoteSnapshotLoader {
    static func load(remote: HuggingFaceRemoteSource,
                     requireKnownSource: Bool,
                     metadataDirectory: String,
                     audit: RepackAudit? = nil) async throws -> RemoteSnapshot {
        let headers = try await loadHeaders(remote: remote,
                                             requireKnownSource: requireKnownSource,
                                             metadataDirectory: metadataDirectory,
                                             audit: audit)
        let arch = try ArchInfo.load(configPath: headers.metadata.configPath)
        return RemoteSnapshot(metadata: headers.metadata,
                              arch: arch,
                              shardHeaders: headers.shardHeaders,
                              ngramHeader: headers.ngramHeader,
                              mtpHeader: headers.mtpHeader,
                              remoteFiles: headers.remoteFiles,
                              resolvedCommit: headers.resolvedCommit,
                              metadataDirectory: headers.metadataDirectory)
    }

    static func loadHeaders(remote: HuggingFaceRemoteSource,
                            requireKnownSource: Bool,
                            metadataDirectory: String,
                            audit: RepackAudit? = nil) async throws -> RemoteHeaderSnapshot {
        try Posix.mkdirP(metadataDirectory)

        let indexInfo = try await remote.resolveFileInfo(filename: "model.safetensors.index.json",
                                                         audit: audit)
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)
        let configInfo = try await pinned.resolveFileInfo(filename: "config.json",
                                                          audit: audit)
        guard configInfo.resolvedCommit == indexInfo.resolvedCommit else {
            throw RepackError.remoteProtocolInvalid(detail: "config commit differs from index commit")
        }

        try await pinned.fetchSmallFile(filename: "model.safetensors.index.json",
                                        info: indexInfo,
                                        capBytes: 4 * 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("model.safetensors.index.json"),
                                        audit: audit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        let arch = try ArchInfo.load(configPath: metadata.configPath)
        if requireKnownSource && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        var files: [String: RemoteFileInfo] = [
            indexInfo.filename: indexInfo,
            configInfo.filename: configInfo,
        ]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) does not advertise byte ranges")
            }
            files[shard] = info
            headers.append(try await loadHeader(remote: pinned,
                                                filename: shard,
                                                info: info,
                                                audit: audit))
        }

        let ngramHeader: Safetensors.Header?
        if let qwen38 = arch.qwen38, qwen38.ngramSidecar {
            guard let filename = qwen38.ngramFile, !filename.isEmpty else {
                throw RepackError.configJsonInvalid(
                    path: metadata.configPath,
                    detail: "ngram sidecar is enabled but ngram_file is missing")
            }
            let info = try await pinned.resolveFileInfo(filename: filename, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "n-gram sidecar commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "n-gram sidecar does not advertise byte ranges")
            }
            files[filename] = info
            ngramHeader = try await loadHeader(remote: pinned,
                                               filename: filename,
                                               info: info,
                                               audit: audit)
        } else {
            ngramHeader = nil
        }

        let mtpHeader: Safetensors.Header?
        if let qwen38 = arch.qwen38,
           let filename = qwen38.mtpFile,
           !filename.isEmpty {
            let info = try await pinned.resolveFileInfo(filename: filename, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "MTP sidecar commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "MTP sidecar does not advertise byte ranges")
            }
            files[filename] = info
            mtpHeader = try await loadHeader(remote: pinned,
                                             filename: filename,
                                             info: info,
                                             audit: audit)
        } else {
            mtpHeader = nil
        }

        return RemoteHeaderSnapshot(metadata: metadata,
                                    shardHeaders: headers,
                                    ngramHeader: ngramHeader,
                                    mtpHeader: mtpHeader,
                                    remoteFiles: files,
                                    resolvedCommit: indexInfo.resolvedCommit,
                                    metadataDirectory: metadataDirectory)
    }

    private static func loadHeader(remote: HuggingFaceRemoteSource,
                                   filename: String,
                                   info: RemoteFileInfo,
                                   audit: RepackAudit?) async throws -> Safetensors.Header {
        let prefix = try await remote.downloadRangeToTempFile(filename: filename,
                                                               info: info,
                                                               offset: 0,
                                                               length: 8,
                                                               audit: audit)
        defer { try? FileManager.default.removeItem(atPath: prefix.path) }
        let prefixData = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
        guard prefixData.count == 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: filename,
                                                       detail: "short header prefix")
        }
        let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for index in 0..<8 {
                value |= UInt64(raw[index]) << UInt64(index * 8)
            }
            return value
        }
        guard info.size >= 8 else {
            throw RepackError.safetensorsHeaderInvalid(
                path: filename, detail: "remote reports \(info.size) bytes")
        }
        if headerSize > Safetensors.maxHeaderBytes || headerSize > info.size - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: filename, size: headerSize)
        }
        let headerFile = try await remote.downloadRangeToTempFile(filename: filename,
                                                                   info: info,
                                                                   offset: 8,
                                                                   length: Int(headerSize),
                                                                   audit: audit)
        defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
        let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
        return try Safetensors.parseHeaderBytes(path: filename,
                                                fileSize: info.size,
                                                headerBytes: headerData)
    }
}
