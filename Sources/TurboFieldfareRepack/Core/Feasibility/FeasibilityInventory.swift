import Foundation

public enum FeasibilityTensorRole: String, Codable, Sendable {
    case textResident = "text_resident"
    case routedExpert = "routed_expert"
    case omittedMultimodal = "omitted_multimodal"
    case unsupported
}

public struct FeasibilityTensorRecord: Codable, Sendable, Equatable {
    public let name: String
    public let shard: String
    public let role: FeasibilityTensorRole
    public let dtype: String
    public let shape: [UInt64]
    public let offset: UInt64
    public let bytes: UInt64

    public init(name: String,
                shard: String,
                role: FeasibilityTensorRole,
                dtype: String,
                shape: [UInt64],
                offset: UInt64,
                bytes: UInt64) {
        self.name = name
        self.shard = shard
        self.role = role
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.bytes = bytes
    }
}

public struct FeasibilityInventorySummary: Codable, Sendable, Equatable {
    public let textResidentBytes: UInt64
    public let routedExpertBytes: UInt64
    public let omittedMultimodalBytes: UInt64
    public let unsupportedBytes: UInt64

    public var totalBytes: UInt64 {
        textResidentBytes + routedExpertBytes + omittedMultimodalBytes + unsupportedBytes
    }
}

public struct FeasibilityInventory: Codable, Sendable, Equatable {
    public let sourceIndexSHA256: String
    public let shardBytes: UInt64
    public let summary: FeasibilityInventorySummary
    public let tensors: [FeasibilityTensorRecord]
    public let sourceRepoID: String?
    public let requestedRevision: String?
    public let resolvedCommit: String?

    public static func scan(snapshotDirectory: String) throws -> FeasibilityInventory {
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        var records: [FeasibilityTensorRecord] = []
        var shardBytes: UInt64 = 0

        for shardName in metadata.shardFilenames {
            let shardPath = (snapshotDirectory as NSString).appendingPathComponent(shardName)
            let fd = try Posix.openRead(shardPath)
            defer { close(fd) }
            let fileSize = try Posix.fileSize(fd: fd, path: shardPath)
            shardBytes = try checkedAdd(shardBytes, fileSize, detail: "shard bytes")
            let header = try readHeader(path: shardPath, fd: fd, fileSize: fileSize)
            records.append(contentsOf: header.tensors.map { tensor in
                FeasibilityTensorRecord(
                    name: tensor.name,
                    shard: shardName,
                    role: role(for: tensor.name),
                    dtype: dtypeName(tensor.dtype),
                    shape: tensor.shape,
                    offset: tensor.absoluteOffset,
                    bytes: tensor.sizeBytes)
            })
        }

        records.sort { ($0.shard, $0.offset, $0.name) < ($1.shard, $1.offset, $1.name) }
        return FeasibilityInventory(sourceIndexSHA256: metadata.indexSha256Hex,
                                    shardBytes: shardBytes,
                                    summary: summary(for: records),
                                    tensors: records,
                                    sourceRepoID: nil,
                                    requestedRevision: nil,
                                    resolvedCommit: nil)
    }

    public static func scanRemote(repoID: String,
                                  revision: String,
                                  token: String? = nil,
                                  workingDirectory: String) async throws -> FeasibilityInventory {
        let metadataDirectory = (workingDirectory as NSString)
            .appendingPathComponent("metadata")
        let remote = HuggingFaceRemoteSource(
            repoID: repoID,
            requestedRevision: revision,
            token: token,
            tempDirectory: (workingDirectory as NSString).appendingPathComponent("ranges"))
        let snapshot = try await RemoteSnapshotLoader.loadHeaders(
            remote: remote,
            requireKnownSource: false,
            metadataDirectory: metadataDirectory)
        var records: [FeasibilityTensorRecord] = []
        var shardBytes: UInt64 = 0
        for (index, shardName) in snapshot.metadata.shardFilenames.enumerated() {
            guard index < snapshot.shardHeaders.count,
                  let info = snapshot.remoteFiles[shardName] else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "snapshot is missing shard metadata for \(shardName)")
            }
            shardBytes = try checkedAdd(shardBytes, info.size, detail: "remote shard bytes")
            records.append(contentsOf: snapshot.shardHeaders[index].tensors.map { tensor in
                FeasibilityTensorRecord(
                    name: tensor.name,
                    shard: shardName,
                    role: role(for: tensor.name),
                    dtype: dtypeName(tensor.dtype),
                    shape: tensor.shape,
                    offset: tensor.absoluteOffset,
                    bytes: tensor.sizeBytes)
            })
        }
        records.sort { ($0.shard, $0.offset, $0.name) < ($1.shard, $1.offset, $1.name) }
        return FeasibilityInventory(sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
                                    shardBytes: shardBytes,
                                    summary: summary(for: records),
                                    tensors: records,
                                    sourceRepoID: repoID,
                                    requestedRevision: revision,
                                    resolvedCommit: snapshot.resolvedCommit)
    }

    private init(sourceIndexSHA256: String,
                 shardBytes: UInt64,
                 summary: FeasibilityInventorySummary,
                 tensors: [FeasibilityTensorRecord],
                 sourceRepoID: String?,
                 requestedRevision: String?,
                 resolvedCommit: String?) {
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.shardBytes = shardBytes
        self.summary = summary
        self.tensors = tensors
        self.sourceRepoID = sourceRepoID
        self.requestedRevision = requestedRevision
        self.resolvedCommit = resolvedCommit
    }

    private static func readHeader(path: String,
                                   fd: Int32,
                                   fileSize: UInt64) throws -> Safetensors.Header {
        var prefix = Data(count: 8)
        try prefix.withUnsafeMutableBytes { raw in
            try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!, count: 8, offset: 0)
        }
        let headerBytes = prefix.withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for index in 0..<8 {
                value |= UInt64(raw[index]) << UInt64(index * 8)
            }
            return value
        }
        guard headerBytes <= UInt64(Int.max) else {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerBytes)
        }
        guard headerBytes > 0 else {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "empty header")
        }
        var data = Data(count: Int(headerBytes))
        try data.withUnsafeMutableBytes { raw in
            try Posix.preadAll(fd: fd,
                               path: path,
                               buf: raw.baseAddress!,
                               count: raw.count,
                               offset: 8)
        }
        return try Safetensors.parseHeaderBytes(path: path,
                                                fileSize: fileSize,
                                                headerBytes: data)
    }

    static func role(for name: String) -> FeasibilityTensorRole {
        let lowercased = name.lowercased()
        if lowercased.contains("vision") || lowercased.contains("visual") ||
            lowercased.contains("audio") || lowercased.contains("mtp") {
            return .omittedMultimodal
        }
        if lowercased.contains(".experts.") || lowercased.contains("switch_mlp") ||
            lowercased.contains("switch_glu") {
            return .routedExpert
        }
        if lowercased.hasPrefix("language_model.") || lowercased.hasPrefix("model.") ||
            lowercased.contains("embed_tokens") || lowercased.contains("lm_head") {
            return .textResident
        }
        return .unsupported
    }

    private static func dtypeName(_ dtype: SourceTensor.Dtype) -> String {
        switch dtype {
        case .u32: return "U32"
        case .bf16: return "BF16"
        case .fp16: return "F16"
        case .fp32: return "F32"
        }
    }

    private static func bytes(for role: FeasibilityTensorRole,
                              records: [FeasibilityTensorRecord]) -> UInt64 {
        records.filter { $0.role == role }.reduce(0) { $0 + $1.bytes }
    }

    private static func summary(for records: [FeasibilityTensorRecord]) -> FeasibilityInventorySummary {
        FeasibilityInventorySummary(
            textResidentBytes: bytes(for: .textResident, records: records),
            routedExpertBytes: bytes(for: .routedExpert, records: records),
            omittedMultimodalBytes: bytes(for: .omittedMultimodal, records: records),
            unsupportedBytes: bytes(for: .unsupported, records: records))
    }

    private static func checkedAdd(_ lhs: UInt64,
                                   _ rhs: UInt64,
                                   detail: String) throws -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw RepackError.configurationInvalid(detail: "overflow while summing \(detail)")
        }
        return value
    }
}