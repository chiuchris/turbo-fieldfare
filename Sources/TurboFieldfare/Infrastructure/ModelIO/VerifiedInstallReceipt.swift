import Foundation

public enum ModelIntegrityPolicy: Sendable, Equatable {
    case fullSha256
    case sizeCheckTrustedReceipt
}

public struct VerifiedInstallReceipt: Codable, Equatable, Sendable {
    public struct FileIdentity: Codable, Equatable, Sendable {
        public let device: UInt64
        public let inode: UInt64
        public let generation: UInt32
        public let size: UInt64
        public let modificationTimeSeconds: Int64
        public let modificationTimeNanoseconds: Int64
        public let statusChangeTimeSeconds: Int64
        public let statusChangeTimeNanoseconds: Int64

        public init(device: UInt64,
                    inode: UInt64,
                    generation: UInt32,
                    size: UInt64,
                    modificationTimeSeconds: Int64,
                    modificationTimeNanoseconds: Int64,
                    statusChangeTimeSeconds: Int64,
                    statusChangeTimeNanoseconds: Int64) {
            self.device = device
            self.inode = inode
            self.generation = generation
            self.size = size
            self.modificationTimeSeconds = modificationTimeSeconds
            self.modificationTimeNanoseconds = modificationTimeNanoseconds
            self.statusChangeTimeSeconds = statusChangeTimeSeconds
            self.statusChangeTimeNanoseconds = statusChangeTimeNanoseconds
        }
    }

    public struct FileEntry: Codable, Equatable, Sendable {
        public let size: UInt64
        public let sha256: String
        public let identity: FileIdentity?

        public init(size: UInt64, sha256: String, identity: FileIdentity? = nil) {
            self.size = size
            self.sha256 = sha256
            self.identity = identity
        }
    }

    public let schemaVersion: Int
    public let manifestSha256: String
    public let modelDirectoryPath: String
    public let sourceRepoID: String?
    public let sourceRevision: String?
    public let verificationTimestamp: String
    public let toolVersion: String
    public let files: [String: FileEntry]

    public init(schemaVersion: Int = 2,
                manifestSha256: String,
                modelDirectoryPath: String,
                sourceRepoID: String? = nil,
                sourceRevision: String? = nil,
                verificationTimestamp: String,
                toolVersion: String,
                files: [String: FileEntry]) {
        self.schemaVersion = schemaVersion
        self.manifestSha256 = manifestSha256
        self.modelDirectoryPath = modelDirectoryPath
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.verificationTimestamp = verificationTimestamp
        self.toolVersion = toolVersion
        self.files = files
    }
}

public enum VerifiedInstallReceiptReader {
    public static let fileName = "verified-install.json"
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    public static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> VerifiedInstallReceipt {
        do {
            let directory = try GTurboModelDirectory(rootURL: directoryURL)
            let data = try directory.readMetadata(fileName, maxBytes: maxBytes)
            return try decode(data: data)
        } catch ModelError.missingFile {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName) is missing")
        } catch let error as ModelError {
            if case .trustedReceiptInvalid = error { throw error }
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        } catch {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        }
    }

    package static func decode(data: Data) throws -> VerifiedInstallReceipt {
        do { return try JSONDecoder().decode(VerifiedInstallReceipt.self, from: data) }
        catch { throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)") }
    }

    public static func validate(_ receipt: VerifiedInstallReceipt,
                                directoryURL: URL,
                                manifest: Manifest,
                                manifestSha256: String,
                                manifestSize: UInt64) throws {
        try validateManifestBinding(receipt,
                                    directoryURL: directoryURL,
                                    manifestSha256: manifestSha256)
        var expectedFiles = Set(manifest.files.keys)
        expectedFiles.insert("manifest.json")
        let receiptFiles = Set(receipt.files.keys)
        guard receiptFiles == expectedFiles else {
            let missing = expectedFiles.subtracting(receiptFiles).sorted()
            let extra = receiptFiles.subtracting(expectedFiles).sorted()
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt file set mismatch missing=\(missing) extra=\(extra)")
        }
        guard let manifestReceiptEntry = receipt.files["manifest.json"] else {
            throw ModelError.trustedReceiptInvalid(detail: "receipt missing manifest.json")
        }
        guard manifestReceiptEntry.size == manifestSize else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json size mismatch")
        }
        guard manifestReceiptEntry.sha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json SHA mismatch")
        }
        try validateIdentity(manifestReceiptEntry, relativePath: "manifest.json")

        for (rel, manifestEntry) in manifest.files {
            guard let receiptEntry = receipt.files[rel] else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt missing \(rel)")
            }
            guard receiptEntry.size == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt size mismatch for \(rel)")
            }
            guard receiptEntry.sha256.lowercased() == manifestEntry.sha256.lowercased() else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt SHA mismatch for \(rel)")
            }
            try validateIdentity(receiptEntry, relativePath: rel)
        }
    }

    package static func validateCurrentFiles(_ receipt: VerifiedInstallReceipt,
                                             modelDirectory: GTurboModelDirectory) throws {
        for relativePath in receipt.files.keys.sorted() {
            guard let entry = receipt.files[relativePath] else { continue }
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                guard try currentIdentityMatches(
                    entry,
                    modelDirectory: modelDirectory,
                    fileDescriptor: fd,
                    relativePath: relativePath) else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "file identity mismatch for \(relativePath)")
                }
            } catch let error as ModelError {
                if case .trustedReceiptInvalid = error { throw error }
                throw ModelError.trustedReceiptInvalid(
                    detail: "file identity unavailable for \(relativePath): \(error)")
            }
        }
    }

    package static func currentIdentityMatches(
        _ entry: VerifiedInstallReceipt.FileEntry,
        modelDirectory: GTurboModelDirectory,
        fileDescriptor: Int32,
        relativePath: String
    ) throws -> Bool {
        guard let expected = entry.identity else { return false }
        let actual = try modelDirectory.fileIdentity(
            fileDescriptor: fileDescriptor,
            relativePath: relativePath)
        return actual == expected
    }

    public static func validateManifestBinding(_ receipt: VerifiedInstallReceipt,
                                               directoryURL: URL,
                                               manifestSha256: String) throws {
        guard receipt.schemaVersion == 2 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "unsupported schemaVersion \(receipt.schemaVersion)")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest SHA mismatch")
        }

        let actualPath = directoryURL.standardizedFileURL.path
        guard receipt.modelDirectoryPath == actualPath else {
            throw ModelError.trustedReceiptInvalid(detail: "model directory mismatch")
        }
    }

    private static func validateIdentity(_ entry: VerifiedInstallReceipt.FileEntry,
                                         relativePath: String) throws {
        guard let identity = entry.identity else {
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt missing file identity for \(relativePath)")
        }
        guard identity.size == entry.size else {
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt identity size mismatch for \(relativePath)")
        }
    }
}
