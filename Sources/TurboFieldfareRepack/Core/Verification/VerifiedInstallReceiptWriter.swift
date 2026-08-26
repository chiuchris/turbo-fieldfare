import Darwin
import Foundation

struct VerifiedInstallFileIdentity: Equatable {
    let device: UInt64
    let inode: UInt64
    let generation: UInt32
    let size: UInt64
    let modificationTimeSeconds: Int64
    let modificationTimeNanoseconds: Int64
    let statusChangeTimeSeconds: Int64
    let statusChangeTimeNanoseconds: Int64

    static func capture(fileDescriptor fd: Int32, path: String) throws
        -> VerifiedInstallFileIdentity {
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size >= 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        return VerifiedInstallFileIdentity(
            device: UInt64(UInt32(bitPattern: st.st_dev)),
            inode: UInt64(st.st_ino),
            generation: UInt32(st.st_gen),
            size: UInt64(st.st_size),
            modificationTimeSeconds: Int64(st.st_mtimespec.tv_sec),
            modificationTimeNanoseconds: Int64(st.st_mtimespec.tv_nsec),
            statusChangeTimeSeconds: Int64(st.st_ctimespec.tv_sec),
            statusChangeTimeNanoseconds: Int64(st.st_ctimespec.tv_nsec))
    }

    var jsonObject: [String: Any] {
        [
            "device": device,
            "inode": inode,
            "generation": generation,
            "size": size,
            "modificationTimeSeconds": modificationTimeSeconds,
            "modificationTimeNanoseconds": modificationTimeNanoseconds,
            "statusChangeTimeSeconds": statusChangeTimeSeconds,
            "statusChangeTimeNanoseconds": statusChangeTimeNanoseconds,
        ]
    }
}

enum VerifiedInstallReceiptWriter {
    static let fileName = "verified-install.json"

    static func encode(outputDir: String,
                       manifestSha256: String,
                       manifestSize: UInt64,
                       manifestIdentity: VerifiedInstallFileIdentity? = nil,
                       fileIdentities: [String: VerifiedInstallFileIdentity] = [:],
                       sourceRepoID: String?,
                       sourceRevision: String?,
                       toolVersion: String = "TurboFieldfareRepack",
                       files: [RepackAudit.OutputFile]) throws -> Data {
        guard let manifestIdentity, manifestIdentity.size == manifestSize else {
            throw RepackError.configurationInvalid(
                detail: "verified receipt requires manifest identity")
        }
        let expectedPaths = Set(files.map(\.relativePath))
        guard expectedPaths.count == files.count,
              Set(fileIdentities.keys) == expectedPaths else {
            throw RepackError.configurationInvalid(
                detail: "verified receipt identity file set mismatch")
        }

        var filesDict: [String: Any] = [:]
        for file in files {
            guard let identity = fileIdentities[file.relativePath],
                  identity.size == file.size else {
                throw RepackError.configurationInvalid(
                    detail: "verified receipt identity mismatch for \(file.relativePath)")
            }
            filesDict[file.relativePath] = [
                "size": file.size,
                "sha256": file.sha256,
                "identity": identity.jsonObject,
            ]
        }
        filesDict["manifest.json"] = [
            "size": manifestSize,
            "sha256": manifestSha256,
            "identity": manifestIdentity.jsonObject,
        ]

        var receipt: [String: Any] = [
            "schemaVersion": 2,
            "manifestSha256": manifestSha256,
            "modelDirectoryPath": URL(fileURLWithPath: outputDir).standardizedFileURL.path,
            "verificationTimestamp": ISO8601DateFormatter().string(from: Date()),
            "toolVersion": toolVersion,
            "files": filesDict,
        ]
        if let sourceRepoID {
            receipt["sourceRepoID"] = sourceRepoID
        }
        if let sourceRevision {
            receipt["sourceRevision"] = sourceRevision
        }
        return try JSONSerialization.data(
            withJSONObject: receipt,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
