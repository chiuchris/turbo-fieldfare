import Foundation
import TurboFieldfareFormat

public struct LocalSnapshotRepackOptions: Sendable {
    public let snapshotDirectory: String
    public let outputDirectory: String
    public let overwrite: Bool
    public let copyAuditPath: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let residentConcurrency: Int
    public let minFreeReserveBytes: UInt64

    public init(snapshotDirectory: String,
                outputDirectory: String,
                overwrite: Bool = false,
                copyAuditPath: String? = nil,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                residentConcurrency: Int = 1,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) {
        self.snapshotDirectory = snapshotDirectory
        self.outputDirectory = outputDirectory
        self.overwrite = overwrite
        self.copyAuditPath = copyAuditPath
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.residentConcurrency = residentConcurrency
        self.minFreeReserveBytes = minFreeReserveBytes
    }
}

public struct LocalSnapshotRepackResult: Sendable {
    public let outputDirectory: String
    public let sourceIndexSHA256: String
    public let sourceBytesCopied: UInt64
    public let outputBytes: UInt64
}

public final class LocalSnapshotRepacker {
    private let options: LocalSnapshotRepackOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: LocalSnapshotRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> LocalSnapshotRepackResult {
        try validateOptions()
        let source = URL(fileURLWithPath: options.snapshotDirectory,
                         isDirectory: true).standardizedFileURL.path
        guard try Posix.entryKind(source) == .directory else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot is not a directory: \(source)")
        }
        let installPaths = try RemoteInstallPaths(
            outputDirectory: options.outputDirectory)
        let metadata = try IndexLoader.load(snapshotDir: source)
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for filename in metadata.shardFilenames {
            let path = (source as NSString).appendingPathComponent(filename)
            guard try Posix.entryKind(path) == .regular else {
                throw RepackError.configurationInvalid(
                    detail: "local snapshot shard is not a regular file: \(path)")
            }
            headers.append(try Safetensors.parseHeader(path: path))
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)
        let ngramHeader: Safetensors.Header?
        if let qwen38 = arch.qwen38, qwen38.ngramSidecar {
            guard let filename = qwen38.ngramFile, !filename.isEmpty else {
                throw RepackError.configJsonInvalid(
                    path: metadata.configPath,
                    detail: "ngram sidecar is enabled but ngram_file is missing")
            }
            let path = (source as NSString).appendingPathComponent(filename)
            guard try Posix.entryKind(path) == .regular else {
                throw RepackError.configurationInvalid(
                    detail: "local n-gram sidecar is not a regular file: \(path)")
            }
            ngramHeader = try Safetensors.parseHeader(path: path)
        } else {
            ngramHeader = nil
        }
        let mtpHeader: Safetensors.Header?
        if let qwen38 = arch.qwen38,
           let filename = qwen38.mtpFile,
           !filename.isEmpty {
            let path = (source as NSString).appendingPathComponent(filename)
            guard try Posix.entryKind(path) == .regular else {
                throw RepackError.configurationInvalid(
                    detail: "local MTP sidecar is not a regular file: \(path)")
            }
            mtpHeader = try Safetensors.parseHeader(path: path)
        } else {
            mtpHeader = nil
        }

        progress(.downloadingMetadata)
        let plan = try RepackPlanner.plan(meta: metadata,
                                          arch: arch,
                                          shardHeaders: headers,
                                          outputDir: installPaths.partialDirectory,
                                          ngramHeader: ngramHeader,
                                          mtpHeader: mtpHeader)
        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: plan,
            rangeChunkBytes: options.rangeChunkBytes)
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + plan.ngramShards.reduce(UInt64(0)) { $0 + $1.fileSize }
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: installPaths.parentDirectory,
            bytes: outputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        progress(.checkingDisk(diskRequirement))

        let lock = try InstallLock.acquire(outputDirectory: options.outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            guard options.overwrite else {
                throw RepackError.configurationInvalid(
                    detail: "output directory already exists: \(paths.finalDirectory)")
            }
            try FileManager.default.removeItem(atPath: paths.finalDirectory)
        }
        if try Posix.entryKind(paths.partialDirectory) != .absent {
            guard options.overwrite else {
                throw RepackError.configurationInvalid(
                    detail: "partial output exists: \(paths.partialDirectory)")
            }
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
        }
        try Posix.mkdirP(paths.partialDirectory)
        do {
            try createOutputFiles(plan: plan, paths: paths, rangePlan: rangePlan)
            let provider = LocalFileSourceByteProvider(
                writeTileBytes: options.writeTileBytes)
            progress(.copyingPayload(
                reusedBytes: 0,
                downloadedThisRunBytes: 0,
                totalBytes: rangePlan.remoteBytesToDownload))
            try await provider.copyBatch(
                rangePlan.coalescedCopies,
                completedRangeIDs: [],
                partialDirectory: paths.partialDirectory,
                temporaryPath: paths.rangeTemporaryFile,
                audit: audit,
                progress: { copiedBytes in
                    progress(.copyingPayload(
                        reusedBytes: 0,
                        downloadedThisRunBytes: copiedBytes,
                        totalBytes: rangePlan.remoteBytesToDownload))
                },
                commit: { _ in })

            try await ResidentWriter.convertStagedEntries(
                plan: plan.resident,
                audit: audit,
                residentConcurrency: options.residentConcurrency)
            try removeStagedSourceFiles(plan: plan, rangePlan: rangePlan)
            try recordOutputFile(relativePath: "model_weights.bin",
                                 path: plan.resident.path,
                                 progress: progress)
            for layer in plan.layers where layer.expertsPerLayer > 0 {
                let relativePath = "packed_experts/" +
                    (layer.path as NSString).lastPathComponent
                try recordOutputFile(relativePath: relativePath,
                                     path: layer.path,
                                     progress: progress)
            }
            for shard in plan.ngramShards {
                let relativePath = "packed_ngrams/" +
                    (shard.path as NSString).lastPathComponent
                try recordOutputFile(relativePath: relativePath,
                                     path: shard.path,
                                     progress: progress)
            }

            let expertStride = plan.layers.first(
                where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
            let layoutPath = (paths.partialDirectory as NSString)
                .appendingPathComponent("packed_experts/layout.json")
            let layoutData = try GTurboJSON.encodeLayout(
                plan: plan, expertStride: expertStride)
            try writeSmall(path: layoutPath, data: layoutData)
            try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
            try recordOutputFile(relativePath: "packed_experts/layout.json",
                                 path: layoutPath,
                                 progress: progress)

            if !plan.ngramShards.isEmpty {
                let ngramLayoutPath = (paths.partialDirectory as NSString)
                    .appendingPathComponent("packed_ngrams/layout.json")
                let ngramData = try GTurboJSON.encodeNgramLayout(plan: plan)
                try writeSmall(path: ngramLayoutPath, data: ngramData)
                _ = try GTurboPackedNgramsLayoutCodec.decode(ngramData)
                try recordOutputFile(relativePath: "packed_ngrams/layout.json",
                                     path: ngramLayoutPath,
                                     progress: progress)
            }

            try copyLocalMetadataSidecars(
                sourceDirectory: source,
                partialDirectory: paths.partialDirectory,
                progress: progress)
            progress(.finalizing)
            try writeManifest(plan: plan,
                              partialDirectory: paths.partialDirectory,
                              metadata: metadata,
                              expertStride: expertStride)
            try Posix.fsyncDirectory(paths.partialDirectory)
            try Posix.rename(from: paths.partialDirectory,
                             to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        } catch {
            throw error
        }

        audit.sourceSnapshotSha256 = metadata.indexSha256Hex
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDirectory)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }
        let verification = try VerifiedInstallTool.run(options: VerifyInstallOptions(
            inputGTurbo: paths.finalDirectory,
            receiptModelDirectoryPath: paths.finalDirectory,
            receiptSourceRevision: "local:" + metadata.indexSha256Hex))
        return LocalSnapshotRepackResult(
            outputDirectory: paths.finalDirectory,
            sourceIndexSHA256: metadata.indexSha256Hex,
            sourceBytesCopied: rangePlan.remoteBytesToDownload,
            outputBytes: verification.bytesVerified)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(
                detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(
                detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard (1...8).contains(options.residentConcurrency) else {
            throw RepackError.configurationInvalid(
                detail: "bad resident concurrency \(options.residentConcurrency)")
        }
    }

    private func createOutputFiles(plan: RepackPlan,
                                   paths: RemoteInstallPaths,
                                   rangePlan: RangeCopyPlan) throws {
        try Posix.mkdirP((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts"))
        for stagedFile in rangePlan.stagedSourceFiles {
            let path = (paths.partialDirectory as NSString)
                .appendingPathComponent(stagedFile.relativePath)
            try Posix.mkdirP((path as NSString).deletingLastPathComponent)
            let descriptor = try Posix.openCreateRW(path)
            try Posix.preallocate(descriptor, path: path, size: stagedFile.size)
            try Posix.fsync(descriptor, path: path)
            close(descriptor)
        }
        if !plan.ngramShards.isEmpty {
            try Posix.mkdirP((paths.partialDirectory as NSString)
                .appendingPathComponent("packed_ngrams"))
        }
        let resident = try ResidentWriter.createAndWriteIndex(
            plan: plan.resident,
            audit: audit)
        try Posix.fsync(resident, path: plan.resident.path)
        close(resident)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            let descriptor = try Posix.openCreateRW(layer.path)
            try Posix.preallocate(descriptor, path: layer.path, size: layer.fileSize)
            try Posix.fsync(descriptor, path: layer.path)
            close(descriptor)
        }
        for shard in plan.ngramShards {
            let descriptor = try Posix.openCreateRW(shard.path)
            try Posix.preallocate(descriptor, path: shard.path, size: shard.fileSize)
            try Posix.fsync(descriptor, path: shard.path)
            close(descriptor)
        }
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        let descriptor = try Posix.openRead(path)
        defer { close(descriptor) }
        let size = try Posix.fileSize(fd: descriptor, path: path)
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath,
                                       size: size,
                                       sha256: sha))
    }

    private func removeStagedSourceFiles(plan: RepackPlan,
                                         rangePlan: RangeCopyPlan) throws {
        let root = (plan.resident.path as NSString).deletingLastPathComponent
        for stagedFile in rangePlan.stagedSourceFiles {
            let path = (root as NSString).appendingPathComponent(stagedFile.relativePath)
            if try Posix.entryKind(path) != .absent {
                try FileManager.default.removeItem(atPath: path)
            }
        }
        let directory = (root as NSString).appendingPathComponent("source-staging")
        if try Posix.entryKind(directory) != .absent {
            try FileManager.default.removeItem(atPath: directory)
        }
    }

    private func copyLocalMetadataSidecars(
        sourceDirectory: String,
        partialDirectory: String,
        progress: @Sendable (ModelInstallProgress) -> Void
    ) throws {
        let names = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "chat_template.json",
        ]
        let tokenizerDirectory = (partialDirectory as NSString)
            .appendingPathComponent("tokenizer")
        for name in names {
            try Task.checkCancellation()
            let source = (sourceDirectory as NSString).appendingPathComponent(name)
            guard try Posix.entryKind(source) == .regular else { continue }
            let destination = (tokenizerDirectory as NSString)
                .appendingPathComponent(name)
            try Posix.mkdirP(tokenizerDirectory)
            try FileManager.default.copyItem(atPath: source, toPath: destination)
            try recordOutputFile(relativePath: "tokenizer/\(name)",
                                 path: destination,
                                 progress: progress)
        }
    }

    private func writeSmall(path: String, data: Data) throws {
        try Posix.mkdirP((path as NSString).deletingLastPathComponent)
        try data.write(to: URL(fileURLWithPath: path), options: [.atomic])
        audit.recordWrite(bytes: data.count)
    }

    private func writeManifest(plan: RepackPlan,
                               partialDirectory: String,
                               metadata: IndexLoader.SourceMetadata,
                               expertStride: UInt64) throws {
        var bits = GTurboJSON.QuantBitWidths(
            embedding: 4,
            attention: 4,
            router: 8,
            sharedExpert: 8,
            routedExpert: 4)
        for entry in plan.resident.entries {
            if entry.name == "language_model.model.embed_tokens.weight",
               let spec = entry.quantSpec {
                bits.embedding = spec.bits
            }
            if entry.name.hasSuffix(".self_attn.q_proj.weight"),
               let spec = entry.quantSpec {
                bits.attention = spec.bits
            }
            if entry.name.contains(".linear_attn."), let spec = entry.quantSpec {
                bits.deltaNet = spec.bits
            }
            if entry.name.hasSuffix(".router.proj.weight"),
               let spec = entry.quantSpec {
                bits.router = spec.bits
            }
            if plan.arch.modelFamily == "qwen4_exp_text",
               entry.name.hasSuffix(".mlp.gate.weight") {
                bits.router = 16
            }
            if (entry.name.hasSuffix(".mlp.gate_proj.weight")
                || entry.name.contains(".mlp.shared_expert.gate_proj.weight")),
               let spec = entry.quantSpec {
                bits.sharedExpert = spec.bits
            }
            if entry.name.contains(".shared_expert_gate"),
               let spec = entry.quantSpec {
                bits.sharedExpertGate = spec.bits
            }
            if entry.name.hasSuffix(".lm_head.weight"), let spec = entry.quantSpec {
                bits.lmHead = spec.bits
            }
        }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
           let routedBits = layer.subTensors.first?.bitsForWeights {
            bits.routedExpert = routedBits
        }
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size,
                                                   sha256: $0.sha256))
        }
        let manifest = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: plan.matchedModelID ?? "unknown/local-snapshot",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(
                where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
        let temporary = (partialDirectory as NSString)
            .appendingPathComponent("manifest.json.tmp")
        let destination = (partialDirectory as NSString)
            .appendingPathComponent("manifest.json")
        try writeSmall(path: temporary, data: manifest)
        try Posix.rename(from: temporary, to: destination)
    }
}
