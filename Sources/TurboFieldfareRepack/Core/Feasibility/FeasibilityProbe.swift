import Darwin
import Foundation
import Darwin.Mach

public struct FeasibilityMaterializationReport: Codable, Sendable, Equatable {
    public let path: String
    public let selectedBytes: UInt64
    public let downloadedBytes: UInt64
    public let materializedBytes: UInt64
    public let rangeCount: Int
    public let sampleExpertCount: Int
    public let sampledLayerCount: Int
    public let maxRangeBytes: Int
}

public struct FeasibilityMemoryMeasurement: Codable, Sendable, Equatable {
    public let phase: String
    public let capacityBytes: UInt64
    public let touchedBytes: UInt64
    public let physFootprintBytes: UInt64
    public let residentBytes: UInt64
    public let memoryPressure: String?
}

public struct FeasibilityMemoryPlan: Codable, Sendable, Equatable {
    public let recurrentStateBytes: UInt64
    public let kvCacheBytes: UInt64
    public let prefillScratchBytes: UInt64
    public let decodeScratchBytes: UInt64
    public let expertCacheSlotBytes: UInt64
    public let expertCache8SlotBytes: UInt64
    public let expertCache16SlotBytes: UInt64

    public init(recurrentStateBytes: UInt64,
                kvCacheBytes: UInt64,
                prefillScratchBytes: UInt64,
                decodeScratchBytes: UInt64,
                expertCacheSlotBytes: UInt64,
                expertCache8SlotBytes: UInt64,
                expertCache16SlotBytes: UInt64) {
        self.recurrentStateBytes = recurrentStateBytes
        self.kvCacheBytes = kvCacheBytes
        self.prefillScratchBytes = prefillScratchBytes
        self.decodeScratchBytes = decodeScratchBytes
        self.expertCacheSlotBytes = expertCacheSlotBytes
        self.expertCache8SlotBytes = expertCache8SlotBytes
        self.expertCache16SlotBytes = expertCache16SlotBytes
    }
}

public struct FeasibilityProbeReport: Codable, Sendable, Equatable {
    public let inventory: FeasibilityInventory
    public let materialization: FeasibilityMaterializationReport
    public let memoryPlan: FeasibilityMemoryPlan
    public let measurements: [FeasibilityMemoryMeasurement]
}

public enum FeasibilityProbe {
    public static let defaultMaxRangeBytes = 64 * 1024 * 1024
    public static let defaultSampleExpertCount = 8

    public static func runRemote(repoID: String,
                                 revision: String,
                                 token: String? = nil,
                                 workingDirectory: String,
                                 sampleExpertCount: Int = defaultSampleExpertCount,
                                 maxRangeBytes: Int = defaultMaxRangeBytes) async throws
        -> FeasibilityProbeReport {
        guard sampleExpertCount > 0, sampleExpertCount <= 256 else {
            throw RepackError.configurationInvalid(
                detail: "sample expert count must be between 1 and 256")
        }
        guard maxRangeBytes > 0 else {
            throw RepackError.configurationInvalid(detail: "maximum range must be positive")
        }

        try Posix.mkdirP(workingDirectory)
        let inventory = try await FeasibilityInventory.scanRemote(
            repoID: repoID,
            revision: revision,
            token: token,
            workingDirectory: workingDirectory)
        let metadataDirectory = (workingDirectory as NSString)
            .appendingPathComponent("metadata")
        let remote = HuggingFaceRemoteSource(
            repoID: repoID,
            requestedRevision: revision,
            resolvedCommit: inventory.resolvedCommit,
            token: token,
            tempDirectory: (workingDirectory as NSString).appendingPathComponent("ranges"))
        let snapshot = try await RemoteSnapshotLoader.loadHeaders(
            remote: remote,
            requireKnownSource: false,
            metadataDirectory: metadataDirectory)
        let ranges = try selectedRanges(snapshot: snapshot,
                                        sampleExpertCount: sampleExpertCount,
                                        maxRangeBytes: maxRangeBytes)
        let payloadPath = (workingDirectory as NSString)
            .appendingPathComponent("materialized-payload.bin")
        let materialization = try await materialize(ranges: ranges,
                                                    remote: remote,
                                                    remoteFiles: snapshot.remoteFiles,
                                                    sampleExpertCount: sampleExpertCount,
                                                    maxRangeBytes: maxRangeBytes,
                                                    outputPath: payloadPath)
        let memoryPlan = try memoryPlan(ranges: ranges,
                                        sampleExpertCount: sampleExpertCount)
        let measurements = try measure(path: payloadPath, plan: memoryPlan)
        return FeasibilityProbeReport(inventory: inventory,
                                      materialization: materialization,
                                      memoryPlan: memoryPlan,
                                      measurements: measurements)
    }

    private struct SelectedRange: Sendable {
        let shard: String
        let offset: UInt64
        let length: Int
        let layer: Int?
        let isExpert: Bool
    }

    private static func selectedRanges(snapshot: RemoteHeaderSnapshot,
                                      sampleExpertCount: Int,
                                      maxRangeBytes: Int) throws -> [SelectedRange] {
        var ranges: [SelectedRange] = []
        for (index, header) in snapshot.shardHeaders.enumerated() {
            guard index < snapshot.metadata.shardFilenames.count else {
                throw RepackError.remoteProtocolInvalid(detail: "header/shard count mismatch")
            }
            let shard = snapshot.metadata.shardFilenames[index]
            for tensor in header.tensors {
                let role = FeasibilityInventory.role(for: tensor.name)
                switch role {
                case .textResident:
                    appendSplitRanges(to: &ranges,
                                      shard: shard,
                                      offset: tensor.absoluteOffset,
                                      length: tensor.sizeBytes,
                                      maxRangeBytes: maxRangeBytes,
                                      layer: nil,
                                      isExpert: false)
                case .routedExpert:
                    guard tensor.name.contains(".switch_mlp.") else { continue }
                    guard let firstDimension = tensor.shape.first,
                          firstDimension == 256,
                          tensor.sizeBytes % firstDimension == 0 else {
                        throw RepackError.configurationInvalid(
                            detail: "routed tensor has no 256-expert leading dimension: \(tensor.name)")
                    }
                    let stride = tensor.sizeBytes / firstDimension
                    let layer = layerIndex(in: tensor.name)
                    for expert in 0..<sampleExpertCount {
                        appendSplitRanges(to: &ranges,
                                          shard: shard,
                                          offset: tensor.absoluteOffset + UInt64(expert) * stride,
                                          length: stride,
                                          maxRangeBytes: maxRangeBytes,
                                          layer: layer,
                                          isExpert: true)
                    }
                case .omittedMultimodal, .unsupported:
                    continue
                }
            }
        }
        let sorted = ranges.sorted {
            ($0.shard, $0.offset, $0.length) < ($1.shard, $1.offset, $1.length)
        }
        var coalesced: [SelectedRange] = []
        for range in sorted {
            if let last = coalesced.last,
               last.shard == range.shard,
               last.offset + UInt64(last.length) == range.offset,
               last.layer == range.layer,
               last.isExpert == range.isExpert {
                coalesced[coalesced.count - 1] = SelectedRange(
                    shard: last.shard,
                    offset: last.offset,
                    length: last.length + range.length,
                    layer: last.layer,
                    isExpert: last.isExpert)
            } else {
                coalesced.append(range)
            }
        }
        return coalesced
    }

    private static func appendSplitRanges(to ranges: inout [SelectedRange],
                                          shard: String,
                                          offset: UInt64,
                                          length: UInt64,
                                          maxRangeBytes: Int,
                                          layer: Int?,
                                          isExpert: Bool) {
        var currentOffset = offset
        var remaining = length
        while remaining > 0 {
            let chunk = min(remaining, UInt64(maxRangeBytes))
            ranges.append(SelectedRange(shard: shard,
                                        offset: currentOffset,
                                        length: Int(chunk),
                                        layer: layer,
                                        isExpert: isExpert))
            currentOffset += chunk
            remaining -= chunk
        }
    }

    private static func materialize(ranges: [SelectedRange],
                                    remote: HuggingFaceRemoteSource,
                                    remoteFiles: [String: RemoteFileInfo],
                                    sampleExpertCount: Int,
                                    maxRangeBytes: Int,
                                    outputPath: String) async throws
        -> FeasibilityMaterializationReport {
        let parent = (outputPath as NSString).deletingLastPathComponent
        try Posix.mkdirP(parent)
        switch try Posix.entryKind(outputPath) {
        case .absent:
            break
        case .regular:
            try FileManager.default.removeItem(atPath: outputPath)
        default:
            throw RepackError.installPathUnsafe(
                path: outputPath,
                detail: "materialization output has the wrong entry type")
        }
        let outputFD = try Posix.openCreateRW(outputPath)
        defer { close(outputFD) }
        var destinationOffset: UInt64 = 0
        var downloadedBytes: UInt64 = 0
        var sampleLayers = Set<Int>()
        for (index, range) in ranges.enumerated() {
            guard let info = remoteFiles[range.shard],
                  info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "missing ranged shard metadata for \(range.shard)")
            }
            let target = (parent as NSString)
                .appendingPathComponent("range-\(index).tmp")
            let temporary = try await remote.downloadRangeToTempFile(
                filename: range.shard,
                info: info,
                offset: range.offset,
                length: range.length,
                targetPath: target)
            do {
                defer { try? FileManager.default.removeItem(atPath: temporary.path) }
                guard temporary.byteCount == UInt64(range.length) else {
                    throw RepackError.remoteProtocolInvalid(detail: "short materialized range")
                }
                let mapping = try MmapHandle(path: temporary.path)
                try Posix.pwriteAll(fd: outputFD,
                                    path: outputPath,
                                    buf: mapping.base,
                                    count: range.length,
                                    offset: destinationOffset)
            }
            destinationOffset += UInt64(range.length)
            downloadedBytes += UInt64(range.length)
            if let layer = range.layer { sampleLayers.insert(layer) }
        }
        try Posix.fsync(outputFD, path: outputPath)
        let selectedBytes = ranges.reduce(UInt64(0)) { $0 + UInt64($1.length) }
        return FeasibilityMaterializationReport(
            path: outputPath,
            selectedBytes: selectedBytes,
            downloadedBytes: downloadedBytes,
            materializedBytes: destinationOffset,
            rangeCount: ranges.count,
            sampleExpertCount: sampleExpertCount,
            sampledLayerCount: sampleLayers.count,
            maxRangeBytes: maxRangeBytes)
    }

    private static func memoryPlan(ranges: [SelectedRange],
                                   sampleExpertCount: Int) throws -> FeasibilityMemoryPlan {
        var expertBytesByLayer: [Int: UInt64] = [:]
        for range in ranges where range.isExpert {
            guard let layer = range.layer else { continue }
            expertBytesByLayer[layer, default: 0] += UInt64(range.length)
        }
        let sampledLayerBytes = expertBytesByLayer.values.max() ?? 0
        let slotBytes = sampledLayerBytes / UInt64(sampleExpertCount)
        let recurrent = try checkedMultiply(30 * 32 * 128 * 128, 4,
                                            detail: "Gated DeltaNet state")
        let kv = try checkedMultiply(10 * 4096 * 2 * 256 * 2, 1,
                                     detail: "4K KV cache")
        let prefill = try checkedMultiply(512 * 2048 * 2, 4,
                                           detail: "prefill scratch")
        let decode = try checkedMultiply(2048 * 2, 4, detail: "decode scratch")
        return FeasibilityMemoryPlan(
            recurrentStateBytes: recurrent,
            kvCacheBytes: kv,
            prefillScratchBytes: prefill,
            decodeScratchBytes: decode,
            expertCacheSlotBytes: slotBytes,
            expertCache8SlotBytes: slotBytes * 8,
            expertCache16SlotBytes: slotBytes * 16)
    }

    private static func measure(path: String,
                                plan: FeasibilityMemoryPlan) throws
        -> [FeasibilityMemoryMeasurement] {
        var measurements: [FeasibilityMemoryMeasurement] = []
        do {
            let mapping = try MmapHandle(path: path)
            appendMeasurement(&measurements,
                              phase: "before_payload_touch",
                              capacity: UInt64(mapping.length),
                              touched: 0)
            let touchedPayload = touch(mapping: mapping)
            appendMeasurement(&measurements,
                              phase: "after_payload_touch",
                              capacity: UInt64(mapping.length),
                              touched: touchedPayload)
            let state = try TouchedAllocation(bytes: Int(plan.recurrentStateBytes))
            state.touch()
            appendMeasurement(&measurements,
                              phase: "after_recurrent_state",
                              capacity: UInt64(mapping.length) + plan.recurrentStateBytes,
                              touched: touchedPayload + plan.recurrentStateBytes)
            let kv = try TouchedAllocation(bytes: Int(plan.kvCacheBytes))
            kv.touch()
            appendMeasurement(&measurements,
                              phase: "after_kv_cache",
                              capacity: UInt64(mapping.length) + plan.recurrentStateBytes + plan.kvCacheBytes,
                              touched: touchedPayload + plan.recurrentStateBytes + plan.kvCacheBytes)
            let scratch = try TouchedAllocation(bytes: Int(plan.prefillScratchBytes + plan.decodeScratchBytes))
            scratch.touch()
            appendMeasurement(&measurements,
                              phase: "after_prefill_decode_scratch",
                              capacity: UInt64(mapping.length) + plan.recurrentStateBytes + plan.kvCacheBytes
                                  + plan.prefillScratchBytes + plan.decodeScratchBytes,
                              touched: touchedPayload + plan.recurrentStateBytes + plan.kvCacheBytes
                                  + plan.prefillScratchBytes + plan.decodeScratchBytes)
            let cache8 = try TouchedAllocation(bytes: Int(plan.expertCache8SlotBytes))
            cache8.touchRepeated(times: 3)
            appendMeasurement(&measurements,
                              phase: "after_expert_cache_8_slots",
                              capacity: UInt64(mapping.length) + plan.recurrentStateBytes + plan.kvCacheBytes
                                  + plan.prefillScratchBytes + plan.decodeScratchBytes + plan.expertCache8SlotBytes,
                              touched: touchedPayload + plan.recurrentStateBytes + plan.kvCacheBytes
                                  + plan.prefillScratchBytes + plan.decodeScratchBytes + plan.expertCache8SlotBytes)
            let cache16 = try TouchedAllocation(bytes: Int(plan.expertCache16SlotBytes))
            cache16.touchRepeated(times: 3)
            appendMeasurement(&measurements,
                              phase: "after_expert_cache_16_slots",
                              capacity: UInt64(mapping.length) + plan.recurrentStateBytes + plan.kvCacheBytes
                                  + plan.prefillScratchBytes + plan.decodeScratchBytes + plan.expertCache16SlotBytes,
                              touched: touchedPayload + plan.recurrentStateBytes + plan.kvCacheBytes
                                  + plan.prefillScratchBytes + plan.decodeScratchBytes + plan.expertCache16SlotBytes)
        }
        appendMeasurement(&measurements,
                          phase: "after_teardown",
                          capacity: 0,
                          touched: 0)
        return measurements
    }

    private static func appendMeasurement(_ measurements: inout [FeasibilityMemoryMeasurement],
                                          phase: String,
                                          capacity: UInt64,
                                          touched: UInt64) {
        let sample = memorySample()
        measurements.append(FeasibilityMemoryMeasurement(
            phase: phase,
            capacityBytes: capacity,
            touchedBytes: touched,
            physFootprintBytes: sample.physFootprint,
            residentBytes: sample.resident,
            memoryPressure: memoryPressure()))
    }

    private static func touch(mapping: MmapHandle) -> UInt64 {
        guard mapping.length > 0 else { return 0 }
        var checksum: UInt8 = 0
        for offset in stride(from: 0, to: mapping.length, by: Posix.pageSize) {
            checksum ^= mapping.base.load(fromByteOffset: offset, as: UInt8.self)
        }
        checksum ^= mapping.base.load(fromByteOffset: mapping.length - 1, as: UInt8.self)
        return checksum == 255 ? UInt64(mapping.length) : UInt64(mapping.length)
    }

    private struct MemorySample {
        let physFootprint: UInt64
        let resident: UInt64
    }

    private static func memorySample() -> MemorySample {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return MemorySample(physFootprint: 0, resident: 0) }
        return MemorySample(physFootprint: UInt64(info.phys_footprint),
                            resident: UInt64(info.resident_size))
    }

    private static func memoryPressure() -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/memory_pressure")
        process.arguments = ["-Q"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
            process.waitUntilExit()
            return String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                          encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }

    private static func checkedMultiply(_ lhs: Int,
                                        _ rhs: Int,
                                        detail: String) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow, value >= 0 else {
            throw RepackError.configurationInvalid(detail: "overflow while calculating \(detail)")
        }
        return UInt64(value)
    }

    private static func layerIndex(in name: String) -> Int? {
        guard let range = name.range(of: ".layers.") else { return nil }
        let tail = name[range.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[..<dot])
    }
}

private final class TouchedAllocation {
    private let pointer: UnsafeMutableRawPointer
    private let bytes: Int

    init(bytes: Int) throws {
        guard bytes > 0, let pointer = malloc(bytes) else {
            throw RepackError.configurationInvalid(detail: "allocation failed for \(bytes) bytes")
        }
        self.pointer = pointer
        self.bytes = bytes
    }

    deinit { free(pointer) }

    func touchRepeated(times: Int = 1) {
        for _ in 0..<times { touch() }
    }

    func touch() {
        for offset in stride(from: 0, to: bytes, by: Posix.pageSize) {
            pointer.storeBytes(of: UInt8(truncatingIfNeeded: offset),
                               toByteOffset: offset,
                               as: UInt8.self)
        }
        pointer.storeBytes(of: UInt8(truncatingIfNeeded: bytes),
                           toByteOffset: bytes - 1,
                           as: UInt8.self)
    }
}