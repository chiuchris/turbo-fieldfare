import Foundation
import TurboFieldfareFormat

struct NgramComponentEntry: Sendable, Equatable {
    let offset: UInt64
    let size: UInt64
    let dtype: String
    let shape: [UInt32]
    let bits: Int?
}

struct NgramShardEntry: Sendable, Equatable {
    let shard: Int
    let file: String
    let fileSize: UInt64
    let weight: NgramComponentEntry
    let scales: NgramComponentEntry
    let biases: NgramComponentEntry
}

struct NgramRowLocation: Sendable, Equatable {
    let shardIndex: Int
    let row: UInt64
}

struct PackedNgramsLayout: Sendable, Equatable {
    let layer: Int
    let splitParts: Int
    let groupSize: Int
    let shards: [NgramShardEntry]

    var rowWidth: Int {
        Int(shards[0].weight.shape[1]) * 8
    }

    var totalRows: UInt64 {
        shards.reduce(0) { $0 + UInt64($1.weight.shape[0]) }
    }

    func shard(_ index: Int) -> NgramShardEntry {
        shards[index]
    }

    func locate(globalRow: Int64) throws -> NgramRowLocation {
        guard globalRow >= 0 else {
            throw StreamerError.offsetOutOfRange(UInt64.max)
        }
        var remaining = UInt64(globalRow)
        for (index, shard) in shards.enumerated() {
            let rows = UInt64(shard.weight.shape[0])
            if remaining < rows {
                return NgramRowLocation(shardIndex: index, row: remaining)
            }
            remaining -= rows
        }
        throw StreamerError.offsetOutOfRange(UInt64(globalRow))
    }
}

enum PackedNgramsLayoutReader {
    static let defaultMaxBytes = GTurboPackedNgramsLayoutCodec.maxBytes

    static func load(directoryURL: URL,
                     manifest: Manifest,
                     maxBytes: UInt64 = defaultMaxBytes) throws -> PackedNgramsLayout {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata(
            "packed_ngrams/layout.json", maxBytes: maxBytes)
        return try decode(data: data, manifest: manifest)
    }

    static func decode(data: Data,
                       manifest: Manifest?) throws -> PackedNgramsLayout {
        let wire: GTurboPackedNgramsLayoutV1
        do {
            wire = try GTurboPackedNgramsLayoutCodec.decode(data)
        } catch {
            throw ModelError.indexCorrupt(
                detail: "packed_ngrams/layout.json: \(error)")
        }
        if let manifest {
            guard manifest.versionMajor == GTurboFormatV3.versionMajor,
                  manifest.files["packed_ngrams/layout.json"] != nil else {
                throw ModelError.indexCorrupt(
                    detail: "packed_ngrams/layout.json is not declared by a v3 manifest")
            }
            for shard in wire.shards {
                let relativePath = "packed_ngrams/\(shard.file)"
                guard manifest.files[relativePath]?.size == shard.fileSize else {
                    throw ModelError.indexCorrupt(
                        detail: "packed_ngrams file size mismatch: \(relativePath)")
                }
            }
        }

        func component(_ value: GTurboNgramComponentV1) -> NgramComponentEntry {
            NgramComponentEntry(
                offset: value.offset,
                size: value.size,
                dtype: value.dtype,
                shape: value.shape,
                bits: value.bits)
        }
        let shards = wire.shards.sorted { $0.shard < $1.shard }.map { shard in
            NgramShardEntry(
                shard: shard.shard,
                file: shard.file,
                fileSize: shard.fileSize,
                weight: component(shard.weight),
                scales: component(shard.scales),
                biases: component(shard.biases))
        }
        return PackedNgramsLayout(
            layer: wire.layer,
            splitParts: wire.splitParts,
            groupSize: wire.groupSize,
            shards: shards)
    }
}
