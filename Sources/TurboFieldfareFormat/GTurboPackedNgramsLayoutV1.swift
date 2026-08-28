import Foundation

package struct GTurboNgramComponentV1: Codable, Equatable, Sendable {
    package let offset: UInt64
    package let size: UInt64
    package let dtype: String
    package let shape: [UInt32]
    package let bits: Int?

    package init(offset: UInt64, size: UInt64, dtype: String,
                 shape: [UInt32], bits: Int?) {
        self.offset = offset
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.bits = bits
    }
}

package struct GTurboNgramShardV1: Codable, Equatable, Sendable {
    package let shard: Int
    package let file: String
    package let fileSize: UInt64
    package let weight: GTurboNgramComponentV1
    package let scales: GTurboNgramComponentV1
    package let biases: GTurboNgramComponentV1

    package init(shard: Int, file: String, fileSize: UInt64,
                 weight: GTurboNgramComponentV1,
                 scales: GTurboNgramComponentV1,
                 biases: GTurboNgramComponentV1) {
        self.shard = shard
        self.file = file
        self.fileSize = fileSize
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }
}

package struct GTurboPackedNgramsLayoutV1: Codable, Equatable, Sendable {
    package let versionMajor: Int
    package let versionMinor: Int
    package let layer: Int
    package let splitParts: Int
    package let groupSize: Int
    package let shards: [GTurboNgramShardV1]

    package init(versionMajor: Int = 1, versionMinor: Int = 0,
                 layer: Int, splitParts: Int, groupSize: Int,
                 shards: [GTurboNgramShardV1]) {
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.layer = layer
        self.splitParts = splitParts
        self.groupSize = groupSize
        self.shards = shards
    }
}

package enum GTurboPackedNgramsLayoutCodec {
    package static let maxBytes: UInt64 = 4 * 1024 * 1024

    package static func decode(_ data: Data) throws -> GTurboPackedNgramsLayoutV1 {
        let layout: GTurboPackedNgramsLayoutV1
        do {
            layout = try JSONDecoder().decode(GTurboPackedNgramsLayoutV1.self, from: data)
        } catch {
            throw GTurboFormatError.invalid(
                field: "packed_ngrams/layout.json", reason: "\(error)")
        }
        try validate(layout)
        return layout
    }

    package static func encode(_ layout: GTurboPackedNgramsLayoutV1) throws -> Data {
        try validate(layout)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(layout))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(
                field: "packed_ngrams/layout.json", reason: "\(error)")
        }
    }

    package static func validate(_ layout: GTurboPackedNgramsLayoutV1) throws {
        guard layout.versionMajor == 1, layout.versionMinor >= 0,
              layout.layer >= 0, layout.splitParts > 0,
              layout.groupSize > 0, layout.shards.count == layout.splitParts else {
            throw GTurboFormatError.invalid(
                field: "packed_ngrams/layout", reason: "invalid geometry")
        }

        var shardIDs = Set<Int>()
        var shardFiles = Set<String>()
        var weightShape: [UInt32]?
        for shard in layout.shards {
            guard shard.shard >= 0, shard.shard < layout.splitParts,
                  shardIDs.insert(shard.shard).inserted else {
                throw GTurboFormatError.invalid(
                    field: "packed_ngrams/layout.shards", reason: "duplicate or invalid shard")
            }
            try GTurboPathValidator.validateBasename(
                shard.file, field: "packed_ngrams/layout.shards[\(shard.shard)].file")
            let fileKey = GTurboPathValidator.appleFilesystemKey(shard.file)
            guard fileKey != "layout.json", shardFiles.insert(fileKey).inserted else {
                throw GTurboFormatError.invalid(
                    field: "packed_ngrams/layout.shards[\(shard.shard)].file",
                    reason: "reserved or duplicate shard filename")
            }
            guard shard.fileSize > 0,
                  shard.fileSize % GTurboFormatV1.alignmentBytes == 0 else {
                throw GTurboFormatError.invalid(
                    field: "packed_ngrams/layout.shards[\(shard.shard)].fileSize",
                    reason: "file size must be page aligned")
            }

            let components = [
                ("weight", shard.weight),
                ("scales", shard.scales),
                ("biases", shard.biases),
            ]
            var previousEnd: UInt64 = 0
            for (name, component) in components {
                guard component.offset % GTurboFormatV1.alignmentBytes == 0,
                      component.size > 0,
                      component.offset <= UInt64.max - component.size,
                      component.offset >= previousEnd,
                      component.offset + component.size <= shard.fileSize,
                      component.shape.count == 2,
                      component.shape.allSatisfy({ $0 > 0 }) else {
                    throw GTurboFormatError.invalid(
                        field: "packed_ngrams/layout.shards[\(shard.shard)].\(name)",
                        reason: "invalid or overlapping component range")
                }
                previousEnd = component.offset + component.size
            }
            guard shard.weight.dtype == "U32", shard.weight.bits == 4,
                  shard.scales.dtype == "BF16", shard.scales.bits == nil,
                  shard.biases.dtype == "BF16", shard.biases.bits == nil,
                  shard.scales.shape == shard.biases.shape,
                  shard.weight.shape[0] == shard.scales.shape[0] else {
                throw GTurboFormatError.invalid(
                    field: "packed_ngrams/layout.shards[\(shard.shard)]",
                    reason: "invalid affine-Q4 components")
            }
            if let expected = weightShape {
                guard shard.weight.shape == expected else {
                    throw GTurboFormatError.invalid(
                        field: "packed_ngrams/layout.shards[\(shard.shard)].weight.shape",
                        reason: "inconsistent shard shape")
                }
            } else {
                weightShape = shard.weight.shape
            }
        }
    }
}
