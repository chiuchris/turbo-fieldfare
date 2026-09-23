import Foundation
import TurboFieldfareFormat

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes = GTurboFormatV1.alignmentBytes
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec of the canonical payload written to the artifact.
    let quantSpec: QuantSpec?
    /// Quantization spec of the source payload, when it differs from the
    /// canonical artifact representation.
    let sourceQuantSpec: QuantSpec?
    /// Temporary file that receives source bytes before canonical conversion.
    let sourceStagingPath: String?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?

    init(name: String, dtype: UInt8, logicalShape4: [UInt32],
         fileOffset: UInt64, sizeBytes: UInt64,
         scaleOffset: UInt64, scaleSize: UInt64,
         biasOffset: UInt64, biasSize: UInt64, quantSpec: QuantSpec?,
         sourceWeight: SourceTensor, sourceScales: SourceTensor?,
         sourceBiases: SourceTensor?, sourceQuantSpec: QuantSpec? = nil,
         sourceStagingPath: String? = nil) {
        self.name = name
        self.dtype = dtype
        self.logicalShape4 = logicalShape4
        self.fileOffset = fileOffset
        self.sizeBytes = sizeBytes
        self.scaleOffset = scaleOffset
        self.scaleSize = scaleSize
        self.biasOffset = biasOffset
        self.biasSize = biasSize
        self.quantSpec = quantSpec
        self.sourceQuantSpec = sourceQuantSpec
        self.sourceStagingPath = sourceStagingPath
        self.sourceWeight = sourceWeight
        self.sourceScales = sourceScales
        self.sourceBiases = sourceBiases
    }
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // "gate" | "up" | "down"
    let component: String              // "weights" | "scales" | "biases"
    let dtype: UInt8                   // 0=U32, 1=BF16
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct NgramShardFilePlan: Sendable {
    let shardIndex: Int
    let path: String
    let fileSize: UInt64
    let weight: SourceTensor
    let weightOffset: UInt64
    let scales: SourceTensor
    let scalesOffset: UInt64
    let biases: SourceTensor
    let biasesOffset: UInt64
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let ngramShards: [NgramShardFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
}

// MARK: - Planner

struct VisionPackPlan: Sendable {
    struct Entry: Sendable {
        let source: SourceTensor
        let executionPosition: Int
        let fileOffset: UInt64
        let quantSpec: QuantSpec?
        /// Affine group size, pinned to 64 by `planVisionCompanion`.
        let groupSize: Int
    }

    let entries: [Entry]
    let weightsFileSize: UInt64
    let sourcePayloadBytes: UInt64
}

enum RepackPlanner {

    static func planVisionCompanion(
        meta: IndexLoader.SourceMetadata,
        shardHeaders: [Safetensors.Header]
    ) throws -> VisionPackPlan {
        guard meta.baseBits == 4, meta.baseGroupSize == 64,
              meta.baseMode.lowercased() == "affine" else {
            throw RepackError.configurationInvalid(
                detail: "vision companion requires MLX affine 4-bit group-64 source metadata")
        }

        let tensors = shardHeaders.flatMap(\.tensors).filter {
            isMultimodalTensorName($0.name)
        }
        guard tensors.count == 358 else {
            throw RepackError.configurationInvalid(
                detail: "expected 358 vision tensors, found \(tensors.count)")
        }
        let sourceBytes = tensors.reduce(UInt64(0)) { $0 + $1.sizeBytes }
        guard sourceBytes == 1_140_925_536 else {
            throw RepackError.configurationInvalid(
                detail: "expected 1140925536 vision bytes, found \(sourceBytes)")
        }

        let ordered = tensors.sorted { visionExecutionKey($0.name) < visionExecutionKey($1.name) }
        var offset: UInt64 = 0
        var entries: [VisionPackPlan.Entry] = []
        entries.reserveCapacity(ordered.count)
        for (position, tensor) in ordered.enumerated() {
            offset = visionAlignUp(offset, to: GTurboVisionFormatV1.alignmentBytes)
            let quantSpec: QuantSpec?
            if tensor.dtype == .u32 {
                let spec = IndexLoader.quantSpec(forTensor: tensor.name, meta: meta)
                // Group size is not carried per tensor in this tree; the base
                // group was already pinned to 64 above.
                guard spec.bits == 4 else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision quantization for \(tensor.name)")
                }
                quantSpec = spec
            } else {
                guard tensor.dtype == .bf16 else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision dtype for \(tensor.name)")
                }
                quantSpec = nil
            }
            entries.append(.init(source: tensor,
                                 executionPosition: position,
                                 fileOffset: offset,
                                 quantSpec: quantSpec,
                                 groupSize: meta.baseGroupSize))
            offset += tensor.sizeBytes
        }
        return VisionPackPlan(entries: entries,
                              weightsFileSize: offset,
                              sourcePayloadBytes: sourceBytes)
    }

    private static func visionExecutionKey(_ name: String) -> String {
        if name.hasPrefix("vision_tower.patch_embedder.") {
            return "0000/\(name)"
        }
        if let layer = layerIndex(in: name), name.hasPrefix("vision_tower.encoder.layers.") {
            return String(format: "1000/%03d/%@", layer, name)
        }
        if name == "vision_tower.std_bias" || name == "vision_tower.std_scale" {
            return "2000/\(name)"
        }
        if name.hasPrefix("embed_vision.") {
            return "3000/\(name)"
        }
        return "9999/\(name)"
    }

    private static func visionAlignUp(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        ((value + alignment - 1) / alignment) * alignment
    }

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case ngramShard(shard: Int)
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int,
                         modelFamily: String = "gemma4") -> Bucket {
        if isExcludedAuxiliaryTensorName(name) {
            return .excludedMultimodal
        }
        if name.hasPrefix("language_model.") {
            if name.hasPrefix("language_model.mtp.") {
                return .lmResident
            }
            if modelFamily == "qwen4_exp_text",
               let shard = ngramShardIndex(in: name) {
                return .ngramShard(shard: shard)
            }
            if let role = routedExpertRole(in: name, modelFamily: modelFamily),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    static func canonicalSourceTensorName(_ name: String) -> String {
        let name = name.hasPrefix("mtp.") ? "language_model.\(name)" : name
        guard name.hasSuffix(".ple.conv_weight") else { return name }
        return String(name.dropLast("conv_weight".count)) + "conv1d.weight"
    }

    static func sourceQuantSpec(for name: String,
                               meta: IndexLoader.SourceMetadata) -> QuantSpec {
        guard name.hasPrefix("language_model.mtp.") else {
            return IndexLoader.quantSpec(forTensor: name, meta: meta)
        }
        if name.contains(".self_attn.indexer.index_qk_proj.") {
            return QuantSpec(bits: 8, groupSize: 64)
        }
        if name.contains(".mlp.switch_mlp.") || name.contains(".self_attn.") {
            return QuantSpec(bits: 4, groupSize: 32)
        }
        if name.contains(".mlp.gate.") ||
           name.contains(".mlp.shared_expert.") ||
           name.contains(".mlp.shared_expert_gate.") {
            return QuantSpec(bits: 8, groupSize: 64)
        }
        return IndexLoader.quantSpec(forTensor: name, meta: meta)
    }

    static func preservesSourceMTPRepresentation(
        _ name: String, modelFamily: String
    ) -> Bool {
        modelFamily == "qwen4_exp_text" &&
            name.hasPrefix("language_model.mtp.")
    }

    static func residentOutputQuantSpec(for name: String,
                                        modelFamily: String) -> QuantSpec {
        if modelFamily == "qwen4_exp_text",
           (name == "language_model.model.embed_tokens.weight"
            || name == "language_model.lm_head.weight") {
            return QuantSpec(bits: 8, groupSize: 64)
        }
        if modelFamily == "qwen3_5_moe_text",
           name.hasSuffix(".mlp.shared_expert_gate.weight") {
            return QuantSpec(bits: 8, groupSize: 64)
        }
        return CanonicalQuantization.target
    }

    private static func routedExpertRole(in name: String, modelFamily: String) -> String? {
        let isQwen = modelFamily == "qwen3_5_moe_text" ||
            modelFamily == "qwen4_exp_text"
        let expertPath = isQwen ? ".mlp.switch_mlp." : ".experts.switch_glu."
        guard name.contains(expertPath) else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    private static func ngramShardIndex(in name: String) -> Int? {
        let marker = ".ple.ple_embedding.ngram_embedding.shard_"
        guard let range = name.range(of: marker) else { return nil }
        let suffix = name[range.upperBound...]
        guard let dot = suffix.firstIndex(of: "."),
              suffix[suffix.index(after: dot)...] == "weight" else { return nil }
        return Int(suffix[..<dot])
    }

    private static func isExcludedAuxiliaryTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.") ||
            name.hasPrefix("mtp.")
    }

    static func isBF16Qwen38Projection(_ name: String) -> Bool {
        if name.hasPrefix("language_model.mtp.") &&
           (name.hasSuffix(".fc_embedding.weight") ||
            name.hasSuffix(".fc_hidden.weight")) {
            return true
        }
        let projectionSuffixes = [
            ".ple.key_proj.weight",
            ".ple.value_proj.weight",
            ".input_mix_weight_down.weight",
            ".input_mix_weight_up.weight",
            ".block_inject_weight.weight",
        ]
        guard projectionSuffixes.contains(where: name.hasSuffix) else { return false }
        return name.contains(".ple.") ||
            name.contains(".attn_hyper_connection.") ||
            name.contains(".mlp_hyper_connection.") ||
            name.contains(".hyper_connection_mixer.")
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        guard let r = name.range(of: ".layers.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Returns packed quantized source tensors the Qwen3.8 runtime cannot execute.
    /// MTP tensors are deliberately excluded because proposal execution is not
    /// active in the current runtime.
    static func qwen38RuntimeCompatibilityIssues(
        meta: IndexLoader.SourceMetadata,
        tensors: [SourceTensor]
    ) -> [String] {
        tensors.compactMap { tensor in
            guard tensor.name.hasPrefix("language_model."),
                  !tensor.name.hasPrefix("language_model.mtp."),
                  tensor.dtype == .u32 else {
                return nil
            }
            let spec = IndexLoader.quantSpec(forTensor: tensor.name, meta: meta)
            guard spec.bits == 4, spec.groupSize == 32 else {
                do {
                    _ = try CanonicalQuantization.layout(
                        shape: tensor.shape, source: spec)
                    return nil
                } catch {
                    return "\(tensor.name): source \(spec.bits)-bit/group-\(spec.groupSize); "
                        + "runtime supports 4-bit/group-32"
                }
            }
            return nil
        }.sorted()
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String,
                            ngramHeader: Safetensors.Header? = nil,
                            mtpHeader: Safetensors.Header? = nil) throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count + (mtpHeader?.tensors.count ?? 0))
        let allHeaders = shardHeaders + (mtpHeader.map { [$0] } ?? [])
        for h in allHeaders {
            for tensor in h.tensors {
                let canonicalName = canonicalSourceTensorName(tensor.name)
                registry[canonicalName] = canonicalName == tensor.name
                    ? tensor
                    : SourceTensor(
                        name: canonicalName,
                        shardPath: tensor.shardPath,
                        dtype: tensor.dtype,
                        shape: tensor.shape,
                        absoluteOffset: tensor.absoluteOffset,
                        sizeBytes: tensor.sizeBytes)
            }
        }
        if arch.modelFamily == "qwen4_exp_text" {
            let issues = qwen38RuntimeCompatibilityIssues(
                meta: meta, tensors: Array(registry.values))
            guard issues.isEmpty else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen3.8 runtime-incompatible quantization:\n"
                        + issues.joined(separator: "\n"))
            }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        var ngramBaseByShard: [Int: String] = [:]
        for (name, _) in registry {
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            let b = classify(name, numLayers: arch.numLayers,
                             modelFamily: arch.modelFamily)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .ngramShard(let shard):
                guard ngramBaseByShard.updateValue(name, forKey: shard) == nil else {
                    throw RepackError.configurationInvalid(
                        detail: "two n-gram tensors for shard \(shard)")
                }
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering())
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            registry: registry, meta: meta,
                                            arch: arch)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"] else {
                if bundle.isEmpty {
                    layerPlans.append(LayerFilePlan(layerIndex: layer,
                                                    path: (layersDir as NSString).appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                                                    expertsPerLayer: 0,
                                                    expertStride: 0,
                                                    subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(detail:
                    "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(path: path, layer: layer,
                                       gateName: gName, upName: uName, downName: dName,
                                       registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        let ngramShards = try planNgramShards(
            arch: arch, baseNames: ngramBaseByShard,
            registry: registry, meta: meta, outputDir: outputDir,
            ngramHeader: ngramHeader)
        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          ngramShards: ngramShards,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames)
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        isExcludedAuxiliaryTensorName(name)
    }

    private static func planNgramShards(
        arch: ArchInfo,
        baseNames: [Int: String],
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata,
        outputDir: String,
        ngramHeader: Safetensors.Header?
    ) throws -> [NgramShardFilePlan] {
        guard arch.modelFamily == "qwen4_exp_text" else {
            guard baseNames.isEmpty else {
                throw RepackError.configurationInvalid(
                    detail: "n-gram tensors require qwen4_exp_text")
            }
            return []
        }
        guard let qwen38 = arch.qwen38,
              qwen38.pleLayerIDs.count == 1,
              meta.baseMode.lowercased() == "affine",
              meta.baseBits == 4,
              meta.baseGroupSize == 32 else {
            throw RepackError.configurationInvalid(
                detail: "Qwen3.8 n-gram shards require complete affine Q4/group-32 metadata")
        }
        let ngramHeads = (qwen38.ngramSize - 1) * qwen38.headsPerNgram
        let headWidth = qwen38.pleEmbeddingSize / ngramHeads
        let totalVocabSize = (0..<ngramHeads).reduce(0) { partial, head in
            partial + nthPrime(after: qwen38.ngramVocabSizeBase - 1, count: head + 1)
        }
        let paddedVocabSize = alignUp(
            totalVocabSize, to: qwen38.ngramVocabSizeDivisor)
        let rows = paddedVocabSize / qwen38.ngramSplitParts
        guard rows * qwen38.ngramSplitParts == paddedVocabSize,
              headWidth * ngramHeads == qwen38.pleEmbeddingSize,
              headWidth % 8 == 0,
              headWidth % meta.baseGroupSize == 0 else {
            throw RepackError.configurationInvalid(detail: "invalid Qwen3.8 n-gram geometry")
        }
        let expectedWeightShape = [UInt64(rows), UInt64(headWidth / 8)]
        let expectedAffineShape = [UInt64(rows), UInt64(headWidth / meta.baseGroupSize)]
        let directory = (outputDir as NSString).appendingPathComponent("packed_ngrams")

        if baseNames.isEmpty {
            guard qwen38.ngramSidecar, let ngramHeader else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen3.8 n-gram shards are missing inline tensors and sidecar")
            }
            return try planNgramSidecar(
                arch: qwen38,
                header: ngramHeader,
                rows: UInt64(rows),
                splitParts: qwen38.ngramSplitParts,
                expectedWeightShape: expectedWeightShape,
                expectedAffineShape: expectedAffineShape,
                directory: directory)
        }
        guard ngramHeader == nil,
              baseNames.count == qwen38.ngramSplitParts else {
            throw RepackError.configurationInvalid(
                detail: "Qwen3.8 n-gram shards have conflicting inline and sidecar layouts")
        }

        return try (0..<qwen38.ngramSplitParts).map { shard in
            guard let name = baseNames[shard],
                  let sourceLayer = layerIndex(in: name),
                  qwen38.pleLayerIDs.contains(sourceLayer + 1),
                  let weight = registry[name],
                  let scales = registry[String(name.dropLast(".weight".count)) + ".scales"],
                  let biases = registry[String(name.dropLast(".weight".count)) + ".biases"],
                  weight.dtype == .u32, weight.shape == expectedWeightShape,
                  scales.dtype == .bf16, scales.shape == expectedAffineShape,
                  biases.dtype == .bf16, biases.shape == expectedAffineShape else {
                throw RepackError.configurationInvalid(
                    detail: "invalid or incomplete Qwen3.8 n-gram shard \(shard)")
            }
            let scalesOffset = roundUpToPage(weight.sizeBytes)
            let biasesOffset = roundUpToPage(scalesOffset + scales.sizeBytes)
            let fileSize = roundUpToPage(biasesOffset + biases.sizeBytes)
            return NgramShardFilePlan(
                shardIndex: shard,
                path: (directory as NSString).appendingPathComponent(
                    "shard_\(String(format: "%03d", shard)).bin"),
                fileSize: fileSize,
                weight: weight,
                weightOffset: 0,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset)
        }
    }

    private static func planNgramSidecar(
        arch: Qwen38ArchInfo,
        header: Safetensors.Header,
        rows: UInt64,
        splitParts: Int,
        expectedWeightShape: [UInt64],
        expectedAffineShape: [UInt64],
        directory: String
    ) throws -> [NgramShardFilePlan] {
        let totalRowsResult = rows.multipliedReportingOverflow(by: UInt64(splitParts))
        guard !totalRowsResult.overflow else {
            throw RepackError.configurationInvalid(
                detail: "Qwen3.8 n-gram sidecar row count overflows")
        }
        let totalRows = totalRowsResult.partialValue
        guard header.tensors.count == 3,
              let weight = header.tensors.first(where: { $0.name == "ngram.weight" }),
              let scales = header.tensors.first(where: { $0.name == "ngram.scales" }),
              let biases = header.tensors.first(where: { $0.name == "ngram.biases" }),
              weight.dtype == .u32,
              weight.shape == [totalRows, expectedWeightShape[1]],
              scales.dtype == .bf16,
              scales.shape == [totalRows, expectedAffineShape[1]],
              biases.dtype == .bf16,
              biases.shape == [totalRows, expectedAffineShape[1]] else {
            throw RepackError.configurationInvalid(
                detail: "invalid Qwen3.8 n-gram sidecar tensors")
        }
        guard let pleLayer = arch.pleLayerIDs.first else {
            throw RepackError.configurationInvalid(
                detail: "Qwen3.8 n-gram sidecar has no PLE layer")
        }
        let basePrefix = "language_model.model.layers.\(pleLayer - 1).ple.ple_embedding.ngram_embedding"
        let weightColumns = expectedWeightShape[1]
        let affineColumns = expectedAffineShape[1]
        let rowWeightBytes = weightColumns.multipliedReportingOverflow(by: 4)
        let rowAffineBytes = affineColumns.multipliedReportingOverflow(by: 2)
        guard !rowWeightBytes.overflow, !rowAffineBytes.overflow else {
            throw RepackError.configurationInvalid(
                detail: "Qwen3.8 n-gram sidecar row size overflows")
        }
        let directoryPath = directory

        return try (0..<splitParts).map { shard in
            func slice(_ source: SourceTensor,
                       name: String,
                       columns: UInt64,
                       rowBytes: UInt64) throws -> SourceTensor {
                let rowStart = UInt64(shard).multipliedReportingOverflow(by: rows)
                let byteStart = rowStart.partialValue.multipliedReportingOverflow(by: rowBytes)
                let size = rows.multipliedReportingOverflow(by: rowBytes)
                let absolute = source.absoluteOffset.addingReportingOverflow(byteStart.partialValue)
                guard !rowStart.overflow, !byteStart.overflow,
                      !size.overflow, !absolute.overflow else {
                    throw RepackError.configurationInvalid(
                        detail: "Qwen3.8 n-gram sidecar shard offset overflows")
                }
                return SourceTensor(
                    name: "\(basePrefix).shard_\(shard).\(name)",
                    shardPath: source.shardPath,
                    dtype: source.dtype,
                    shape: [rows, columns],
                    absoluteOffset: absolute.partialValue,
                    sizeBytes: size.partialValue)
            }

            let shardWeight = try slice(weight, name: "weight",
                                        columns: weightColumns,
                                        rowBytes: rowWeightBytes.partialValue)
            let shardScales = try slice(scales, name: "scales",
                                        columns: affineColumns,
                                        rowBytes: rowAffineBytes.partialValue)
            let shardBiases = try slice(biases, name: "biases",
                                        columns: affineColumns,
                                        rowBytes: rowAffineBytes.partialValue)
            let scalesOffset = roundUpToPage(shardWeight.sizeBytes)
            let biasesOffset = roundUpToPage(scalesOffset + shardScales.sizeBytes)
            let fileSize = roundUpToPage(biasesOffset + shardBiases.sizeBytes)
            return NgramShardFilePlan(
                shardIndex: shard,
                path: (directoryPath as NSString).appendingPathComponent(
                    "shard_\(String(format: "%03d", shard)).bin"),
                fileSize: fileSize,
                weight: shardWeight,
                weightOffset: 0,
                scales: shardScales,
                scalesOffset: scalesOffset,
                biases: shardBiases,
                biasesOffset: biasesOffset)
        }
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata,
                                         arch: ArchInfo) throws
                                        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for n in baseNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for (entryIndex, name) in baseNames.enumerated() {
            guard let weight = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let dtype = ietnyDtype(weight.dtype)
            if arch.modelFamily == "qwen3_5_moe_text",
               name.hasSuffix(".mlp.gate.weight"),
               weight.dtype == .bf16 {
                let rows = UInt64(arch.numExperts)
                let columns = UInt64(arch.hiddenSize)
                let groupSize = 64
                guard weight.shape == [rows, columns],
                      arch.hiddenSize % groupSize == 0,
                      weight.sizeBytes == rows * columns
                          * UInt64(MemoryLayout<UInt16>.size) else {
                    throw RepackError.shapeMismatch(
                        name: name,
                        detail: "expected BF16 Qwen3.6 router shape [\(rows), \(columns)]")
                }
                let outputSpec = QuantSpec(bits: 8, groupSize: groupSize)
                let weightSize = rows * columns
                let companionSize = rows * (columns / UInt64(groupSize))
                    * UInt64(MemoryLayout<UInt16>.size)
                let weightOffset = fileCursor
                let scaleOffset = weightOffset + weightSize
                let biasOffset = scaleOffset + companionSize
                fileCursor = biasOffset + companionSize
                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4([rows, columns]),
                    fileOffset: weightOffset, sizeBytes: weightSize,
                    scaleOffset: scaleOffset, scaleSize: companionSize,
                    biasOffset: biasOffset, biasSize: companionSize,
                    quantSpec: outputSpec,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil,
                    sourceQuantSpec: QuantSpec(bits: 16, groupSize: groupSize)))
                continue
            }
            if arch.modelFamily == "qwen4_exp_text",
               !preservesSourceMTPRepresentation(name, modelFamily: arch.modelFamily),
               name.hasSuffix(".mlp.gate.weight"),
               weight.dtype == .u32 {
                let base = String(name.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"],
                      let biases = registry[base + ".biases"],
                      scales.dtype == .bf16, biases.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(
                        name: name,
                        detail: "Qwen3.8 router requires BF16 scale and bias companions")
                }
                let sourceSpec = sourceQuantSpec(for: name, meta: meta)
                let conversionLayout = try CanonicalQuantization.layout(
                    shape: weight.shape, source: sourceSpec)
                let logicalShape = logicalShape(
                    forPackedSource: weight.shape, bits: sourceSpec.bits)
                guard logicalShape == [UInt64(arch.numExperts), UInt64(arch.hiddenSize)] else {
                    throw RepackError.shapeMismatch(
                        name: name,
                        detail: "expected Qwen3.8 router shape [\(arch.numExperts), \(arch.hiddenSize)], got \(logicalShape)")
                }
                let weightSize = UInt64(
                    conversionLayout.rowCount * conversionLayout.inputWidth
                        * MemoryLayout<UInt16>.size)
                let offset = fileCursor
                fileCursor += weightSize
                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.bf16.rawValue,
                    logicalShape4: padTo4(logicalShape),
                    fileOffset: offset, sizeBytes: weightSize,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: scales,
                    sourceBiases: biases, sourceQuantSpec: sourceSpec,
                    sourceStagingPath: ((path as NSString).deletingLastPathComponent as NSString)
                        .appendingPathComponent("source-staging/\(entryIndex).bin")))
                continue
            }
            if isBF16Qwen38Projection(name) {
                guard weight.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(
                        name: name, detail: "expected BF16 PLE projection, got \(weight.dtype)")
                }
                if preservesSourceMTPRepresentation(
                    name, modelFamily: arch.modelFamily) {
                    let offset = fileCursor
                    fileCursor += weight.sizeBytes
                    entries.append(ResidentEntry(
                        name: name, dtype: GTurboFormatV1.DType.bf16.rawValue,
                        logicalShape4: padTo4(weight.shape),
                        fileOffset: offset, sizeBytes: weight.sizeBytes,
                        scaleOffset: 0, scaleSize: 0,
                        biasOffset: 0, biasSize: 0,
                        quantSpec: nil,
                        sourceWeight: weight, sourceScales: nil, sourceBiases: nil,
                        sourceQuantSpec: QuantSpec(bits: 16, groupSize: 32)))
                    continue
                }
                let wSize = try CanonicalQuantization.bf16OutputWeightBytes(
                    shape: weight.shape)
                let companionSize = try CanonicalQuantization.bf16OutputCompanionBytes(
                    shape: weight.shape)
                let wOff = fileCursor
                let sOff = wOff + wSize
                let bOff = sOff + companionSize
                fileCursor = bOff + companionSize
                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: companionSize,
                    biasOffset: bOff, biasSize: companionSize,
                    quantSpec: CanonicalQuantization.target,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil,
                    sourceQuantSpec: QuantSpec(bits: 16, groupSize: 32),
                    sourceStagingPath: ((path as NSString).deletingLastPathComponent as NSString)
                        .appendingPathComponent("source-staging/\(entryIndex).bin")))
                continue
            }
            let isQuantizedPacked = (weight.dtype == .u32) && name.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(name.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let sourceSpec = sourceQuantSpec(for: name, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: sourceSpec.bits)
                let outputSpec: QuantSpec
                if preservesSourceMTPRepresentation(
                    name, modelFamily: arch.modelFamily) {
                    outputSpec = sourceSpec
                } else {
                    outputSpec = residentOutputQuantSpec(
                        for: name, modelFamily: arch.modelFamily)
                }
                let wSize: UInt64
                let sSize: UInt64
                let bSize: UInt64
                if sourceSpec == outputSpec {
                    wSize = weight.sizeBytes
                    sSize = scales.sizeBytes
                    bSize = biases.sizeBytes
                } else if outputSpec == CanonicalQuantization.target {
                    wSize = try CanonicalQuantization.outputWeightBytes(
                        shape: weight.shape, source: sourceSpec)
                    sSize = try CanonicalQuantization.outputCompanionBytes(
                        shape: weight.shape, source: sourceSpec)
                    bSize = sSize
                } else {
                    throw RepackError.configurationInvalid(
                        detail: "Qwen3.6 shared expert gate requires affine-8/group-64 source")
                }

                let wOff = fileCursor
                let sOff = wOff + wSize
                let bOff = sOff + sSize
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: outputSpec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases,
                    sourceQuantSpec: sourceSpec,
                    sourceStagingPath: sourceSpec == outputSpec ? nil :
                        ((path as NSString).deletingLastPathComponent as NSString)
                            .appendingPathComponent("source-staging/\(entryIndex).bin")))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: GTurboFormatV1.DType.u32.rawValue,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        case .i64: 4
        }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, packedWords]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        guard !source.isEmpty, bits > 0 else { return source }
        let packedBits = source[source.count - 1] * 32
        guard packedBits % UInt64(bits) == 0 else { return source }
        var out = source
        out[out.count - 1] = packedBits / UInt64(bits)
        return out
    }

    private static func alignUp(_ value: Int, to alignment: Int) -> Int {
        ((value + alignment - 1) / alignment) * alignment
    }

    private static func nthPrime(after start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0..<count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    private static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor <= value / divisor {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm.
    private static func lmResidentOrdering() -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot = slotRank(in: n)
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight") || n.contains(".mlp.gate.weight") {
            return 6
        }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.hasSuffix(".input_layernorm.weight") { return 12 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 13 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 14 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 15 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 18 }
        if n.hasSuffix(".layer_scalar")           { return 19 }
        return 100
    }
}
