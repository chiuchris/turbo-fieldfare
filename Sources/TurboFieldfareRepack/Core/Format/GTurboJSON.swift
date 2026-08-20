import Foundation
import TurboFieldfareFormat

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = GTurboFormatV1.magic
    static let versionMajor = GTurboFormatV1.versionMajor
    static let versionMinor = GTurboFormatV1.versionMinor

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
        var deltaNet: Int = 4
        var sharedExpertGate: Int = 8
        var lmHead: Int = 4
        var norm: Int = 16
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        if plan.arch.modelFamily == "qwen3_5_moe_text" {
            return try encodeManifestV2(
                plan: plan,
                modelID: modelID,
                sourceSnapshotHash: sourceSnapshotHash,
                files: files,
                expertsPerLayer: expertsPerLayer,
                numLayers: numLayers,
                expertStride: expertStride,
                bitWidths: bitWidths)
        }
        return try encodeManifestV1(
            plan: plan,
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            files: files,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride,
            bitWidths: bitWidths)
    }

    private static func encodeManifestV1(plan: RepackPlan,
                                         modelID: String,
                                         sourceSnapshotHash: String,
                                         files: [(relativePath: String, info: FileEntry)],
                                         expertsPerLayer: Int,
                                         numLayers: Int,
                                         expertStride: UInt64,
                                         bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        let bitWidthsByQuantSlot = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        let wireArch = GTurboManifestArchV1(
            hiddenSize: arch.hiddenSize,
            ffnIntermediate: arch.intermediateSize,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            hiddenActivation: arch.hiddenActivation,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map(Int.init))
        func slot(_ name: String) throws -> GTurboManifestQuantSlotV1 {
            guard let weightBits = bitWidthsByQuantSlot[name] else {
                throw RepackError.configurationInvalid(
                    detail: "missing manifest quant slot bit width for \(name)")
            }
            return GTurboManifestQuantSlotV1(
                weightBits: weightBits,
                scheme: plan.baseMode,
                scaleType: "BF16",
                biasType: "BF16",
                groupSize: plan.baseGroupSize)
        }
        let quant = GTurboManifestQuantV1(
            embedding: try slot("embedding"),
            attention: try slot("attention"),
            router: try slot("router"),
            sharedExpert: try slot("sharedExpert"),
            routedExpert: try slot("routedExpert"))
        var wireFiles: [String: GTurboManifestFileV1] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                GTurboManifestFileV1(size: file.info.size, sha256: file.info.sha256),
                forKey: file.relativePath) == nil else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate manifest file entry \(file.relativePath)")
            }
        }
        return try GTurboManifestCodec.encode(GTurboManifestV1(
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: wireArch,
            quant: quant,
            files: wireFiles,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride,
            bitWidthOverridesHonored: plan.bitsOverrideCount))
    }

    private static func encodeManifestV2(plan: RepackPlan,
                                         modelID: String,
                                         sourceSnapshotHash: String,
                                         files: [(relativePath: String, info: FileEntry)],
                                         expertsPerLayer: Int,
                                         numLayers: Int,
                                         expertStride: UInt64,
                                         bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        let fullAttention = GTurboManifestV2FullAttention(
            queryHeads: arch.numHeads,
            keyValueHeads: arch.numKVHeads,
            headDim: arch.fullHeadDim,
            ropeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor)
        let gatedDeltaNet = GTurboManifestV2GatedDeltaNet(
            keyHeads: arch.linearNumKeyHeads,
            valueHeads: arch.linearNumValueHeads,
            keyHeadDim: arch.linearKeyHeadDim,
            valueHeadDim: arch.linearValueHeadDim,
            convolutionKernel: arch.linearConvKernelDim,
            stateDType: "FP32")
        let quantSlot: (Int, String, String, String, Int) -> GTurboManifestQuantSlotV2 = {
            bits, scheme, scaleType, biasType, groupSize in
            GTurboManifestQuantSlotV2(
                weightBits: bits,
                scheme: scheme,
                scaleType: scaleType,
                biasType: biasType,
                groupSize: groupSize)
        }
        let quant = GTurboManifestQuantV2(roles: [
            "embedding": quantSlot(bitWidths.embedding, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "attention": quantSlot(bitWidths.attention, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "deltaNet": quantSlot(bitWidths.deltaNet, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "router": quantSlot(bitWidths.router, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "sharedExpert": quantSlot(bitWidths.sharedExpert, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "sharedExpertGate": quantSlot(bitWidths.sharedExpertGate, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "routedExpert": quantSlot(bitWidths.routedExpert, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "lmHead": quantSlot(bitWidths.lmHead, plan.baseMode, "BF16", "BF16", plan.baseGroupSize),
            "norm": quantSlot(bitWidths.norm, "none", "none", "none", 1),
        ])
        let wireFiles = Dictionary(uniqueKeysWithValues: files.map {
            ($0.relativePath, GTurboManifestFileV1(size: $0.info.size, sha256: $0.info.sha256))
        })
        let wireArch = GTurboManifestV2Arch(
            modelFamily: arch.modelFamily,
            hiddenSize: arch.hiddenSize,
            vocabSize: arch.vocabSize,
            numLayers: arch.numLayers,
            layerKinds: arch.fullAttentionLayerMask.map {
                $0 == 1 ? "fullAttention" : "gatedDeltaNet"
            },
            numRoutedExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            routedExpertIntermediateSize: arch.moeIntermediateSize,
            sharedExpertIntermediateSize: arch.intermediateSize,
            routerActivation: "sigmoid",
            routedExpertActivation: arch.hiddenActivation,
            sharedExpertActivation: arch.hiddenActivation,
            sharedExpertGateActivation: "sigmoid",
            tieWordEmbeddings: arch.tieWordEmbeddings,
            fullAttention: fullAttention,
            gatedDeltaNet: gatedDeltaNet,
            finalRopeTheta: arch.fullRopeTheta)
        return try GTurboManifestV2Codec.encode(GTurboManifestV2(
            flags: ["streamingPresent": true, "untiedHead": true],
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: wireArch,
            quant: quant,
            files: wireFiles,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride))
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layers: [GTurboLayerV1] = []
        layers.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [GTurboExpertV1] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let physicalRank = lp.physicalRank(for: e)
                let base = UInt64(physicalRank) * lp.expertStride
                var tensors: [String: GTurboSubTensorV1] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard slice.dtype == GTurboFormatV1.DType.u32.rawValue
                            || slice.dtype == GTurboFormatV1.DType.bf16.rawValue else {
                        throw RepackError.configurationInvalid(
                            detail: "unsupported packed expert dtype \(slice.dtype) for \(key)")
                    }
                    let shape = try slice.logicalShape.enumerated().map { index, value in
                        guard value <= UInt64(UInt32.max) else {
                            throw RepackError.configurationInvalid(
                                detail: "packed expert shape[\(index)] exceeds UInt32")
                        }
                        return UInt32(value)
                    }
                    let previous = tensors.updateValue(GTurboSubTensorV1(
                        offset: slice.offsetInExpertBlob,
                        size: slice.sizeInExpertBlob,
                        dtype: slice.dtype == GTurboFormatV1.DType.u32.rawValue ? "U32" : "BF16",
                        shape: shape,
                        bits: slice.bitsForWeights), forKey: key)
                    guard previous == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "duplicate packed expert tensor key \(key)")
                    }
                }
                experts.append(GTurboExpertV1(
                    expert: e,
                    physicalRank: nil,
                    offset: base,
                    size: lp.expertStride,
                    tensors: tensors))
            }
            layers.append(GTurboLayerV1(layer: lp.layerIndex,
                                        file: layerFile,
                                        experts: experts))
        }
        return try GTurboPackedExpertsLayoutCodec.encode(
            GTurboPackedExpertsLayoutV1(
                expertStride: expertStride,
                numLayers: arch.numLayers,
                expertsPerLayer: plan.layers.first?.expertsPerLayer ?? 0,
                layers: layers))
    }
}
