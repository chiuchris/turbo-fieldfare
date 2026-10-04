import Foundation
import Metal
import TurboFieldfareFormat

public struct QwenVisionFeatures: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let tokenCount: Int
    public let hiddenSize: Int
    public let geometry: QwenImageGeometry
    public let gpuNanoseconds: UInt64
    public let scratchBytes: Int
    public let preprocessingWallNanoseconds: UInt64
}

public final class QwenVisionRuntime {
    public let config: QwenVisionConfig

    private let context: MetalContext
    private let store: VisionWeightStore
    private let useLease: VisionPackUseLease
    private let linear: VisionLinearBF16
    private let primitives: QwenVisionPrimitives
    private let preprocessor: QwenImagePreprocessor

    private static let patchTensorNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.patch_embed.proj.bias",
        "vision_tower.pos_embed.weight",
    ]

    private static let layerTensorSuffixes = [
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
    ]

    private static let mergerTensorNames = [
        "vision_tower.merger.norm.weight",
        "vision_tower.merger.norm.bias",
        "vision_tower.merger.linear_fc1.weight",
        "vision_tower.merger.linear_fc1.bias",
        "vision_tower.merger.linear_fc2.weight",
        "vision_tower.merger.linear_fc2.bias",
    ]

    public static func open(
        textModelURL: URL,
        context: MetalContext,
        visionPackURL: URL? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> QwenVisionRuntime {
        try VisionRuntime.requireSupportedDevice(context.device)
        let textManifest = try ManifestReader.load(
            directoryURL: textModelURL,
            expecting: .qwen36MoeText)
        guard let source = textManifest.sourceSnapshotHash else {
            throw VisionRuntimeError.invalidInput("text model has no source identity")
        }
        let textManifestSHA = try Sha256Verifier.hashFile(
            at: textModelURL.appendingPathComponent("manifest.json"))
        let companion: URL
        if let visionPackURL {
            companion = visionPackURL
        } else {
            companion = try VisionPackLocation.companionURL(forTextModel: textModelURL)
        }
        guard FileManager.default.fileExists(atPath: companion.path) else {
            throw VisionPackError.packNotFound(companion.path)
        }
        let visionManifestPath = companion
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile).path
        guard FileManager.default.fileExists(atPath: visionManifestPath) else {
            throw VisionPackError.invalidMetadata(
                "no \(GTurboVisionFormatV1.manifestFile) in \(companion.path)")
        }

        let useLease = try VisionPackUseLease.acquireShared(companionURL: companion)
        let store = try VisionWeightStore.open(
            directoryURL: companion,
            compatibleTextSourceSnapshotHash: source,
            compatibleTextManifestSha256: textManifestSHA)
        guard store.manifest.artifactKind == GTurboVisionFormatV1.qwenArtifactKind else {
            throw VisionPackError.invalidMetadata("companion is not a Qwen vision pack")
        }
        return try QwenVisionRuntime(
            context: context,
            store: store,
            useLease: useLease,
            environment: environment)
    }

    private init(
        context: MetalContext,
        store: VisionWeightStore,
        useLease: VisionPackUseLease,
        environment: [String: String]
    ) throws {
        self.context = context
        self.store = store
        self.useLease = useLease
        config = QwenVisionConfig()
        linear = try VisionLinearBF16(context: context, environment: environment)
        primitives = try QwenVisionPrimitives(context: context)
        preprocessor = QwenImagePreprocessor(device: context.device)
    }

    public func encode(fileURL: URL) throws -> QwenVisionFeatures {
        try encode(preprocessor.preprocess(fileURL: fileURL))
    }

    public func encode(_ pixels: QwenImagePixels) throws -> QwenVisionFeatures {
        let rows = pixels.geometry.patchCount
        let tokenCount = pixels.geometry.tokenCount
        guard rows > 0, rows.isMultiple(of: 4), tokenCount * 4 == rows,
              pixels.geometry.patchGridWidth > 0, pixels.geometry.patchGridHeight > 0 else {
            throw VisionRuntimeError.invalidInput("invalid Qwen patch grid")
        }

        let states = try makeBuffer(rows: rows, width: config.hiddenSize)
        let normalized = try makeBuffer(rows: rows, width: config.hiddenSize)
        let q = try makeBuffer(rows: rows, width: config.hiddenSize)
        let k = try makeBuffer(rows: rows, width: config.hiddenSize)
        let v = try makeBuffer(rows: rows, width: config.hiddenSize)
        let attention = try makeBuffer(rows: rows, width: config.hiddenSize)
        let branch = try makeBuffer(rows: rows, width: config.hiddenSize)
        let intermediate = try makeBuffer(rows: rows, width: config.intermediateSize)
        let mergerInput = try makeBuffer(rows: tokenCount, width: config.mergerInputSize)
        let mergerHidden = try makeBuffer(rows: tokenCount, width: config.mergerInputSize)
        let features = try makeBuffer(rows: tokenCount, width: config.outputHiddenSize)

        var mappedRegions: [VisionMappedWeightRegion] = []
        let patchWeights = try mapRegion(Self.patchTensorNames, retaining: &mappedRegions)
        let patchWeightOffset = try patchWeights.offset(
            of: "vision_tower.patch_embed.proj.weight")
        let patchCommandBuffer = try makeCommandBuffer()
        linear.encode(
            commandBuffer: patchCommandBuffer,
            input: pixels.patchesBF16,
            weights: patchWeights.buffer,
            weightsOffset: patchWeightOffset,
            output: states,
            m: rows,
            n: config.hiddenSize,
            k: QwenImageGeometry.patchVectorDimension)
        primitives.encodePatchBiasPosition(
            commandBuffer: patchCommandBuffer,
            hidden: states,
            weights: patchWeights.buffer,
            positionOffset: try patchWeights.offset(of: "vision_tower.pos_embed.weight"),
            biasOffset: try patchWeights.offset(of: "vision_tower.patch_embed.proj.bias"),
            positions: pixels.positionsInt32x2,
            rows: rows,
            gridWidth: pixels.geometry.patchGridWidth,
            gridHeight: pixels.geometry.patchGridHeight)
        patchCommandBuffer.commit()
        patchCommandBuffer.waitUntilCompleted()
        if let error = patchCommandBuffer.error {
            throw VisionRuntimeError.commandFailed(String(describing: error))
        }
        var gpuNanoseconds = UInt64(max(
            0, patchCommandBuffer.gpuEndTime - patchCommandBuffer.gpuStartTime)
            * 1_000_000_000)

        for layer in 0..<config.numLayers {
            let prefix = "vision_tower.blocks.\(layer)."
            let expectedNames = Self.layerTensorSuffixes.map { prefix + $0 }
            let actualNames = Set(store.tensorNames(withPrefix: prefix))
            guard Set(expectedNames) == actualNames else {
                throw VisionRuntimeError.invalidInput(
                    "vision block \(layer) has an unexpected tensor set")
            }
            let weights = try mapRegion(expectedNames, retaining: &mappedRegions)
            func offset(_ suffix: String) throws -> Int {
                try weights.offset(of: prefix + suffix)
            }

            let command = try makeCommandBuffer()
            primitives.encodeLayerNorm(
                commandBuffer: command,
                input: states,
                scale: weights.buffer,
                scaleOffset: try offset("norm1.weight"),
                bias: weights.buffer,
                biasOffset: try offset("norm1.bias"),
                output: normalized,
                rows: rows,
                width: config.hiddenSize)

            let qkvWeightOffset = try offset("attn.qkv.weight")
            let singleProjectionBytes = config.hiddenSize * config.hiddenSize
                * MemoryLayout<UInt16>.stride
            for (buffer, projectionOffset) in [
                (q, qkvWeightOffset),
                (k, qkvWeightOffset + singleProjectionBytes),
                (v, qkvWeightOffset + singleProjectionBytes * 2),
            ] {
                linear.encode(
                    commandBuffer: command,
                    input: normalized,
                    weights: weights.buffer,
                    weightsOffset: projectionOffset,
                    output: buffer,
                    m: rows,
                    n: config.hiddenSize,
                    k: config.hiddenSize)
            }
            primitives.encodeQKVRotary(
                commandBuffer: command,
                q: q,
                k: k,
                v: v,
                weights: weights.buffer,
                qBiasOffset: try offset("attn.qkv.bias"),
                kBiasOffset: try offset("attn.qkv.bias")
                    + config.hiddenSize * MemoryLayout<UInt16>.stride,
                vBiasOffset: try offset("attn.qkv.bias")
                    + config.hiddenSize * 2 * MemoryLayout<UInt16>.stride,
                positions: pixels.positionsInt32x2,
                rows: rows)
            primitives.encodeAttention(
                commandBuffer: command,
                q: q,
                k: k,
                v: v,
                output: attention,
                rows: rows)
            linear.encode(
                commandBuffer: command,
                input: attention,
                weights: weights.buffer,
                weightsOffset: try offset("attn.proj.weight"),
                output: branch,
                m: rows,
                n: config.hiddenSize,
                k: config.hiddenSize)
            primitives.encodeResidual(
                commandBuffer: command,
                state: states,
                branch: branch,
                bias: weights.buffer,
                biasOffset: try offset("attn.proj.bias"),
                rows: rows,
                width: config.hiddenSize)
            primitives.encodeLayerNorm(
                commandBuffer: command,
                input: states,
                scale: weights.buffer,
                scaleOffset: try offset("norm2.weight"),
                bias: weights.buffer,
                biasOffset: try offset("norm2.bias"),
                output: normalized,
                rows: rows,
                width: config.hiddenSize)
            linear.encode(
                commandBuffer: command,
                input: normalized,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.linear_fc1.weight"),
                output: intermediate,
                m: rows,
                n: config.intermediateSize,
                k: config.hiddenSize)
            primitives.encodeBiasGELU(
                commandBuffer: command,
                values: intermediate,
                bias: weights.buffer,
                biasOffset: try offset("mlp.linear_fc1.bias"),
                rows: rows,
                width: config.intermediateSize)
            linear.encode(
                commandBuffer: command,
                input: intermediate,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.linear_fc2.weight"),
                output: branch,
                m: rows,
                n: config.hiddenSize,
                k: config.intermediateSize)
            primitives.encodeResidual(
                commandBuffer: command,
                state: states,
                branch: branch,
                bias: weights.buffer,
                biasOffset: try offset("mlp.linear_fc2.bias"),
                rows: rows,
                width: config.hiddenSize)
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error {
                throw VisionRuntimeError.commandFailed(String(describing: error))
            }
            gpuNanoseconds += UInt64(max(
                0, command.gpuEndTime - command.gpuStartTime) * 1_000_000_000)
        }

        let mergerWeights = try mapRegion(Self.mergerTensorNames, retaining: &mappedRegions)
        let mergeCommand = try makeCommandBuffer()
        primitives.encodeLayerNorm(
            commandBuffer: mergeCommand,
            input: states,
            scale: mergerWeights.buffer,
            scaleOffset: try mergerWeights.offset(of: "vision_tower.merger.norm.weight"),
            bias: mergerWeights.buffer,
            biasOffset: try mergerWeights.offset(of: "vision_tower.merger.norm.bias"),
            output: normalized,
            rows: rows,
            width: config.hiddenSize)
        primitives.encodeMergeGather(
            commandBuffer: mergeCommand,
            input: normalized,
            output: mergerInput,
            tokenCount: tokenCount)
        linear.encode(
            commandBuffer: mergeCommand,
            input: mergerInput,
            weights: mergerWeights.buffer,
            weightsOffset: try mergerWeights.offset(
                of: "vision_tower.merger.linear_fc1.weight"),
            output: mergerHidden,
            m: tokenCount,
            n: config.mergerInputSize,
            k: config.mergerInputSize)
        primitives.encodeBiasGELU(
            commandBuffer: mergeCommand,
            values: mergerHidden,
            bias: mergerWeights.buffer,
            biasOffset: try mergerWeights.offset(
                of: "vision_tower.merger.linear_fc1.bias"),
            rows: tokenCount,
            width: config.mergerInputSize)
        linear.encode(
            commandBuffer: mergeCommand,
            input: mergerHidden,
            weights: mergerWeights.buffer,
            weightsOffset: try mergerWeights.offset(
                of: "vision_tower.merger.linear_fc2.weight"),
            output: features,
            m: tokenCount,
            n: config.outputHiddenSize,
            k: config.mergerInputSize)
        primitives.encodeBiasAdd(
            commandBuffer: mergeCommand,
            input: features,
            bias: mergerWeights.buffer,
            biasOffset: try mergerWeights.offset(
                of: "vision_tower.merger.linear_fc2.bias"),
            output: features,
            rows: tokenCount,
            width: config.outputHiddenSize)
        mergeCommand.commit()
        mergeCommand.waitUntilCompleted()
        if let error = mergeCommand.error {
            throw VisionRuntimeError.commandFailed(String(describing: error))
        }
        gpuNanoseconds += UInt64(max(
            0, mergeCommand.gpuEndTime - mergeCommand.gpuStartTime) * 1_000_000_000)

        let buffers = [
            states, normalized, q, k, v, attention, branch, intermediate,
            mergerInput, mergerHidden, features,
        ]
        return QwenVisionFeatures(
            buffer: features,
            tokenCount: tokenCount,
            hiddenSize: config.outputHiddenSize,
            geometry: pixels.geometry,
            gpuNanoseconds: gpuNanoseconds,
            scratchBytes: buffers.reduce(0) { $0 + $1.length },
            preprocessingWallNanoseconds: pixels.wallNanoseconds)
    }

    private func mapRegion(
        _ names: [String],
        retaining regions: inout [VisionMappedWeightRegion]
    ) throws -> VisionMappedWeightRegion {
        let region = try store.mapRegion(tensorNames: names, device: context.device)
        regions.append(region)
        return region
    }

    private func makeBuffer(rows: Int, width: Int) throws -> MTLBuffer {
        let (elements, elementOverflow) = rows.multipliedReportingOverflow(by: width)
        let (bytes, byteOverflow) = elements.multipliedReportingOverflow(
            by: MemoryLayout<UInt16>.stride)
        guard !elementOverflow, !byteOverflow,
              UInt64(bytes) <= UInt64(context.device.maxBufferLength),
              let buffer = context.device.makeBuffer(
                length: bytes, options: .storageModePrivate) else {
            throw VisionRuntimeError.invalidInput(
                "Qwen vision scratch allocation failed for \(rows) by \(width)")
        }
        return buffer
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        return commandBuffer
    }

}
