import Metal

package final class QwenVisionPrimitives {
    private let patchBiasPosition: MTLComputePipelineState
    private let layerNorm: MTLComputePipelineState
    private let qkvRoPE: MTLComputePipelineState
    private let attention: MTLComputePipelineState
    private let biasGELU: MTLComputePipelineState
    private let residual: MTLComputePipelineState
    private let biasAdd: MTLComputePipelineState
    private let mergeGather: MTLComputePipelineState
    private let config: QwenVisionConfig

    package init(context: MetalContext, config: QwenVisionConfig = QwenVisionConfig()) throws {
        self.config = config
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "qwen_vision")
        func makePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return try context.device.makeComputePipelineState(function: function)
        }
        patchBiasPosition = try makePipeline("qwen_vision_patch_bias_position")
        layerNorm = try makePipeline("qwen_vision_layer_norm")
        qkvRoPE = try makePipeline("qwen_vision_qkv_rope")
        attention = try makePipeline("qwen_vision_attention")
        biasGELU = try makePipeline("qwen_vision_bias_gelu")
        residual = try makePipeline("qwen_vision_residual")
        biasAdd = try makePipeline("qwen_vision_bias_add")
        mergeGather = try makePipeline("qwen_vision_merge_gather")
    }

    package func encodePatchBiasPosition(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        weights: MTLBuffer,
        positionOffset: Int,
        biasOffset: Int,
        positions: MTLBuffer,
        rows: Int,
        gridWidth: Int,
        gridHeight: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(patchBiasPosition)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: positionOffset, index: 1)
        encoder.setBuffer(weights, offset: biasOffset, index: 2)
        encoder.setBuffer(positions, offset: 0, index: 3)
        var rowCount = UInt32(rows)
        var width = UInt32(gridWidth)
        var height = UInt32(gridHeight)
        var hiddenSize = UInt32(config.hiddenSize)
        var positionGridSide = UInt32(config.positionGridSize)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&height, length: 4, index: 6)
        encoder.setBytes(&hiddenSize, length: 4, index: 7)
        encoder.setBytes(&positionGridSide, length: 4, index: 8)
        dispatch1D(encoder, count: rows * config.hiddenSize, pipeline: patchBiasPosition)
    }

    package func encodeLayerNorm(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        scale: MTLBuffer,
        scaleOffset: Int,
        bias: MTLBuffer,
        biasOffset: Int,
        output: MTLBuffer,
        rows: Int,
        width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(layerNorm)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: scaleOffset, index: 1)
        encoder.setBuffer(bias, offset: biasOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rowCount = UInt32(rows)
        var dimension = UInt32(width)
        var epsilon = config.layerNormEpsilon
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&dimension, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeQKVRotary(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        weights: MTLBuffer,
        qBiasOffset: Int,
        kBiasOffset: Int,
        vBiasOffset: Int,
        positions: MTLBuffer,
        rows: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qkvRoPE)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: qBiasOffset, index: 3)
        encoder.setBuffer(weights, offset: kBiasOffset, index: 4)
        encoder.setBuffer(weights, offset: vBiasOffset, index: 5)
        encoder.setBuffer(positions, offset: 0, index: 6)
        var rowCount = UInt32(rows)
        var headCount = UInt32(config.numHeads)
        var theta = config.ropeTheta
        encoder.setBytes(&rowCount, length: 4, index: 7)
        encoder.setBytes(&headCount, length: 4, index: 8)
        encoder.setBytes(&theta, length: 4, index: 9)
        dispatch1D(encoder, count: rows * config.numHeads, pipeline: qkvRoPE)
    }

    package func encodeAttention(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        output: MTLBuffer,
        rows: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(attention)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rowCount = UInt32(rows)
        var headCount = UInt32(config.numHeads)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&headCount, length: 4, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: config.numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeBiasGELU(
        commandBuffer: MTLCommandBuffer,
        values: MTLBuffer,
        bias: MTLBuffer,
        biasOffset: Int,
        rows: Int,
        width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(biasGELU)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        var count = UInt32(rows * width)
        var dimension = UInt32(width)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.setBytes(&dimension, length: 4, index: 3)
        dispatch1D(encoder, count: rows * width, pipeline: biasGELU)
    }

    package func encodeResidual(
        commandBuffer: MTLCommandBuffer,
        state: MTLBuffer,
        branch: MTLBuffer,
        bias: MTLBuffer,
        biasOffset: Int,
        rows: Int,
        width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residual)
        encoder.setBuffer(state, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: biasOffset, index: 2)
        encoder.setBuffer(state, offset: 0, index: 3)
        var count = UInt32(rows * width)
        var dimension = UInt32(width)
        encoder.setBytes(&count, length: 4, index: 4)
        encoder.setBytes(&dimension, length: 4, index: 5)
        dispatch1D(encoder, count: rows * width, pipeline: residual)
    }

    package func encodeBiasAdd(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        bias: MTLBuffer,
        biasOffset: Int,
        output: MTLBuffer,
        rows: Int,
        width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(biasAdd)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var count = UInt32(rows * width)
        var dimension = UInt32(width)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.setBytes(&dimension, length: 4, index: 4)
        dispatch1D(encoder, count: rows * width, pipeline: biasAdd)
    }

    package func encodeMergeGather(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        tokenCount: Int
    ) {
        let width = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
        let count = tokenCount * width
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mergeGather)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var outputCount = UInt32(count)
        var hiddenSize = UInt32(config.hiddenSize)
        var mergeUnit = UInt32(config.spatialMergeSize * config.spatialMergeSize)
        encoder.setBytes(&outputCount, length: 4, index: 2)
        encoder.setBytes(&hiddenSize, length: 4, index: 3)
        encoder.setBytes(&mergeUnit, length: 4, index: 4)
        dispatch1D(encoder, count: count, pipeline: mergeGather)
    }

    private func dispatch1D(
        _ encoder: MTLComputeCommandEncoder,
        count: Int,
        pipeline: MTLComputePipelineState
    ) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
