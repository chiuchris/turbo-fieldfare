import Metal

final class QwenElementwise {
    private let deltaParametersPSO: MTLComputePipelineState
    private let gatedNormPSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState
    private let prefillDeltaParametersPSO: MTLComputePipelineState
    private let prefillGatedNormPSO: MTLComputePipelineState
    private let prefillResidualAddPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.deltaParametersPSO = try context.pipeline("qwen_delta_parameters")
        self.gatedNormPSO = try context.pipeline("qwen_gated_rmsnorm")
        self.residualAddPSO = try context.pipeline("qwen_residual_add")
        self.prefillDeltaParametersPSO = try context.pipeline("qwen_prefill_delta_parameters")
        self.prefillGatedNormPSO = try context.pipeline("qwen_prefill_gated_rmsnorm")
        self.prefillResidualAddPSO = try context.pipeline("qwen_prefill_residual_add")
    }

    func encodeDeltaParameters(commandBuffer: MTLCommandBuffer,
                               a: MTLBuffer,
                               betaInput: MTLBuffer,
                               aLog: MTLBuffer,
                               aLogOffset: Int = 0,
                               dtBias: MTLBuffer,
                               dtBiasOffset: Int = 0,
                               decay: MTLBuffer,
                               beta: MTLBuffer,
                               count: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        encoder.setComputePipelineState(deltaParametersPSO)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(betaInput, offset: 0, index: 1)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 2)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 3)
        encoder.setBuffer(decay, offset: 0, index: 4)
        encoder.setBuffer(beta, offset: 0, index: 5)
        var countValue = count
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(deltaParametersPSO.maxTotalThreadsPerThreadgroup), 32),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeGatedNorm(commandBuffer: MTLCommandBuffer,
                          input: MTLBuffer,
                          gate: MTLBuffer,
                          weight: MTLBuffer,
                          weightOffset: Int = 0,
                          output: MTLBuffer,
                          headCount: UInt32,
                          headDimension: UInt32,
                          epsilon: Float) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        encoder.setComputePipelineState(gatedNormPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var heads = headCount
        var dimension = headDimension
        var epsilonValue = epsilon
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(headCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(gatedNormPSO.maxTotalThreadsPerThreadgroup), 32),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                           lhs: MTLBuffer,
                           rhs: MTLBuffer,
                           output: MTLBuffer,
                           count: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var countValue = count
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(residualAddPSO.maxTotalThreadsPerThreadgroup), 256),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeDeltaParametersBatch(commandBuffer: MTLCommandBuffer,
                                    a: MTLBuffer,
                                    betaInput: MTLBuffer,
                                    aLog: MTLBuffer,
                                    aLogOffset: Int = 0,
                                    dtBias: MTLBuffer,
                                    dtBiasOffset: Int = 0,
                                    decay: MTLBuffer,
                                    beta: MTLBuffer,
                                    tokenCount: UInt32,
                                    headCount: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(prefillDeltaParametersPSO)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(betaInput, offset: 0, index: 1)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 2)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 3)
        encoder.setBuffer(decay, offset: 0, index: 4)
        encoder.setBuffer(beta, offset: 0, index: 5)
        var tokens = tokenCount
        var heads = headCount
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: Int(headCount), height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(headCount), 32), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeGatedNormBatch(commandBuffer: MTLCommandBuffer,
                              input: MTLBuffer,
                              gate: MTLBuffer,
                              weight: MTLBuffer,
                              weightOffset: Int = 0,
                              output: MTLBuffer,
                              tokenCount: UInt32,
                              headCount: UInt32,
                              headDimension: UInt32,
                              epsilon: Float) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(prefillGatedNormPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var tokens = tokenCount
        var heads = headCount
        var dimension = headDimension
        var epsilonValue = epsilon
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.stride, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: Int(headCount), height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(headCount), 32), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeResidualAddBatch(commandBuffer: MTLCommandBuffer,
                                lhs: MTLBuffer,
                                rhs: MTLBuffer,
                                output: MTLBuffer,
                                tokenCount: UInt32,
                                dimension: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(prefillResidualAddPSO)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var tokens = tokenCount
        var width = dimension
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: Int(dimension), height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(dimension), 256), height: 1, depth: 1))
        encoder.endEncoding()
    }
}