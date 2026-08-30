import Metal

enum QwenSharedExpertError: Error, CustomStringConvertible {
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    var description: String {
        switch self {
        case .dimensionMismatch(let detail):
            return "Qwen shared expert dimension mismatch: \(detail)"
        case .scratchTooSmall(let detail):
            return "Qwen shared expert scratch too small: \(detail)"
        }
    }
}

final class QwenSharedExpertInt4 {
    private let int4: DequantInt4GEMV
    private let qmm: PrefillInt4QMM
    private let siluMulPSO: MTLComputePipelineState
    private let siluMulBlockPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(
            context: context,
            groupSize: Quantization.qwen38GroupSize)
        self.qmm = try PrefillInt4QMM(
            context: context,
            groupSize: Quantization.qwen38GroupSize)
        self.siluMulPSO = try context.pipeline("silu_mul_fp16")
        self.siluMulBlockPSO = try context.pipeline("silu_mul_fp16_block")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                x: MTLBuffer,
                gate: SharedExpertProjection,
                up: SharedExpertProjection,
                down: SharedExpertProjection,
                y: MTLBuffer,
                scratchGate: MTLBuffer,
                scratchUp: MTLBuffer,
                scratchAct: MTLBuffer) throws {
        try encodeBlock(
            commandBuffer: commandBuffer,
            x: x,
            y: y,
            gate: gate,
            up: up,
            down: down,
            scratchGate: scratchGate,
            scratchUp: scratchUp,
            scratchAct: scratchAct,
            queryCount: 1,
            d: Int(gate.cols),
            intermediate: Int(gate.rows),
            xStrideElements: Int(gate.cols),
            yStrideElements: Int(down.rows))
    }

    func encodeBlock(commandBuffer: MTLCommandBuffer,
                     x: MTLBuffer,
                     y: MTLBuffer,
                     gate: SharedExpertProjection,
                     up: SharedExpertProjection,
                     down: SharedExpertProjection,
                     scratchGate: MTLBuffer,
                     scratchUp: MTLBuffer,
                     scratchAct: MTLBuffer,
                     queryCount: Int,
                     d: Int,
                     intermediate: Int,
                     xStrideElements: Int,
                     yStrideElements: Int) throws {
        guard queryCount > 0,
              d > 0,
              intermediate > 0,
              xStrideElements == d,
              yStrideElements == d,
              gate.rows == UInt32(intermediate),
              gate.cols == UInt32(d),
              up.rows == UInt32(intermediate),
              up.cols == UInt32(d),
              down.rows == UInt32(d),
              down.cols == UInt32(intermediate) else {
            throw QwenSharedExpertError.dimensionMismatch(
                "block shape queryCount=\(queryCount) d=\(d) intermediate=\(intermediate)")
        }
        let rowBytes = d * MemoryLayout<Float16>.stride
        let intermediateBytes = queryCount * intermediate * MemoryLayout<Float16>.stride
        guard x.length >= queryCount * rowBytes,
              y.length >= queryCount * rowBytes,
              scratchGate.length >= intermediateBytes,
              scratchUp.length >= intermediateBytes,
              scratchAct.length >= intermediateBytes else {
            throw QwenSharedExpertError.scratchTooSmall(
                "block queryCount=\(queryCount) d=\(d) intermediate=\(intermediate)")
        }

        qmm.encode(commandBuffer: commandBuffer,
                   weights: gate.weights,
                   weightsOffset: gate.weightsOffset,
                   scales: gate.scales,
                   scalesOffset: gate.scalesOffset,
                   biases: gate.biases,
                   biasesOffset: gate.biasesOffset,
                   x: x,
                   y: scratchGate,
                   t: queryCount,
                   n: intermediate,
                   k: d)
        qmm.encode(commandBuffer: commandBuffer,
                   weights: up.weights,
                   weightsOffset: up.weightsOffset,
                   scales: up.scales,
                   scalesOffset: up.scalesOffset,
                   biases: up.biases,
                   biasesOffset: up.biasesOffset,
                   x: x,
                   y: scratchUp,
                   t: queryCount,
                   n: intermediate,
                   k: d)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(siluMulBlockPSO)
        encoder.setBuffer(scratchGate, offset: 0, index: 0)
        encoder.setBuffer(scratchUp, offset: 0, index: 1)
        encoder.setBuffer(scratchAct, offset: 0, index: 2)
        var tokenCount = UInt32(queryCount)
        var featureCount = UInt32(intermediate)
        encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&featureCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: intermediate, height: queryCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()

        qmm.encode(commandBuffer: commandBuffer,
                   weights: down.weights,
                   weightsOffset: down.weightsOffset,
                   scales: down.scales,
                   scalesOffset: down.scalesOffset,
                   biases: down.biases,
                   biasesOffset: down.biasesOffset,
                   x: scratchAct,
                   y: y,
                   t: queryCount,
                   n: d,
                   k: intermediate)
    }
}