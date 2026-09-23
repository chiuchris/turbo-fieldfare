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
    private let int8: DequantInt8GEMV
    private let siluMulPSO: MTLComputePipelineState
    private let siluMulBlockPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(
            context: context,
            groupSize: Quantization.qwen38GroupSize)
        self.qmm = try PrefillInt4QMM(
            context: context,
            groupSize: Quantization.qwen38GroupSize)
        self.int8 = try DequantInt8GEMV(context: context)
        self.siluMulPSO = try context.pipeline("silu_mul_fp16")
        self.siluMulBlockPSO = try context.pipeline("silu_mul_fp16_block")
    }

    func encodeAffine(commandBuffer: MTLCommandBuffer,
                      x: MTLBuffer,
                      gate: SharedExpertProjection,
                      up: SharedExpertProjection,
                      down: SharedExpertProjection,
                      y: MTLBuffer,
                      scratchGate: MTLBuffer,
                      scratchUp: MTLBuffer,
                      scratchAct: MTLBuffer) throws {
        guard gate.rows == up.rows,
              gate.cols == up.cols,
              down.rows == gate.cols,
              down.cols == gate.rows else {
            throw QwenSharedExpertError.dimensionMismatch(
                "affine shared expert projection shapes are inconsistent")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let intermediateBytes = Int(gate.rows) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard x.length >= inputBytes,
              y.length >= outputBytes,
              scratchGate.length >= intermediateBytes,
              scratchUp.length >= intermediateBytes,
              scratchAct.length >= intermediateBytes else {
            throw QwenSharedExpertError.scratchTooSmall(
                "affine shared expert buffers are smaller than the projection shapes")
        }

        int8.encode(commandBuffer: commandBuffer,
                    weights: gate.weights,
                    weightsOffset: gate.weightsOffset,
                    scales: gate.scales,
                    scalesOffset: gate.scalesOffset,
                    biases: gate.biases,
                    biasesOffset: gate.biasesOffset,
                    x: x,
                    y: scratchGate,
                    m: gate.rows,
                    n: gate.cols)
        int8.encode(commandBuffer: commandBuffer,
                    weights: up.weights,
                    weightsOffset: up.weightsOffset,
                    scales: up.scales,
                    scalesOffset: up.scalesOffset,
                    biases: up.biases,
                    biasesOffset: up.biasesOffset,
                    x: x,
                    y: scratchUp,
                    m: up.rows,
                    n: up.cols)
        try encodeSiluMultiply(commandBuffer: commandBuffer,
                               gate: scratchGate,
                               up: scratchUp,
                               act: scratchAct,
                               tokenCount: 1,
                               featureCount: Int(gate.rows))
        int8.encode(commandBuffer: commandBuffer,
                    weights: down.weights,
                    weightsOffset: down.weightsOffset,
                    scales: down.scales,
                    scalesOffset: down.scalesOffset,
                    biases: down.biases,
                    biasesOffset: down.biasesOffset,
                    x: scratchAct,
                    y: y,
                    m: down.rows,
                    n: down.cols)
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

        try encodeSiluMultiply(commandBuffer: commandBuffer,
                               gate: scratchGate,
                               up: scratchUp,
                               act: scratchAct,
                               tokenCount: queryCount,
                               featureCount: intermediate)

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

    private func encodeSiluMultiply(commandBuffer: MTLCommandBuffer,
                                    gate: MTLBuffer,
                                    up: MTLBuffer,
                                    act: MTLBuffer,
                                    tokenCount: Int,
                                    featureCount: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(siluMulBlockPSO)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(act, offset: 0, index: 2)
        var tokenCountValue = UInt32(tokenCount)
        var featureCountValue = UInt32(featureCount)
        encoder.setBytes(&tokenCountValue,
                         length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&featureCountValue,
                         length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: featureCount, height: tokenCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()
    }
}