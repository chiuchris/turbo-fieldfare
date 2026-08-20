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
    private let siluMulPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(context: context)
        self.siluMulPSO = try context.pipeline("silu_mul_fp16")
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
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw QwenSharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let intermediateBytes = Int(gate.rows) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard x.length >= inputBytes, scratchGate.length >= intermediateBytes,
              scratchUp.length >= intermediateBytes, scratchAct.length >= intermediateBytes,
              y.length >= outputBytes else {
            throw QwenSharedExpertError.scratchTooSmall(
                "input=\(inputBytes) intermediate=\(intermediateBytes) output=\(outputBytes)")
        }

        int4.encode(commandBuffer: commandBuffer,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, scalesOffset: gate.scalesOffset,
                    biases: gate.biases, biasesOffset: gate.biasesOffset,
                    x: x, y: scratchGate, m: gate.rows, n: gate.cols)
        int4.encode(commandBuffer: commandBuffer,
                    weights: up.weights, weightsOffset: up.weightsOffset,
                    scales: up.scales, scalesOffset: up.scalesOffset,
                    biases: up.biases, biasesOffset: up.biasesOffset,
                    x: x, y: scratchUp, m: up.rows, n: up.cols)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(siluMulPSO)
        encoder.setBuffer(scratchGate, offset: 0, index: 0)
        encoder.setBuffer(scratchUp, offset: 0, index: 1)
        encoder.setBuffer(scratchAct, offset: 0, index: 2)
        var count = UInt32(gate.rows)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(gate.rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(siluMulPSO.maxTotalThreadsPerThreadgroup, 256),
                                            height: 1, depth: 1))
        encoder.endEncoding()

        int4.encode(commandBuffer: commandBuffer,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales: down.scales, scalesOffset: down.scalesOffset,
                    biases: down.biases, biasesOffset: down.biasesOffset,
                    x: scratchAct, y: y, m: down.rows, n: down.cols)
    }
}