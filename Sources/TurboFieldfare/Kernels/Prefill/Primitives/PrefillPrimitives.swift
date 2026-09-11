import Foundation
import Metal

final class PrefillEmbedLookupInt4 {
    private let pso: MTLComputePipelineState
    private let groupSize: Int

    init(context: MetalContext,
         groupSize: Int = Quantization.groupSize,
         weightBits: Int = 4) throws {
        precondition(groupSize == Quantization.groupSize ||
                     groupSize == Quantization.qwen38GroupSize)
        precondition(weightBits == 4 || weightBits == 8,
                     "embedding weight bits must be 4 or 8")
        self.groupSize = groupSize
        if weightBits == 8 {
            precondition(groupSize == Quantization.qwen38GroupSize,
                         "q8 embedding requires the Qwen3.8 group size")
            self.pso = try context.pipeline("prefill_embed_lookup_int8_block")
        } else {
            let constants = groupSize == Quantization.qwen38GroupSize
                ? [MetalFunctionConstant(index: 78, value: .uint32(UInt32(groupSize)))]
                : []
            self.pso = try context.pipeline(
                "prefill_embed_lookup_int4_block",
                constants: constants)
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       table: MTLBuffer, tableOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       tokens: MTLBuffer, tokensOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       t: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(groupSize) == 0,
                     "D must be a multiple of \(groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(tokens, offset: tokensOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var tVar = t
        var dVar = d
        var scaleVar = outScale
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(t), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillRMSNorm {
    private let psoBF16W: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_block")
    }

    func encodeBF16W(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            t: UInt32,
                            d: UInt32,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var tVar = t
        var dVar = d
        var epsVar = eps
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 5)
        let threads = min(Int(psoBF16W.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

enum QwenPrefillProjectionPath: String, Sendable, Equatable {
    case repeatedGEMV
    case batchedQMM
}

/// Dispatches contiguous Qwen projection rows through the prefill matrix path.
/// The one-row fallback preserves the decode kernel's exact buffer contract.
final class QwenPrefillProjectionBatch {
    private let qmm: PrefillInt4QMM
    private let gemv: DequantInt4GEMV

    init(context: MetalContext) throws {
        self.qmm = try PrefillInt4QMM(context: context)
        self.gemv = try DequantInt4GEMV(context: context)
    }

    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                input: MTLBuffer,
                output: MTLBuffer,
                tokenCount: Int,
                outputWidth: Int,
                inputWidth: Int) -> QwenPrefillProjectionPath {
        precondition(tokenCount > 0, "tokenCount must be positive")
        precondition(outputWidth > 0, "outputWidth must be positive")
        precondition(inputWidth > 0, "inputWidth must be positive")
        precondition(inputWidth.isMultiple(of: Quantization.groupSize),
                     "inputWidth must be a multiple of the quantization group size")

        if tokenCount == 1 {
            gemv.encode(commandBuffer: commandBuffer,
                        weights: weights,
                        weightsOffset: weightsOffset,
                        scales: scales,
                        scalesOffset: scalesOffset,
                        biases: biases,
                        biasesOffset: biasesOffset,
                        x: input,
                        y: output,
                        m: UInt32(outputWidth),
                        n: UInt32(inputWidth))
            return .repeatedGEMV
        }

        qmm.encode(commandBuffer: commandBuffer,
                   weights: weights,
                   weightsOffset: weightsOffset,
                   scales: scales,
                   scalesOffset: scalesOffset,
                   biases: biases,
                   biasesOffset: biasesOffset,
                   x: input,
                   y: output,
                   t: tokenCount,
                   n: outputWidth,
                   k: inputWidth)
        return .batchedQMM
    }
}

final class PrefillInt4QMM {
    private let pso: MTLComputePipelineState
    private let groupSize: Int

    init(context: MetalContext,
         groupSize: Int = Quantization.groupSize) throws {
        precondition(groupSize == Quantization.groupSize ||
                     groupSize == Quantization.qwen38GroupSize)
        self.groupSize = groupSize
        let constants = groupSize == Quantization.qwen38GroupSize
            ? [MetalFunctionConstant(index: 78, value: .uint32(UInt32(groupSize)))]
            : []
        self.pso = try context.pipeline(
            "prefill_dequant_int4_qmm_f16_block",
            constants: constants)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       t: Int,
                       n: Int,
                       k: Int) {
        precondition(k % groupSize == 0,
                     "K must be a multiple of \(groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }
}
