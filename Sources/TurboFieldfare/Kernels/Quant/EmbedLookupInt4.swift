import Foundation
import Metal

/// Affine embedding lookup with fused output scale.
///
/// The default remains the packed 4-bit path used by existing models. Qwen3.8
/// diagnostic artifacts may preserve an 8-bit affine embedding table, selected
/// from the manifest's tensor quantization descriptor.
final class EmbedLookupInt4 {
    private let pso: MTLComputePipelineState
    private let weightBits: Int

    init(context: MetalContext,
         groupSize: Int = Quantization.groupSize,
         weightBits: Int = 4) throws {
        precondition(groupSize == Quantization.groupSize ||
                     groupSize == Quantization.qwen38GroupSize)
        precondition(weightBits == 4 || weightBits == 8,
                     "embedding weight bits must be 4 or 8")
        self.weightBits = weightBits
        if weightBits == 8 {
            precondition(groupSize == Quantization.qwen38GroupSize,
                         "q8 embedding requires the Qwen3.8 group size")
            self.pso = try context.pipeline("embed_lookup_int8")
        } else {
            let constants = groupSize == Quantization.qwen38GroupSize
                ? [MetalFunctionConstant(index: 27, value: .uint32(UInt32(groupSize)))]
                : []
            self.pso = try context.pipeline("embed_lookup_int4", constants: constants)
        }
    }

    /// Encodes the lookup. `table`, `scales`, `biases` typically live inside
    /// one resident blob — pass that buffer with the per-region offsets.
    /// Pass `outScale = 1.0` to write the raw dequantized row.
    func encode(commandBuffer: MTLCommandBuffer,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(Quantization.qwen38GroupSize) == 0,
                 "D must be a multiple of \(Quantization.qwen38GroupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table,  offset: tableOffset,  index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var tokenVar = tokenId
        var dVar     = d
        var sVar     = outScale
        enc.setBytes(&tokenVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&sVar,     length: MemoryLayout<Float>.size,  index: 6)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: Int(d), height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
