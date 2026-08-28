import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8
    static let inputRowsPerThreadgroup = 8

    private static let rowSummaryStride = 2
    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeVocab: UInt32 = 262144
    private static let qwenDecodeD: UInt32 = 2048
    private static let qwenDecodeVocab: UInt32 = 248320

    private let rms: RMSNorm
    private let prefillRMS: PrefillRMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowGreedySpecialized: MTLComputePipelineState
    private let rowGreedyQwenSpecialized: MTLComputePipelineState
    private let rowReducer: MTLComputePipelineState
    private let batchedRowGreedy: MTLComputePipelineState
    private let batchedRowGreedySpecialized: MTLComputePipelineState
    private let batchedRowGreedyQwenSpecialized: MTLComputePipelineState
    private let batchedRowReducer: MTLComputePipelineState
    private let xNormedBuffer: MTLBuffer
    private let rowSummariesBuffer: MTLBuffer
    private let maxD: Int
    private let maxVocab: Int
    private let maxBatchRows: Int

    init(context: MetalContext,
         maxD: Int = 2816,
         maxVocab: Int = 262144,
         maxBatchRows: Int = 16) throws {
        self.rms = try RMSNorm(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.rowGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_raw")
        self.rowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: Self.headConstants(d: Self.realDecodeD,
                                          vocab: Self.realDecodeVocab))
        self.rowGreedyQwenSpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: Self.headConstants(d: Self.qwenDecodeD,
                                          vocab: Self.qwenDecodeVocab))
        self.rowReducer = try context.pipeline("lm_head_greedy_int4_rows_reduce")
        self.batchedRowGreedy = try context.pipeline(
            "lm_head_greedy_int4_batched_rows_chunk_raw")
        self.batchedRowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_batched_rows_chunk_raw",
            constants: Self.headConstants(d: Self.realDecodeD,
                                          vocab: Self.realDecodeVocab))
        self.batchedRowGreedyQwenSpecialized = try context.pipeline(
            "lm_head_greedy_int4_batched_rows_chunk_raw",
            constants: Self.headConstants(d: Self.qwenDecodeD,
                                          vocab: Self.qwenDecodeVocab))
        self.batchedRowReducer = try context.pipeline(
            "lm_head_greedy_int4_batched_rows_reduce")
        self.maxD = maxD
        self.maxVocab = maxVocab
        self.maxBatchRows = maxBatchRows

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = max(maxD * maxBatchRows, 1) * MemoryLayout<Float16>.size
        let summaryLength = rowGroups * maxBatchRows * Self.rowSummaryStride
            * MemoryLayout<Float>.size
        guard let xNormedBuffer = context.device.makeBuffer(
                  length: xLength,
                  options: .storageModePrivate),
              let rowSummariesBuffer = context.device.makeBuffer(
                  length: summaryLength,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.xNormedBuffer = xNormedBuffer
        self.rowSummariesBuffer = rowSummariesBuffer
    }

    func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int = 0,
                            normWeight: MTLBuffer,
                            normOffset: Int = 0,
                            weights: MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: MTLBuffer,
                            biasesOffset: Int = 0,
                            outToken: MTLBuffer,
                            outTokenOffset: Int = 0,
                            d: UInt32,
                            vocab: UInt32,
                            rmsEps: Float = 1e-6) {
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % Quantization.groupSize == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(outTokenOffset >= 0, "outTokenOffset must be non-negative")
        precondition(outTokenOffset % MemoryLayout<UInt32>.alignment == 0,
                 "outTokenOffset must be UInt32-aligned")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_raw needs a 2-aligned weightsOffset")

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hidden,
                        xOffset: hiddenOffset,
                        weight: normWeight,
                        weightOffset: normOffset,
                        out: xNormedBuffer,
                        d: d,
                        eps: rmsEps)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let pipeline: MTLComputePipelineState
            if d == Self.realDecodeD && vocab == Self.realDecodeVocab {
                pipeline = rowGreedySpecialized
            } else if d == Self.qwenDecodeD && vocab == Self.qwenDecodeVocab {
                pipeline = rowGreedyQwenSpecialized
            } else {
                pipeline = rowGreedy
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(xNormedBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)

            let threadgroupSize = MTLSize(
                width: 32 * Self.rowsPerThreadgroup,
                height: 1,
                depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outToken, offset: outTokenOffset, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)

            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(threadgroupSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    func encodeGreedyRows(commandBuffer: MTLCommandBuffer,
                          hidden: MTLBuffer,
                          hiddenOffset: Int = 0,
                          rowCount: Int,
                          rowStrideElements: Int,
                          normWeight: MTLBuffer,
                          normOffset: Int = 0,
                          weights: MTLBuffer,
                          weightsOffset: Int = 0,
                          scales: MTLBuffer,
                          scalesOffset: Int = 0,
                          biases: MTLBuffer,
                          biasesOffset: Int = 0,
                          outTokens: MTLBuffer,
                          outTokensOffset: Int = 0,
                          d: UInt32,
                          vocab: UInt32,
                          rmsEps: Float = 1e-6) {
        precondition(rowCount > 0, "rowCount must be positive")
        precondition(rowStrideElements >= Int(d), "row stride must cover d")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(outTokensOffset >= 0, "outTokensOffset must be non-negative")
        precondition(outTokensOffset % MemoryLayout<UInt32>.alignment == 0,
                     "outTokensOffset must be UInt32-aligned")
        precondition(weightsOffset % 2 == 0,
                     "batched greedy head needs a 2-aligned weightsOffset")

        guard rowCount > 1,
              rowCount <= maxBatchRows,
              rowStrideElements == Int(d) else {
            for row in 0..<rowCount {
                encodeGreedyDecode(
                    commandBuffer: commandBuffer,
                    hidden: hidden,
                    hiddenOffset: hiddenOffset
                        + row * rowStrideElements * MemoryLayout<Float16>.stride,
                    normWeight: normWeight,
                    normOffset: normOffset,
                    weights: weights,
                    weightsOffset: weightsOffset,
                    scales: scales,
                    scalesOffset: scalesOffset,
                    biases: biases,
                    biasesOffset: biasesOffset,
                    outToken: outTokens,
                    outTokenOffset: outTokensOffset
                        + row * MemoryLayout<UInt32>.stride,
                    d: d,
                    vocab: vocab,
                    rmsEps: rmsEps)
            }
            return
        }

        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % Quantization.groupSize == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        prefillRMS.encodeBF16W(
            commandBuffer: commandBuffer,
            x: hidden,
            xOffset: hiddenOffset,
            weight: normWeight,
            weightOffset: normOffset,
            out: xNormedBuffer,
            t: UInt32(rowCount),
            d: d,
            eps: rmsEps)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let pipeline: MTLComputePipelineState
            if d == Self.realDecodeD && vocab == Self.realDecodeVocab {
                pipeline = batchedRowGreedySpecialized
            } else if d == Self.qwenDecodeD && vocab == Self.qwenDecodeVocab {
                pipeline = batchedRowGreedyQwenSpecialized
            } else {
                pipeline = batchedRowGreedy
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(xNormedBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            var rowCountValue = UInt32(rowCount)
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&rowCountValue, length: MemoryLayout<UInt32>.size, index: 7)

            let threadgroupSize = MTLSize(
                width: 32 * Self.rowsPerThreadgroup,
                height: 1,
                depth: 1)
            let inputRowGroups = (rowCount + Self.inputRowsPerThreadgroup - 1)
                / Self.inputRowsPerThreadgroup
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: inputRowGroups, depth: 1),
                threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(batchedRowReducer)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outTokens, offset: outTokensOffset, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            var rowCountValue = UInt32(rowCount)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&rowCountValue, length: MemoryLayout<UInt32>.size, index: 3)
            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    private static func headConstants(d: UInt32,
                                      vocab: UInt32) -> [MetalFunctionConstant] {
        [
            MetalFunctionConstant(index: 10, value: .uint32(d)),
            MetalFunctionConstant(index: 11, value: .uint32(vocab)),
            MetalFunctionConstant(index: 13, value: .bool(true)),
        ]
    }
}
