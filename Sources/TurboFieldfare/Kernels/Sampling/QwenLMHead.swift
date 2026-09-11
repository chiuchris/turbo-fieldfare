import Metal

struct QwenLMHeadGeometry: Sendable, Equatable {
    let vocabularySize: Int
    let hiddenSize: Int

    static let qwen = QwenLMHeadGeometry(
        vocabularySize: 248_320,
        hiddenSize: 2_048)
}

/// Independent Qwen output projection. It intentionally does not reuse the
/// embedding lookup or assume tied input/output weights.
final class QwenUntiedLMHead {
    let geometry: QwenLMHeadGeometry
    private let int4Gemv: DequantInt4GEMV?
    private let int8Gemv: DequantInt8GEMV?

    init(context: MetalContext,
            geometry: QwenLMHeadGeometry = .qwen,
            groupSize: Int = Quantization.groupSize,
            weightBits: Int = 4) throws {
        precondition(geometry.vocabularySize > 0,
                     "vocabulary size must be positive")
        precondition(geometry.hiddenSize > 0,
                     "hidden size must be positive")
        precondition(weightBits == 4 || weightBits == 8,
                     "lm-head weight bits must be 4 or 8")
        self.geometry = geometry
        if weightBits == 8 {
            precondition(groupSize == Quantization.qwen38GroupSize,
                         "q8 lm-head requires the Qwen3.8 group size")
            self.int4Gemv = nil
            self.int8Gemv = try DequantInt8GEMV(context: context)
        } else {
            self.int4Gemv = try DequantInt4GEMV(context: context, groupSize: groupSize)
            self.int8Gemv = nil
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                hidden: MTLBuffer,
                hiddenOffset: Int = 0,
                logits: MTLBuffer,
                logitsOffset: Int = 0) {
        if let int8Gemv {
            int8Gemv.encode(
                commandBuffer: commandBuffer,
                weights: weights,
                weightsOffset: weightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                x: hidden,
                xOffset: hiddenOffset,
                y: logits,
                yOffset: logitsOffset,
                m: UInt32(geometry.vocabularySize),
                n: UInt32(geometry.hiddenSize))
        } else {
            int4Gemv!.encode(
                commandBuffer: commandBuffer,
                weights: weights,
                weightsOffset: weightsOffset,
                scales: scales,
                scalesOffset: scalesOffset,
                biases: biases,
                biasesOffset: biasesOffset,
                x: hidden,
                xOffset: hiddenOffset,
                y: logits,
                yOffset: logitsOffset,
                m: UInt32(geometry.vocabularySize),
                n: UInt32(geometry.hiddenSize))
        }
    }
}