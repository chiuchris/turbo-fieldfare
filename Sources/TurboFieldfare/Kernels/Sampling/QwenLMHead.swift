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
    private let gemv: DequantInt4GEMV

    init(context: MetalContext,
         geometry: QwenLMHeadGeometry = .qwen) throws {
        precondition(geometry.vocabularySize > 0,
                     "vocabulary size must be positive")
        precondition(geometry.hiddenSize > 0,
                     "hidden size must be positive")
        self.geometry = geometry
        self.gemv = try DequantInt4GEMV(context: context)
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
        gemv.encode(
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