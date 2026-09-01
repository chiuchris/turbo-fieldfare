import Metal
import TurboFieldfareFormat

struct Qwen38HyperConnectionGeometry: Sendable, Equatable {
    let streamCount: UInt32
    let hiddenSize: UInt32
    let lowRankSize: UInt32

    static let qwen = Qwen38HyperConnectionGeometry(
        streamCount: 4,
        hiddenSize: 2_560,
        lowRankSize: 320)

    var hyperWidth: UInt32 {
        streamCount * hiddenSize
    }
}

struct Qwen38HyperConnectionWeights {
    let norm: MTLBuffer
    let normOffset: Int
    let inputMixDown: Qwen38PLEQuantizedProjection
    let inputMixUp: Qwen38PLEQuantizedProjection
    let blockInject: Qwen38PLEQuantizedProjection?

    init(norm: MTLBuffer,
         normOffset: Int = 0,
         inputMixDown: Qwen38PLEQuantizedProjection,
         inputMixUp: Qwen38PLEQuantizedProjection,
         blockInject: Qwen38PLEQuantizedProjection?) {
        self.norm = norm
        self.normOffset = normOffset
        self.inputMixDown = inputMixDown
        self.inputMixUp = inputMixUp
        self.blockInject = blockInject
    }

    init(norm: TensorView,
         inputMixDown: TensorView,
         inputMixUp: TensorView,
         blockInject: TensorView?,
         geometry: Qwen38HyperConnectionGeometry = .qwen) throws {
        try Self.validateNorm(norm, geometry: geometry)
        self.norm = norm.buffer
        self.normOffset = Int(norm.offset)
        self.inputMixDown = try Self.projection(
            inputMixDown,
            rows: geometry.lowRankSize,
            columns: geometry.hyperWidth)
        self.inputMixUp = try Self.projection(
            inputMixUp,
            rows: geometry.hyperWidth,
            columns: geometry.lowRankSize)
        self.blockInject = try blockInject.map {
            try Self.projection(
                $0,
                rows: geometry.streamCount,
                columns: geometry.hyperWidth)
        }
    }

    private static func validateNorm(
        _ view: TensorView,
        geometry: Qwen38HyperConnectionGeometry
    ) throws {
        let expectedBytes = UInt64(geometry.hyperWidth)
            * UInt64(MemoryLayout<UInt16>.stride)
        guard view.dtype == GTurboFormatV1.DType.bf16.rawValue,
              view.shape.0 == geometry.hyperWidth,
              view.shape.1 == 0, view.shape.2 == 0, view.shape.3 == 0,
              view.length == expectedBytes,
              view.scaleLength == 0, view.biasLength == 0,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 Hyper-Connection norm metadata mismatch")
        }
    }

    private static func projection(
        _ view: TensorView,
        rows: UInt32,
        columns: UInt32
    ) throws -> Qwen38PLEQuantizedProjection {
        let groupSize: UInt32 = 32
        let weightBytes = UInt64(rows) * UInt64(columns) / 2
        let auxiliaryBytes = UInt64(rows) * UInt64(columns / groupSize)
            * UInt64(MemoryLayout<UInt16>.stride)
        guard columns.isMultiple(of: groupSize),
              view.dtype == GTurboFormatV1.DType.u32.rawValue,
              view.shape.0 == rows, view.shape.1 == columns,
              view.shape.2 == 0, view.shape.3 == 0,
              view.length == weightBytes,
              view.scaleLength == auxiliaryBytes,
              view.biasLength == auxiliaryBytes,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 Hyper-Connection affine-Q4 metadata mismatch")
        }
        return Qwen38PLEQuantizedProjection(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset))
    }
}

struct Qwen38HyperConnectionScratch {
    let normalized: MTLBuffer
    let lowRank: MTLBuffer
    let activatedLowRank: MTLBuffer
    let mixLogits: MTLBuffer
    let injectionLogits: MTLBuffer
    let injectionWeights: MTLBuffer
}

final class Qwen38HyperConnection {
    private let projection: Qwen38PLEProjection
    private let residual: Qwen38GatedResidual
    let geometry: Qwen38HyperConnectionGeometry

    init(context: MetalContext,
         geometry: Qwen38HyperConnectionGeometry = .qwen) throws {
        precondition(geometry.streamCount > 0 && geometry.hiddenSize > 0)
        precondition(geometry.hyperWidth.isMultiple(of: 32))
        precondition(geometry.lowRankSize > 0 && geometry.lowRankSize.isMultiple(of: 32))
        self.geometry = geometry
        self.projection = try Qwen38PLEProjection(context: context)
        self.residual = try Qwen38GatedResidual(context: context)
    }

    func encodePrepare(commandBuffer: MTLCommandBuffer,
                       hyperInput: MTLBuffer,
                       weights: Qwen38HyperConnectionWeights,
                       scratch: Qwen38HyperConnectionScratch,
                       mixedInput: MTLBuffer,
                       tokenCount: UInt32,
                       epsilon: Float) {
        precondition(tokenCount > 0)
        let hyperElements = Int(tokenCount * geometry.hyperWidth)
        let hyperBytes = hyperElements * MemoryLayout<Float16>.stride
        let lowRankBytes = Int(tokenCount * geometry.lowRankSize)
            * MemoryLayout<Float16>.stride
        let streamBytes = Int(tokenCount * geometry.streamCount)
            * MemoryLayout<Float16>.stride
        let mixedBytes = Int(tokenCount * geometry.hiddenSize)
            * MemoryLayout<Float16>.stride
        precondition(hyperInput.length >= hyperBytes)
        precondition(scratch.normalized.length >= hyperBytes)
        precondition(scratch.lowRank.length >= lowRankBytes)
        precondition(scratch.activatedLowRank.length >= lowRankBytes)
        precondition(scratch.mixLogits.length >= hyperBytes)
        precondition(mixedInput.length >= mixedBytes)

        residual.encodeGroupedNorm(
            commandBuffer: commandBuffer,
            input: hyperInput,
            weight: weights.norm,
            weightOffset: weights.normOffset,
            output: scratch.normalized,
            tokenCount: tokenCount,
            streamCount: geometry.streamCount,
            hiddenSize: geometry.hiddenSize,
            epsilon: epsilon)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.inputMixDown,
            input: scratch.normalized,
            output: scratch.lowRank,
            tokenCount: tokenCount,
            outputWidth: geometry.lowRankSize,
            inputWidth: geometry.hyperWidth)
        residual.encodeLowRankSiLU(
            commandBuffer: commandBuffer,
            input: scratch.lowRank,
            output: scratch.activatedLowRank,
            count: tokenCount * geometry.lowRankSize,
            streamCount: geometry.streamCount)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.inputMixUp,
            input: scratch.activatedLowRank,
            output: scratch.mixLogits,
            tokenCount: tokenCount,
            outputWidth: geometry.hyperWidth,
            inputWidth: geometry.lowRankSize)
        residual.encodeMixStreams(
            commandBuffer: commandBuffer,
            normalized: scratch.normalized,
            mixLogits: scratch.mixLogits,
            output: mixedInput,
            tokenCount: tokenCount,
            streamCount: geometry.streamCount,
            hiddenSize: geometry.hiddenSize)

        guard let blockInject = weights.blockInject else { return }
        precondition(scratch.injectionLogits.length >= streamBytes)
        precondition(scratch.injectionWeights.length >= streamBytes)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: blockInject,
            input: scratch.normalized,
            output: scratch.injectionLogits,
            tokenCount: tokenCount,
            outputWidth: geometry.streamCount,
            inputWidth: geometry.hyperWidth)
        residual.encodeInjectionWeights(
            commandBuffer: commandBuffer,
            logits: scratch.injectionLogits,
            output: scratch.injectionWeights,
            tokenCount: tokenCount,
            streamCount: geometry.streamCount)
    }

    func encodeInject(commandBuffer: MTLCommandBuffer,
                      hyperInput: MTLBuffer,
                      branchOutput: MTLBuffer,
                      injectionWeights: MTLBuffer,
                      output: MTLBuffer,
                      tokenCount: UInt32) {
        precondition(tokenCount > 0)
        residual.encodeInjectStreams(
            commandBuffer: commandBuffer,
            hyperInput: hyperInput,
            blockOutput: branchOutput,
            injectionWeights: injectionWeights,
            output: output,
            tokenCount: tokenCount,
            streamCount: geometry.streamCount,
            hiddenSize: geometry.hiddenSize)
    }

    private func encodeProjection(commandBuffer: MTLCommandBuffer,
                                  weights: Qwen38PLEQuantizedProjection,
                                  input: MTLBuffer,
                                  output: MTLBuffer,
                                  tokenCount: UInt32,
                                  outputWidth: UInt32,
                                  inputWidth: UInt32) {
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.weights,
            weightsOffset: weights.weightsOffset,
            scales: weights.scales,
            scalesOffset: weights.scalesOffset,
            biases: weights.biases,
            biasesOffset: weights.biasesOffset,
            input: input,
            output: output,
            tokenCount: tokenCount,
            outputWidth: outputWidth,
            inputWidth: inputWidth)
    }
}

final class Qwen38GatedResidual {
    private let groupedNormPSO: MTLComputePipelineState
    private let rmsNormPSO: MTLComputePipelineState
    private let lowRankSiLUPSO: MTLComputePipelineState
    private let mixStreamsPSO: MTLComputePipelineState
    private let injectionWeightsPSO: MTLComputePipelineState
    private let injectStreamsPSO: MTLComputePipelineState
    private let repeatStreamsPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.groupedNormPSO = try context.pipeline("qwen38_grouped_rmsnorm")
        self.rmsNormPSO = try context.pipeline("qwen38_rmsnorm")
        self.lowRankSiLUPSO = try context.pipeline("qwen38_low_rank_silu")
        self.mixStreamsPSO = try context.pipeline("qwen38_mix_streams")
        self.injectionWeightsPSO = try context.pipeline("qwen38_injection_weights")
        self.injectStreamsPSO = try context.pipeline("qwen38_inject_streams")
        self.repeatStreamsPSO = try context.pipeline("qwen38_repeat_streams")
    }

    func encodeGroupedNorm(commandBuffer: MTLCommandBuffer,
                           input: MTLBuffer,
                           weight: MTLBuffer,
                           weightOffset: Int = 0,
                           output: MTLBuffer,
                           tokenCount: UInt32,
                           streamCount: UInt32,
                           hiddenSize: UInt32,
                           epsilon: Float) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(groupedNormPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var tokens = tokenCount
        var streams = streamCount
        var hidden = hiddenSize
        var epsilonValue = epsilon
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&streams, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&hidden, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(streamCount), height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(streamCount), groupedNormPSO.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeRMSNorm(commandBuffer: MTLCommandBuffer,
                       input: MTLBuffer,
                       weight: MTLBuffer,
                       weightOffset: Int = 0,
                       output: MTLBuffer,
                       tokenCount: UInt32,
                       width: UInt32,
                       epsilon: Float) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rmsNormPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var tokens = tokenCount
        var widthValue = width
        var epsilonValue = epsilon
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&widthValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.stride, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: Int(tokenCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(tokenCount), rmsNormPSO.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeLowRankSiLU(commandBuffer: MTLCommandBuffer,
                           input: MTLBuffer,
                           output: MTLBuffer,
                           count: UInt32,
                           streamCount: UInt32) {
        encodeVectorActivation(
            commandBuffer: commandBuffer,
            pipeline: lowRankSiLUPSO,
            input: input,
            output: output,
            count: count,
            divisor: Float(streamCount))
    }

    func encodeMixStreams(commandBuffer: MTLCommandBuffer,
                          normalized: MTLBuffer,
                          mixLogits: MTLBuffer,
                          output: MTLBuffer,
                          tokenCount: UInt32,
                          streamCount: UInt32,
                          hiddenSize: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mixStreamsPSO)
        encoder.setBuffer(normalized, offset: 0, index: 0)
        encoder.setBuffer(mixLogits, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var tokens = tokenCount
        var streams = streamCount
        var hidden = hiddenSize
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&streams, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&hidden, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: Int(hiddenSize), height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(hiddenSize), mixStreamsPSO.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeInjectionWeights(commandBuffer: MTLCommandBuffer,
                                logits: MTLBuffer,
                                output: MTLBuffer,
                                tokenCount: UInt32,
                                streamCount: UInt32) {
        encodeVectorActivation(
            commandBuffer: commandBuffer,
            pipeline: injectionWeightsPSO,
            input: logits,
            output: output,
            count: tokenCount * streamCount,
            divisor: Float(streamCount))
    }

    func encodeInjectStreams(commandBuffer: MTLCommandBuffer,
                             hyperInput: MTLBuffer,
                             blockOutput: MTLBuffer,
                             injectionWeights: MTLBuffer,
                             output: MTLBuffer,
                             tokenCount: UInt32,
                             streamCount: UInt32,
                             hiddenSize: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(injectStreamsPSO)
        encoder.setBuffer(hyperInput, offset: 0, index: 0)
        encoder.setBuffer(blockOutput, offset: 0, index: 1)
        encoder.setBuffer(injectionWeights, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encodeStreamGeometry(
            encoder: encoder,
            pipeline: injectStreamsPSO,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize,
            argumentStart: 4)
    }

    func encodeRepeatStreams(commandBuffer: MTLCommandBuffer,
                             input: MTLBuffer,
                             output: MTLBuffer,
                             tokenCount: UInt32,
                             streamCount: UInt32,
                             hiddenSize: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(repeatStreamsPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encodeStreamGeometry(
            encoder: encoder,
            pipeline: repeatStreamsPSO,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize,
            argumentStart: 2)
    }

    private func encodeVectorActivation(commandBuffer: MTLCommandBuffer,
                                        pipeline: MTLComputePipelineState,
                                        input: MTLBuffer,
                                        output: MTLBuffer,
                                        count: UInt32,
                                        divisor: Float) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var elementCount = count
        var divisorValue = divisor
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&divisorValue, length: MemoryLayout<Float>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(count), pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    private func encodeStreamGeometry(encoder: MTLComputeCommandEncoder,
                                      pipeline: MTLComputePipelineState,
                                      tokenCount: UInt32,
                                      streamCount: UInt32,
                                      hiddenSize: UInt32,
                                      argumentStart: Int) {
        var tokens = tokenCount
        var streams = streamCount
        var hidden = hiddenSize
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: argumentStart)
        encoder.setBytes(&streams, length: MemoryLayout<UInt32>.stride, index: argumentStart + 1)
        encoder.setBytes(&hidden, length: MemoryLayout<UInt32>.stride, index: argumentStart + 2)
        encoder.dispatchThreads(
            MTLSize(width: Int(hiddenSize), height: Int(streamCount), depth: Int(tokenCount)),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(hiddenSize), pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }
}
