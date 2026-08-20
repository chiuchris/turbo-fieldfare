import Metal

/// Qwen3.5 MoE routing primitives.
///
/// This path is deliberately separate from `MoE`: Qwen has no Gemma router
/// scaling tensor and its top-k weights are normalized directly from router
/// logits.
final class QwenMoE {
    static let topK = 8

    private let routerPSO: MTLComputePipelineState
    private let selectPSO: MTLComputePipelineState
    private let routedPhase1PSO: MTLComputePipelineState
    private let routedPhase2PSO: MTLComputePipelineState
    private let sharedExpert: QwenSharedExpertInt4
    private let int8: DequantInt8GEMV
    private let combineSharedPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let sharedGateLogit: MTLBuffer
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    init(context: MetalContext) throws {
        self.routerPSO = try context.pipeline("qwen_router_gemv")
        self.selectPSO = try context.pipeline("qwen_router_topk_select_k8")
        self.routedPhase1PSO = try context.pipeline("qwen_moe_phase1_gate_up_silu")
        self.routedPhase2PSO = try context.pipeline("qwen_moe_phase2_down_reduce_k8")
        self.sharedExpert = try QwenSharedExpertInt4(context: context)
        self.int8 = try DequantInt8GEMV(context: context)
        self.combineSharedPSO = try context.pipeline("qwen_combine_shared_silu")
        guard let logits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        guard let sharedGateLogit = context.device.makeBuffer(
            length: MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.sharedGateLogit = sharedGateLogit
        guard let phase1Function = context.library.makeFunction(
            name: "qwen_moe_phase1_gate_up_silu") else {
            throw MetalError.noDevice
        }
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouter(commandBuffer: MTLCommandBuffer,
                      weights: MTLBuffer, weightsOffset: Int = 0,
                      scales: MTLBuffer, scalesOffset: Int = 0,
                      biases: MTLBuffer, biasesOffset: Int = 0,
                      hidden: MTLBuffer,
                      outIndices: MTLBuffer,
                      outWeights: MTLBuffer,
                      numExperts: UInt32,
                      d: UInt32,
                      topK: UInt32 = UInt32(QwenMoE.topK)) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(numExperts >= UInt32(Self.topK) && numExperts <= 256)
        precondition(topK == UInt32(Self.topK))

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(routerPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(routerLogits, offset: 0, index: 4)
            var expertCount = numExperts
            var dimension = d
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(selectPSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(outIndices, offset: 0, index: 1)
            encoder.setBuffer(outWeights, offset: 0, index: 2)
            var expertCount = numExperts
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer]) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer]) -> MTLBuffer {
        validate(routedBlobs: routedBlobs)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPhase1(commandBuffer: MTLCommandBuffer,
                            routedArgBuffer: MTLBuffer,
                            routedBlobs: [MTLBuffer],
                            routedOffsets: MoEExpertOffsets,
                            x: MTLBuffer,
                            acts: MTLBuffer,
                            d: UInt32,
                            f: UInt32) {
        validate(routedBlobs: routedBlobs)
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        var offsets = routedOffsets
        var dimension = d
        var intermediate = f
        var topK = UInt32(Self.topK)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routedPhase1PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob, usage: .read) }
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&topK, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Self.topK * Int(f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPhase2(commandBuffer: MTLCommandBuffer,
                            routedArgBuffer: MTLBuffer,
                            routedBlobs: [MTLBuffer],
                            routedOffsets: MoEExpertOffsets,
                            acts: MTLBuffer,
                            routingWeights: MTLBuffer,
                            residual: MTLBuffer,
                            y: MTLBuffer,
                            d: UInt32,
                            f: UInt32) {
        validate(routedBlobs: routedBlobs)
        var offsets = routedOffsets
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routedPhase2PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for blob in routedBlobs { encoder.useResource(blob, usage: .read) }
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeSharedExpert(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer,
                            gate: SharedExpertProjection,
                            up: SharedExpertProjection,
                            down: SharedExpertProjection,
                            y: MTLBuffer,
                            scratchGate: MTLBuffer,
                            scratchUp: MTLBuffer,
                            scratchAct: MTLBuffer) throws {
        try sharedExpert.encode(commandBuffer: commandBuffer,
                                x: x,
                                gate: gate,
                                up: up,
                                down: down,
                                y: y,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct)
    }

    func encodeSharedGateAndCombine(commandBuffer: MTLCommandBuffer,
                                    x: MTLBuffer,
                                    gate: SharedExpertProjection,
                                    sharedOutput: MTLBuffer,
                                    routedOutput: MTLBuffer,
                                    y: MTLBuffer,
                                    d: UInt32) {
        precondition(gate.rows == 1 && gate.cols == d)
        int8.encode(commandBuffer: commandBuffer,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, scalesOffset: gate.scalesOffset,
                    biases: gate.biases, biasesOffset: gate.biasesOffset,
                    x: x,
                    y: sharedGateLogit,
                    m: 1,
                    n: d)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(combineSharedPSO)
        encoder.setBuffer(sharedGateLogit, offset: 0, index: 0)
        encoder.setBuffer(sharedOutput, offset: 0, index: 1)
        encoder.setBuffer(routedOutput, offset: 0, index: 2)
        encoder.setBuffer(y, offset: 0, index: 3)
        var count = d
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(combineSharedPSO.maxTotalThreadsPerThreadgroup, 256),
                                            height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [MTLBuffer]) {
        precondition(routedBlobs.count == Self.topK)
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob, offset: 0, index: index)
        }
    }
}