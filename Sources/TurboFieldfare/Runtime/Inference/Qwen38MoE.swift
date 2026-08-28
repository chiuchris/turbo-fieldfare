import Foundation
import Metal
import TurboFieldfareFormat

struct Qwen38MoEWeights {
    let router: TensorView
    let routerScale: TensorView
    let perExpertScale: TensorView
    let sharedExpertGate: TensorView
    let sharedExpertUp: TensorView
    let sharedExpertDown: TensorView
    let sharedExpertGateWeight: TensorView

    init(model: Model, layer: Int) throws {
        guard model.config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(model.config.modelFamily)")
        }
        self.router = try model.qwen38MoE(layer: layer, tensor: .router)
        self.routerScale = try model.routerScale(layer: layer)
        self.perExpertScale = try model.routerPerExpertScale(layer: layer)
        self.sharedExpertGate = try model.qwen38MoE(
            layer: layer, tensor: .sharedExpertGate)
        self.sharedExpertUp = try model.qwen38MoE(
            layer: layer, tensor: .sharedExpertUp)
        self.sharedExpertDown = try model.qwen38MoE(
            layer: layer, tensor: .sharedExpertDown)
        self.sharedExpertGateWeight = try model.qwen38MoE(
            layer: layer, tensor: .sharedExpertMultiplier)
        try Self.validateSharedExpertGate(sharedExpertGateWeight,
                                           hiddenSize: model.config.hiddenSize)
    }

    private static func validateSharedExpertGate(_ view: TensorView,
                                                 hiddenSize: Int) throws {
        guard view.dtype == GTurboFormatV1.DType.bf16.rawValue,
              view.shape.0 == 1,
              view.shape.1 == UInt32(hiddenSize),
              view.shape.2 == 0, view.shape.3 == 0,
              view.length == UInt64(hiddenSize * MemoryLayout<UInt16>.stride),
              view.scaleLength == 0, view.biasLength == 0,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 shared expert gate metadata mismatch")
        }
    }
}

final class Qwen38MoE {
    static let topK = QwenMoE.qwen38TopK
    static let numExperts = QwenMoE.qwen38NumExperts

    private let routerPipeline: MTLComputePipelineState
    private let selectPipeline: MTLComputePipelineState
    private let phase1Pipeline: MTLComputePipelineState
    private let phase2Pipeline: MTLComputePipelineState
    private let sharedGatePipeline: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let routeIndices: MTLBuffer
    private let routeWeights: MTLBuffer
    private let acts: MTLBuffer
    private let sharedGateValue: MTLBuffer
    private let routedArgumentEncoder: MTLArgumentEncoder
    private let routedArgumentBuffer: MTLBuffer

    init(context: MetalContext) throws {
        self.routerPipeline = try context.pipeline("qwen38_router_gemv")
        self.selectPipeline = try context.pipeline("qwen38_router_topk_select_k10")
        self.phase1Pipeline = try context.pipeline("qwen38_moe_phase1_gate_up_silu")
        self.phase2Pipeline = try context.pipeline("qwen38_moe_phase2_down_reduce_k10")
        self.sharedGatePipeline = try context.pipeline("qwen38_shared_expert_gate_sigmoid")
        guard let routerLogits = context.device.makeBuffer(
            length: Self.numExperts * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let routeIndices = context.device.makeBuffer(
                  length: Self.topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let routeWeights = context.device.makeBuffer(
                  length: Self.topK * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              let acts = context.device.makeBuffer(
                  length: Self.topK * 640 * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              let sharedGateValue = context.device.makeBuffer(
                  length: MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                  name: "qwen38_moe_phase1_gate_up_silu") else {
            throw MetalError.noDevice
        }
        self.routerLogits = routerLogits
        self.routeIndices = routeIndices
        self.routeWeights = routeWeights
        self.acts = acts
        self.sharedGateValue = sharedGateValue
        self.routedArgumentEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let routedArgumentBuffer = context.device.makeBuffer(
            length: routedArgumentEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.routedArgumentBuffer = routedArgumentBuffer
    }

    func encodeRouter(commandBuffer: MTLCommandBuffer,
                      weights: Qwen38MoEWeights,
                      hidden: MTLBuffer,
                      hiddenSize: UInt32) {
        precondition(hiddenSize.isMultiple(of: UInt32(Quantization.groupSize)))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerPipeline)
        encoder.setBuffer(weights.router.buffer,
                          offset: Int(weights.router.offset), index: 0)
        encoder.setBuffer(weights.router.buffer,
                          offset: Int(weights.router.scaleOffset), index: 1)
        encoder.setBuffer(weights.router.buffer,
                          offset: Int(weights.router.biasOffset), index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(weights.routerScale.buffer,
                          offset: Int(weights.routerScale.offset), index: 4)
        encoder.setBuffer(routerLogits, offset: 0, index: 5)
        var expertCount = UInt32(Self.numExperts)
        var dimension = hiddenSize
        encoder.setBytes(&expertCount,
                         length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&dimension,
                         length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Self.numExperts + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeSelection(commandBuffer: MTLCommandBuffer,
                         weights: Qwen38MoEWeights) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(selectPipeline)
        encoder.setBuffer(routerLogits, offset: 0, index: 0)
        encoder.setBuffer(routeIndices, offset: 0, index: 1)
        encoder.setBuffer(routeWeights, offset: 0, index: 2)
        encoder.setBuffer(weights.perExpertScale.buffer,
                          offset: Int(weights.perExpertScale.offset), index: 3)
        var expertCount = UInt32(Self.numExperts)
        encoder.setBytes(&expertCount,
                         length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func selectedExperts() -> [Int] {
        let pointer = routeIndices.contents().assumingMemoryBound(to: UInt32.self)
        return (0..<Self.topK).map { Int(pointer[$0]) }
    }

    func makeRoutedArgumentBuffer(experts: [TensorView]) throws -> MTLBuffer {
        guard experts.count == Self.topK else {
            throw ModelError.archMismatch(
                field: "qwen38RoutedExperts.count",
                expected: "\(Self.topK)",
                actual: "\(experts.count)")
        }
        routedArgumentEncoder.setArgumentBuffer(routedArgumentBuffer, offset: 0)
        for (index, expert) in experts.enumerated() {
            routedArgumentEncoder.setBuffer(
                expert.buffer, offset: Int(expert.offset), index: index)
        }
        return routedArgumentBuffer
    }

    func encodeRouted(commandBuffer: MTLCommandBuffer,
                      routedArgumentBuffer: MTLBuffer,
                      routedOffsets: MoEExpertOffsets,
                      input: MTLBuffer,
                      residual: MTLBuffer,
                      output: MTLBuffer,
                      hiddenSize: UInt32,
                      intermediateSize: UInt32,
                      sharedExpertGateWeight: TensorView) {
        precondition(sharedExpertGateWeight.dtype == GTurboFormatV1.DType.bf16.rawValue)
        guard let gateEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        gateEncoder.setComputePipelineState(sharedGatePipeline)
        gateEncoder.setBuffer(input, offset: 0, index: 0)
        gateEncoder.setBuffer(sharedExpertGateWeight.buffer,
                              offset: Int(sharedExpertGateWeight.offset), index: 1)
        gateEncoder.setBuffer(sharedGateValue, offset: 0, index: 2)
        var gateDimension = hiddenSize
        gateEncoder.setBytes(&gateDimension,
                             length: MemoryLayout<UInt32>.stride, index: 3)
        let gateThreads = min(sharedGatePipeline.maxTotalThreadsPerThreadgroup, 256)
        gateEncoder.dispatchThreads(
            MTLSize(width: gateThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: gateThreads, height: 1, depth: 1))
        gateEncoder.endEncoding()
        precondition(intermediateSize > 0)
        guard let phase1 = commandBuffer.makeComputeCommandEncoder() else { return }
        phase1.setComputePipelineState(phase1Pipeline)
        phase1.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        var offsets = routedOffsets
        phase1.setBytes(&offsets,
                length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        phase1.setBuffer(input, offset: 0, index: 2)
        phase1.setBuffer(acts, offset: 0, index: 3)
        var dimension = hiddenSize
        var intermediate = intermediateSize
        phase1.setBytes(&dimension,
                        length: MemoryLayout<UInt32>.stride, index: 4)
        phase1.setBytes(&intermediate,
                        length: MemoryLayout<UInt32>.stride, index: 5)
        phase1.dispatchThreadgroups(
            MTLSize(width: (Self.topK * Int(intermediateSize) + 7) / 8,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        phase1.endEncoding()

        guard let phase2 = commandBuffer.makeComputeCommandEncoder() else { return }
        phase2.setComputePipelineState(phase2Pipeline)
        phase2.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        phase2.setBytes(&offsets,
                length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        phase2.setBuffer(acts, offset: 0, index: 2)
        phase2.setBuffer(routeWeights, offset: 0, index: 3)
        phase2.setBuffer(residual, offset: 0, index: 4)
        phase2.setBuffer(output, offset: 0, index: 5)
        phase2.setBytes(&dimension,
                        length: MemoryLayout<UInt32>.stride, index: 6)
        phase2.setBytes(&intermediate,
                        length: MemoryLayout<UInt32>.stride, index: 7)
        phase2.setBuffer(sharedGateValue, offset: 0, index: 8)
        phase2.dispatchThreadgroups(
            MTLSize(width: Int(hiddenSize), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.topK * 32,
                                            height: 1, depth: 1))
        phase2.endEncoding()
    }

    func fetchSelectedExperts(model: Model, layer: Int) async throws
        -> (plan: RoutedExpertFetchPlan,
            views: [TensorView], offsets: MoEExpertOffsets) {
        let experts = selectedExperts()
        guard Set(experts).count == Self.topK,
              experts.allSatisfy({ $0 >= 0 && $0 < Self.numExperts }) else {
            throw ModelError.archMismatch(
                field: "qwen38RouteIndices",
                expected: "ten unique experts in 0..<512",
                actual: "\(experts)")
        }
        guard let plan = try model.planRoutedExperts(layer: layer, experts: experts) else {
            throw ModelError.residentBufferWrapFailed
        }
        let views = try await model.fetchRoutedExperts(plan: plan)
        return (plan, views, model.routedExpertOffsets(layer: layer))
    }
}
