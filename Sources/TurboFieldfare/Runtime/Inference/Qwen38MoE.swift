import Foundation
import Metal
import TurboFieldfareFormat

struct Qwen38MoEWeights {
    let router: TensorView
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
        try Self.validateRouter(self.router,
                                numExperts: Qwen38MoE.numExperts,
                                hiddenSize: model.config.hiddenSize)
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

    private static func validateRouter(_ view: TensorView,
                                       numExperts: Int,
                                       hiddenSize: Int) throws {
        let expectedLength = UInt64(numExperts * hiddenSize * MemoryLayout<UInt16>.stride)
        guard view.dtype == GTurboFormatV1.DType.bf16.rawValue,
              view.shape.0 == UInt32(numExperts),
              view.shape.1 == UInt32(hiddenSize),
              view.shape.2 == 0, view.shape.3 == 0,
              view.length == expectedLength,
              view.scaleLength == 0, view.biasLength == 0,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 router gate metadata mismatch")
        }
    }

    private static func validateSharedExpertGate(_ view: TensorView,
                                                 hiddenSize: Int) throws {
        let groupSize = Quantization.qwen38GroupSize
        let groupCount = hiddenSize / groupSize
        let weightLength = UInt64((hiddenSize + 1) / 2)
        let auxiliaryLength = UInt64(groupCount * MemoryLayout<UInt16>.stride)
        guard hiddenSize.isMultiple(of: groupSize),
              view.dtype == GTurboFormatV1.DType.u32.rawValue,
              view.shape.0 == 1,
              view.shape.1 == UInt32(hiddenSize),
              view.shape.2 == 0, view.shape.3 == 0,
              view.length == weightLength,
              view.scaleLength == auxiliaryLength,
              view.biasLength == auxiliaryLength,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt32>.alignment)),
              view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 shared expert gate metadata mismatch")
        }
    }
}

final class Qwen38MoE {
    static let topK = QwenMoE.qwen38TopK
    static let numExperts = QwenMoE.qwen38NumExperts
    static let canonicalHiddenSize = 2560
    static let canonicalIntermediateSize = 640
    static let diagnosticSlotCount = 2

    private let routerPipeline: MTLComputePipelineState
    private let selectPipeline: MTLComputePipelineState
    private let phase1Pipeline: MTLComputePipelineState
    private let phase2Pipeline: MTLComputePipelineState
    private let sharedGatePipeline: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let routeIndices: MTLBuffer
    private let routeWeights: MTLBuffer
    private let diagnosticRouterLogits: [MTLBuffer]
    private let diagnosticRouteIndices: [MTLBuffer]
    private let diagnosticRouteWeights: [MTLBuffer]
    private let diagnosticSharedGateValue: [MTLBuffer]
    private let acts: MTLBuffer
    private let diagnosticActs: [MTLBuffer]
    private let sharedGateValue: MTLBuffer
    private let routedArgumentEncoder: MTLArgumentEncoder
    private let device: MTLDevice
    private var routedArgumentBuffers: [Int: MTLBuffer] = [:]

    static let allowedPrefillBatchCapacities = [8, 16, 32, 64, 128, 256, 512, 1024]
    static let prefillBatchCapacity: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "TURBO_FIELDFARE_QWEN38_PREFILL_BATCH"],
              let requested = Int(raw),
              allowedPrefillBatchCapacities.contains(requested) else {
            return 8
        }
        return requested
    }()

    init(context: MetalContext) throws {
        let groupConstants = [
            MetalFunctionConstant(
                index: 44,
                value: .uint32(UInt32(Quantization.qwen38GroupSize)))
        ]
        let routerConstants = groupConstants + [
            MetalFunctionConstant(index: 40, value: .uint32(UInt32(Self.numExperts))),
            MetalFunctionConstant(index: 41, value: .uint32(UInt32(Self.canonicalHiddenSize))),
            MetalFunctionConstant(index: 42, value: .uint32(UInt32(Self.topK))),
            MetalFunctionConstant(index: 43, value: .bool(true))
        ]
        let moeConstants = groupConstants + [
            MetalFunctionConstant(index: 0, value: .uint32(UInt32(Self.canonicalHiddenSize))),
            MetalFunctionConstant(index: 1, value: .uint32(UInt32(Self.canonicalIntermediateSize))),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(Self.topK))),
            MetalFunctionConstant(index: 3, value: .bool(true))
        ]
        self.routerPipeline = try context.pipeline(
            "qwen38_router_gemv", constants: routerConstants)
        self.selectPipeline = try context.pipeline(
            "qwen38_router_topk_select_k10", constants: routerConstants)
        self.phase1Pipeline = try context.pipeline(
            "qwen38_moe_phase1_gate_up_silu", constants: moeConstants)
        self.phase2Pipeline = try context.pipeline(
            "qwen38_moe_phase2_down_reduce_k10", constants: moeConstants)
        self.sharedGatePipeline = try context.pipeline(
            "qwen38_shared_expert_gate_sigmoid", constants: groupConstants)
        let diagnosticRouterLogits = (0..<Self.diagnosticSlotCount).compactMap { _ in
            context.device.makeBuffer(
                length: Self.numExperts * MemoryLayout<Float>.stride,
                options: .storageModeShared)
        }
        let diagnosticRouteIndices = (0..<Self.diagnosticSlotCount).compactMap { _ in
            context.device.makeBuffer(
                length: Self.topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)
        }
        let diagnosticRouteWeights = (0..<Self.diagnosticSlotCount).compactMap { _ in
            context.device.makeBuffer(
                length: Self.topK * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        }
        let diagnosticSharedGateValue = (0..<Self.diagnosticSlotCount).compactMap { _ in
            context.device.makeBuffer(
                length: MemoryLayout<Float>.stride,
                options: .storageModeShared)
        }
        let diagnosticActs = (0..<Self.diagnosticSlotCount).compactMap { _ in
            context.device.makeBuffer(
                length: Self.topK * Self.canonicalIntermediateSize
                    * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        }
        guard let routerLogits = context.device.makeBuffer(
            length: Self.prefillBatchCapacity * Self.numExperts
                * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let routeIndices = context.device.makeBuffer(
                  length: Self.prefillBatchCapacity * Self.topK
                      * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let routeWeights = context.device.makeBuffer(
                  length: Self.prefillBatchCapacity * Self.topK
                      * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              diagnosticRouterLogits.count == Self.diagnosticSlotCount,
              diagnosticRouteIndices.count == Self.diagnosticSlotCount,
              diagnosticRouteWeights.count == Self.diagnosticSlotCount,
              diagnosticSharedGateValue.count == Self.diagnosticSlotCount,
              diagnosticActs.count == Self.diagnosticSlotCount,
              let acts = context.device.makeBuffer(
                  length: Self.prefillBatchCapacity * Self.topK * 640
                      * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              let sharedGateValue = context.device.makeBuffer(
                  length: Self.prefillBatchCapacity * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                  name: "qwen38_moe_phase1_gate_up_silu") else {
            throw MetalError.noDevice
        }
        self.routerLogits = routerLogits
        self.routeIndices = routeIndices
        self.routeWeights = routeWeights
        self.diagnosticRouterLogits = diagnosticRouterLogits
        self.diagnosticRouteIndices = diagnosticRouteIndices
        self.diagnosticRouteWeights = diagnosticRouteWeights
        self.diagnosticSharedGateValue = diagnosticSharedGateValue
        self.acts = acts
        self.diagnosticActs = diagnosticActs
        self.sharedGateValue = sharedGateValue
        self.routedArgumentEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        self.device = context.device
    }

    func encodeRouter(commandBuffer: MTLCommandBuffer,
                      weights: Qwen38MoEWeights,
                      hidden: MTLBuffer,
                      hiddenSize: UInt32,
                      tokenIndex: Int = 0) {
        precondition(hiddenSize > 0)
        precondition(tokenIndex >= 0 && tokenIndex < Self.prefillBatchCapacity)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerPipeline)
        encoder.setBuffer(weights.router.buffer,
                          offset: Int(weights.router.offset), index: 0)
        encoder.setBuffer(hidden, offset: tokenIndex * Int(hiddenSize)
                          * MemoryLayout<Float16>.stride, index: 1)
        encoder.setBuffer(routerLogits,
                          offset: tokenIndex * Self.numExperts
                              * MemoryLayout<Float>.stride, index: 2)
        var expertCount = UInt32(Self.numExperts)
        var dimension = hiddenSize
        encoder.setBytes(&expertCount,
                         length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&dimension,
                         length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Self.numExperts + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeSelection(commandBuffer: MTLCommandBuffer,
                         weights: Qwen38MoEWeights,
                         tokenIndex: Int = 0) {
        precondition(tokenIndex >= 0 && tokenIndex < Self.prefillBatchCapacity)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(selectPipeline)
        encoder.setBuffer(routerLogits,
                          offset: tokenIndex * Self.numExperts
                              * MemoryLayout<Float>.stride, index: 0)
        encoder.setBuffer(routeIndices,
                          offset: tokenIndex * Self.topK
                              * MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(routeWeights,
                          offset: tokenIndex * Self.topK
                              * MemoryLayout<Float16>.stride, index: 2)
        var expertCount = UInt32(Self.numExperts)
        encoder.setBytes(&expertCount,
                         length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeDiagnosticSnapshot(commandBuffer: MTLCommandBuffer,
                                  tokenIndex: Int = 0,
                                  diagnosticSlot: Int = 0) {
        precondition(tokenIndex >= 0 && tokenIndex < Self.prefillBatchCapacity)
        precondition(diagnosticSlot >= 0 && diagnosticSlot < Self.diagnosticSlotCount)
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(
            from: routerLogits,
            sourceOffset: tokenIndex * Self.numExperts * MemoryLayout<Float>.stride,
            to: diagnosticRouterLogits[diagnosticSlot],
            destinationOffset: 0,
            size: Self.numExperts * MemoryLayout<Float>.stride)
        blit.copy(
            from: routeIndices,
            sourceOffset: tokenIndex * Self.topK * MemoryLayout<UInt32>.stride,
            to: diagnosticRouteIndices[diagnosticSlot],
            destinationOffset: 0,
            size: Self.topK * MemoryLayout<UInt32>.stride)
        blit.copy(
            from: routeWeights,
            sourceOffset: tokenIndex * Self.topK * MemoryLayout<Float16>.stride,
            to: diagnosticRouteWeights[diagnosticSlot],
            destinationOffset: 0,
            size: Self.topK * MemoryLayout<Float16>.stride)
        blit.endEncoding()
    }

    func selectedExperts(tokenIndex: Int = 0) -> [Int] {
        precondition(tokenIndex >= 0 && tokenIndex < Self.prefillBatchCapacity)
        let pointer = routeIndices.contents().assumingMemoryBound(to: UInt32.self)
        let start = tokenIndex * Self.topK
        return (0..<Self.topK).map { Int(pointer[start + $0]) }
    }

    func diagnosticSelectedExperts(diagnosticSlot: Int = 0) -> [Int] {
        precondition(diagnosticSlot >= 0 && diagnosticSlot < Self.diagnosticSlotCount)
        let pointer = diagnosticRouteIndices[diagnosticSlot]
            .contents().assumingMemoryBound(to: UInt32.self)
        return (0..<Self.topK).map { Int(pointer[$0]) }
    }

    func selectedRouteWeightBits(tokenIndex: Int = 0,
                                 diagnosticSlot: Int = 0) -> [UInt16] {
        precondition(tokenIndex == 0)
        precondition(diagnosticSlot >= 0 && diagnosticSlot < Self.diagnosticSlotCount)
        let pointer = diagnosticRouteWeights[diagnosticSlot]
            .contents().assumingMemoryBound(to: Float16.self)
        return (0..<Self.topK).map { pointer[$0].bitPattern }
    }

    func sharedGateValue(tokenIndex: Int = 0,
                         diagnosticSlot: Int = 0) -> Float {
        precondition(tokenIndex == 0)
        precondition(diagnosticSlot >= 0 && diagnosticSlot < Self.diagnosticSlotCount)
        let pointer = diagnosticSharedGateValue[diagnosticSlot]
            .contents().assumingMemoryBound(to: Float.self)
        return pointer[0]
    }

    func routerLogitValues(tokenIndex: Int = 0,
                           diagnosticSlot: Int = 0) -> [Float] {
        precondition(tokenIndex == 0)
        precondition(diagnosticSlot >= 0 && diagnosticSlot < Self.diagnosticSlotCount)
        let pointer = diagnosticRouterLogits[diagnosticSlot]
            .contents().assumingMemoryBound(to: Float.self)
        return (0..<Self.numExperts).map { pointer[$0] }
    }

    func makeRoutedArgumentBuffer(experts: [TensorView]) throws -> MTLBuffer {
        try makeRoutedArgumentBuffer(layer: 0, slot: 0, experts: experts)
    }

    func makeRoutedArgumentBuffer(layer: Int,
                                  slot: Int,
                                  experts: [TensorView]) throws -> MTLBuffer {
        guard layer >= 0, slot >= 0 && slot < Self.prefillBatchCapacity else {
            throw ModelError.archMismatch(
                field: "qwen38RoutedExperts.slot",
                expected: "a non-negative layer and slot in 0..<\(Self.prefillBatchCapacity)",
                actual: "layer=\(layer), slot=\(slot)")
        }
        guard experts.count == Self.topK else {
            throw ModelError.archMismatch(
                field: "qwen38RoutedExperts.count",
                expected: "\(Self.topK)",
                actual: "\(experts.count)")
        }
        let cacheKey = layer * Self.prefillBatchCapacity + slot
        let routedArgumentBuffer: MTLBuffer
        if let cached = routedArgumentBuffers[cacheKey] {
            routedArgumentBuffer = cached
        } else {
            guard let allocated = device.makeBuffer(
                length: routedArgumentEncoder.encodedLength,
                options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            routedArgumentBuffers[cacheKey] = allocated
            routedArgumentBuffer = allocated
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
                      routedResources: [MTLBuffer],
                      hiddenSize: UInt32,
                      intermediateSize: UInt32,
                      sharedExpertGateWeight: TensorView,
                      tokenIndex: Int = 0,
                      captureDiagnostics: Bool = false,
                      diagnosticSlot: Int = 0) {
        precondition(sharedExpertGateWeight.dtype == GTurboFormatV1.DType.u32.rawValue)
        precondition(tokenIndex >= 0 && tokenIndex < Self.prefillBatchCapacity)
        precondition(diagnosticSlot >= 0 && diagnosticSlot < Self.diagnosticSlotCount)
        guard let gateEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        gateEncoder.setComputePipelineState(sharedGatePipeline)
        gateEncoder.setBuffer(input, offset: tokenIndex * Int(hiddenSize)
                              * MemoryLayout<Float16>.stride, index: 0)
        gateEncoder.setBuffer(sharedExpertGateWeight.buffer,
                              offset: Int(sharedExpertGateWeight.offset), index: 1)
        gateEncoder.setBuffer(sharedExpertGateWeight.buffer,
                              offset: Int(sharedExpertGateWeight.scaleOffset), index: 2)
        gateEncoder.setBuffer(sharedExpertGateWeight.buffer,
                              offset: Int(sharedExpertGateWeight.biasOffset), index: 3)
        gateEncoder.setBuffer(sharedGateValue,
                              offset: tokenIndex * MemoryLayout<Float>.stride, index: 4)
        var gateDimension = hiddenSize
        gateEncoder.setBytes(&gateDimension,
                             length: MemoryLayout<UInt32>.stride, index: 5)
        let gateThreads = sharedGatePipeline.threadExecutionWidth
        gateEncoder.dispatchThreads(
            MTLSize(width: gateThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: gateThreads, height: 1, depth: 1))
        gateEncoder.endEncoding()
        if captureDiagnostics {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            blit.copy(
                from: sharedGateValue,
                sourceOffset: tokenIndex * MemoryLayout<Float>.stride,
                to: diagnosticSharedGateValue[diagnosticSlot],
                destinationOffset: 0,
                size: MemoryLayout<Float>.stride)
            blit.endEncoding()
        }
        precondition(intermediateSize > 0 && intermediateSize <= 640)
        guard let phase1 = commandBuffer.makeComputeCommandEncoder() else { return }
        phase1.setComputePipelineState(phase1Pipeline)
        for resource in routedResources {
            phase1.useResource(resource, usage: .read)
        }
        phase1.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        var offsets = routedOffsets
        phase1.setBytes(&offsets,
                length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        phase1.setBuffer(input, offset: tokenIndex * Int(hiddenSize)
                         * MemoryLayout<Float16>.stride, index: 2)
        phase1.setBuffer(acts, offset: tokenIndex * Self.topK * Int(intermediateSize)
                         * MemoryLayout<Float16>.stride, index: 3)
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

        if captureDiagnostics {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            blit.copy(
                from: acts,
                sourceOffset: tokenIndex * Self.topK * Int(intermediateSize)
                    * MemoryLayout<Float16>.stride,
                to: diagnosticActs[diagnosticSlot],
                destinationOffset: 0,
                size: Self.topK * Int(intermediateSize)
                    * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }

        guard let phase2 = commandBuffer.makeComputeCommandEncoder() else { return }
        phase2.setComputePipelineState(phase2Pipeline)
        for resource in routedResources {
            phase2.useResource(resource, usage: .read)
        }
        phase2.setBuffer(routedArgumentBuffer, offset: 0, index: 0)
        phase2.setBytes(&offsets,
                length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        phase2.setBuffer(acts, offset: tokenIndex * Self.topK * Int(intermediateSize)
                         * MemoryLayout<Float16>.stride, index: 2)
        phase2.setBuffer(routeWeights, offset: tokenIndex * Self.topK
                         * MemoryLayout<Float16>.stride, index: 3)
        phase2.setBuffer(residual, offset: tokenIndex * Int(hiddenSize)
                         * MemoryLayout<Float16>.stride, index: 4)
        phase2.setBuffer(output, offset: tokenIndex * Int(hiddenSize)
                         * MemoryLayout<Float16>.stride, index: 5)
        phase2.setBytes(&dimension,
                        length: MemoryLayout<UInt32>.stride, index: 6)
        phase2.setBytes(&intermediate,
                        length: MemoryLayout<UInt32>.stride, index: 7)
        phase2.setBuffer(sharedGateValue,
                         offset: tokenIndex * MemoryLayout<Float>.stride, index: 8)
        phase2.dispatchThreadgroups(
            MTLSize(width: Int(hiddenSize), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.topK * 32,
                                            height: 1, depth: 1))
        phase2.endEncoding()
    }

    func diagnosticActivationBuffer(slot: Int = 0) -> MTLBuffer {
        precondition(slot >= 0 && slot < Self.diagnosticSlotCount)
        return diagnosticActs[slot]
    }

    func planSelectedExperts(model: Model, layer: Int,
                             tokenIndex: Int = 0) throws -> RoutedExpertFetchPlan {
        let experts = selectedExperts(tokenIndex: tokenIndex)
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
        return plan
    }

    func fetchSelectedExperts(model: Model, layer: Int,
                               tokenIndex: Int = 0) async throws
        -> (plan: RoutedExpertFetchPlan,
            views: [TensorView], offsets: MoEExpertOffsets,
            cacheHits: Int, cacheMisses: Int,
            readDiagnostics: ExpertReadDiagnostics) {
        let plan = try planSelectedExperts(
            model: model, layer: layer, tokenIndex: tokenIndex)
        let fetch = try await model.fetchRoutedExpertsWithDiagnostics(plan: plan)
        return (plan, fetch.views, model.routedExpertOffsets(layer: layer),
                plan.hits, plan.misses.count, fetch.readDiagnostics)
    }
}
