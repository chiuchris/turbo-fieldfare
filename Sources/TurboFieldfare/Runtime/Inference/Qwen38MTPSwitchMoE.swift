import Foundation
import Metal
import TurboFieldfareFormat

struct Qwen38MTPSwitchMoEGeometry: Equatable {
    let expertCount: Int
    let topK: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let groupSize: Int

    static let qwen = Qwen38MTPSwitchMoEGeometry(
        expertCount: 512,
        topK: 10,
        hiddenSize: 2_560,
        intermediateSize: 640,
        groupSize: 32)

    var expertWeightElements: Int {
        intermediateSize * hiddenSize
    }

    var expertWeightBytes: UInt64 {
        UInt64(expertWeightElements / 2)
    }

    var expertAuxiliaryBytes: UInt64 {
        UInt64(expertWeightElements / groupSize * MemoryLayout<UInt16>.stride)
    }
}

struct Qwen38MTPSwitchExpertViews {
    let expertIndex: Int
    let gate: TensorView
    let up: TensorView
    let down: TensorView
}

struct Qwen38MTPSwitchMoEWeights {
    let router: TensorView
    let gate: TensorView
    let up: TensorView
    let down: TensorView
    let sharedExpertGate: TensorView
    let sharedExpertUp: TensorView
    let sharedExpertDown: TensorView
    let sharedExpertMultiplier: TensorView
    let geometry: Qwen38MTPSwitchMoEGeometry

    init(mtp: Qwen38MTPWeights,
         geometry: Qwen38MTPSwitchMoEGeometry = .qwen) throws {
        self.geometry = geometry
        self.router = try mtp.tensor(role: .mlpRouter)
        self.gate = try mtp.tensor(role: .switchExpertGate)
        self.up = try mtp.tensor(role: .switchExpertUp)
        self.down = try mtp.tensor(role: .switchExpertDown)
        self.sharedExpertGate = try mtp.tensor(role: .sharedExpertGate)
        self.sharedExpertUp = try mtp.tensor(role: .sharedExpertUp)
        self.sharedExpertDown = try mtp.tensor(role: .sharedExpertDown)
        self.sharedExpertMultiplier = try mtp.tensor(role: .sharedExpertMultiplier)
        try Self.validateRouter(self.router, geometry: geometry)
        try Self.validateAffine(self.sharedExpertGate, role: .sharedExpertGate,
                               rows: geometry.intermediateSize,
                               columns: geometry.hiddenSize)
        try Self.validateAffine(self.sharedExpertUp, role: .sharedExpertUp,
                               rows: geometry.intermediateSize,
                               columns: geometry.hiddenSize)
        try Self.validateAffine(self.sharedExpertDown, role: .sharedExpertDown,
                               rows: geometry.hiddenSize,
                               columns: geometry.intermediateSize)
        try Self.validateAffine(self.sharedExpertMultiplier,
                               role: .sharedExpertMultiplier,
                               rows: 1,
                               columns: geometry.hiddenSize)
        try Self.validateStacked(self.gate, geometry: geometry,
                                 role: Qwen38MTPRole.switchExpertGate,
                                 rows: geometry.intermediateSize,
                                 columns: geometry.hiddenSize)
        try Self.validateStacked(self.up, geometry: geometry,
                                 role: Qwen38MTPRole.switchExpertUp,
                                 rows: geometry.intermediateSize,
                                 columns: geometry.hiddenSize)
        try Self.validateStacked(self.down, geometry: geometry,
                                 role: Qwen38MTPRole.switchExpertDown,
                                 rows: geometry.hiddenSize,
                                 columns: geometry.intermediateSize)
    }

    func expertViews(indices: [Int]) throws -> [Qwen38MTPSwitchExpertViews] {
        guard indices.count == geometry.topK,
              Set(indices).count == geometry.topK,
              indices.allSatisfy({ $0 >= 0 && $0 < geometry.expertCount }) else {
            throw ModelError.archMismatch(
                field: "mtp.switchMoE.experts",
                expected: "ten unique experts in 0..<512",
                actual: "\(indices)")
        }
        return try indices.map { index in
            Qwen38MTPSwitchExpertViews(
                expertIndex: index,
                gate: try Self.expertView(
                    gate, expert: index, geometry: geometry,
                    rows: geometry.intermediateSize,
                    columns: geometry.hiddenSize),
                up: try Self.expertView(
                    up, expert: index, geometry: geometry,
                    rows: geometry.intermediateSize,
                    columns: geometry.hiddenSize),
                down: try Self.expertView(
                    down, expert: index, geometry: geometry,
                    rows: geometry.hiddenSize,
                    columns: geometry.intermediateSize))
        }
    }

    private static func validateRouter(
        _ view: TensorView,
        geometry: Qwen38MTPSwitchMoEGeometry
    ) throws {
        let elementCount = UInt64(geometry.expertCount * geometry.hiddenSize)
        let q8AuxiliaryBytes = elementCount / 64 * UInt64(MemoryLayout<UInt16>.stride)
        let shapeMatches = view.shape.0 == UInt32(geometry.expertCount)
            && view.shape.1 == UInt32(geometry.hiddenSize)
            && view.shape.2 == 0 && view.shape.3 == 0
        let denseMatches = view.dtype == GTurboFormatV1.DType.bf16.rawValue
            && view.length == elementCount * UInt64(MemoryLayout<UInt16>.stride)
            && view.scaleLength == 0 && view.biasLength == 0
            && view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment))
        let q8Matches = view.dtype == GTurboFormatV1.DType.u32.rawValue
            && view.quantization == TensorQuantizationDescriptor(bits: 8, groupSize: 64)
            && view.length == elementCount
            && view.scaleLength == q8AuxiliaryBytes
            && view.biasLength == q8AuxiliaryBytes
            && view.offset.isMultiple(of: UInt64(MemoryLayout<UInt32>.alignment))
            && view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment))
            && view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment))
        guard shapeMatches, denseMatches || q8Matches,
              view.offset + view.length <= UInt64(view.buffer.length),
              view.scaleOffset + view.scaleLength <= UInt64(view.buffer.length),
              view.biasOffset + view.biasLength <= UInt64(view.buffer.length) else {
            throw ModelError.indexCorrupt(
                detail: "MTP switch-MoE router metadata mismatch")
        }
    }

    private static func validateAffine(
        _ view: TensorView,
        role: Qwen38MTPRole,
        rows: Int,
        columns: Int
    ) throws {
        let elementCount = UInt64(rows * columns)
        let quantization = view.quantization
        let isQ4 = quantization == TensorQuantizationDescriptor(bits: 4, groupSize: 32)
        let isQ8 = quantization == TensorQuantizationDescriptor(bits: 8, groupSize: 64)
        let groupSize = UInt64(quantization?.groupSize ?? 1)
        let bits = UInt64(quantization?.bits ?? 0)
        let auxiliaryBytes = elementCount / groupSize * UInt64(MemoryLayout<UInt16>.stride)
        guard (isQ4 || isQ8),
              columns.isMultiple(of: Int(groupSize)),
              view.dtype == GTurboFormatV1.DType.u32.rawValue,
              view.shape.0 == UInt32(rows),
              view.shape.1 == UInt32(columns),
              view.shape.2 == 0, view.shape.3 == 0,
              view.length == elementCount * bits / 8,
              view.scaleLength == auxiliaryBytes,
              view.biasLength == auxiliaryBytes,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt32>.alignment)),
              view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.offset + view.length <= UInt64(view.buffer.length),
              view.scaleOffset + view.scaleLength <= UInt64(view.buffer.length),
              view.biasOffset + view.biasLength <= UInt64(view.buffer.length) else {
            throw ModelError.indexCorrupt(
                detail: "MTP switch-MoE \(role.rawValue) Q8 metadata mismatch")
        }
    }

    private static func validateStacked(
        _ view: TensorView,
        geometry: Qwen38MTPSwitchMoEGeometry,
        role: Qwen38MTPRole,
        rows: Int,
        columns: Int
    ) throws {
        let elementCount = UInt64(geometry.expertCount * rows * columns)
        let auxiliaryBytes = elementCount / UInt64(geometry.groupSize)
            * UInt64(MemoryLayout<UInt16>.stride)
        guard columns.isMultiple(of: geometry.groupSize),
              view.dtype == GTurboFormatV1.DType.u32.rawValue,
              view.quantization == TensorQuantizationDescriptor(bits: 4, groupSize: 32),
              view.shape.0 == UInt32(geometry.expertCount),
              view.shape.1 == UInt32(rows),
              view.shape.2 == UInt32(columns),
              view.shape.3 == 0,
              view.length == elementCount / 2,
              view.scaleLength == auxiliaryBytes,
              view.biasLength == auxiliaryBytes,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt32>.alignment)),
              view.scaleOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.biasOffset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)),
              view.offset + view.length <= UInt64(view.buffer.length),
              view.scaleOffset + view.scaleLength <= UInt64(view.buffer.length),
              view.biasOffset + view.biasLength <= UInt64(view.buffer.length) else {
            throw ModelError.indexCorrupt(
                detail: "MTP switch-MoE \(role.rawValue) stacked Q4 metadata mismatch")
        }
    }

    private static func expertView(
        _ stacked: TensorView,
        expert: Int,
        geometry: Qwen38MTPSwitchMoEGeometry,
        rows: Int,
        columns: Int
    ) throws -> TensorView {
        let weightBytes = UInt64(rows * columns / 2)
        let auxiliaryBytes = UInt64(rows * columns / geometry.groupSize
                                    * MemoryLayout<UInt16>.stride)
        let expertOffset = UInt64(expert) * weightBytes
        let auxiliaryOffset = UInt64(expert) * auxiliaryBytes
        let offset = stacked.offset + expertOffset
        let scaleOffset = stacked.scaleOffset + auxiliaryOffset
        let biasOffset = stacked.biasOffset + auxiliaryOffset
        guard offset + weightBytes <= UInt64(stacked.buffer.length),
              scaleOffset + auxiliaryBytes <= UInt64(stacked.buffer.length),
              biasOffset + auxiliaryBytes <= UInt64(stacked.buffer.length) else {
            throw ModelError.indexCorrupt(
                detail: "MTP switch-MoE expert offset exceeds resident buffer")
        }
        return TensorView(
            buffer: stacked.buffer,
            offset: offset,
            length: weightBytes,
            scaleOffset: scaleOffset,
            scaleLength: auxiliaryBytes,
            biasOffset: biasOffset,
            biasLength: auxiliaryBytes,
            shape: (UInt32(rows), UInt32(columns), 0, 0),
            dtype: stacked.dtype,
            quantization: stacked.quantization)
    }
}

final class Qwen38MTPSwitchMoEExecutor {
    let geometry: Qwen38MTPSwitchMoEGeometry

    private let routerPipeline: MTLComputePipelineState
    private let routerQ8Pipeline: MTLComputePipelineState
    private let selectPipeline: MTLComputePipelineState
    private let phase1Pipeline: MTLComputePipelineState
    private let phase2Pipeline: MTLComputePipelineState
    private let sharedGatePipeline: MTLComputePipelineState
    private let sharedGateQ8Pipeline: MTLComputePipelineState
    private let sharedExpert: QwenSharedExpertInt4
    private let routerLogits: MTLBuffer
    private let routeIndices: MTLBuffer
    private let routeWeights: MTLBuffer
    private let acts: MTLBuffer
    private let sharedGate: MTLBuffer
    private let expertArgumentEncoder: MTLArgumentEncoder
    private let expertArgumentBuffer: MTLBuffer

    init(context: MetalContext,
         geometry: Qwen38MTPSwitchMoEGeometry = .qwen) throws {
        guard geometry == .qwen else {
            throw ModelError.archMismatch(
                field: "mtp.switchMoE.geometry",
                expected: "Qwen3.8 Flash-Next geometry",
                actual: "custom geometry")
        }
        self.geometry = geometry
        let groupConstants = [
            MetalFunctionConstant(
                index: 44,
                value: .uint32(UInt32(geometry.groupSize)))
        ]
        let routerConstants = groupConstants + [
            MetalFunctionConstant(index: 40, value: .uint32(UInt32(geometry.expertCount))),
            MetalFunctionConstant(index: 41, value: .uint32(UInt32(geometry.hiddenSize))),
            MetalFunctionConstant(index: 42, value: .uint32(UInt32(geometry.topK))),
            MetalFunctionConstant(index: 43, value: .bool(true))
        ]
        let routerQ8Constants = [
            MetalFunctionConstant(index: 44, value: .uint32(64)),
            MetalFunctionConstant(index: 40, value: .uint32(UInt32(geometry.expertCount))),
            MetalFunctionConstant(index: 41, value: .uint32(UInt32(geometry.hiddenSize))),
            MetalFunctionConstant(index: 42, value: .uint32(UInt32(geometry.topK))),
            MetalFunctionConstant(index: 43, value: .bool(true))
        ]
        let moeConstants = groupConstants + [
            MetalFunctionConstant(index: 0, value: .uint32(UInt32(geometry.hiddenSize))),
            MetalFunctionConstant(index: 1, value: .uint32(UInt32(geometry.intermediateSize))),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(geometry.topK))),
            MetalFunctionConstant(index: 3, value: .bool(true))
        ]
        self.routerPipeline = try context.pipeline(
            "qwen38_router_gemv", constants: routerConstants)
        self.routerQ8Pipeline = try context.pipeline(
            "qwen38_router_gemv_q8", constants: routerQ8Constants)
        self.selectPipeline = try context.pipeline(
            "qwen38_router_topk_select_k10", constants: routerConstants)
        self.phase1Pipeline = try context.pipeline(
            "qwen38_mtp_moe_phase1_gate_up_silu", constants: moeConstants)
        self.phase2Pipeline = try context.pipeline(
            "qwen38_mtp_moe_phase2_down_reduce_k10", constants: moeConstants)
        self.sharedGatePipeline = try context.pipeline(
            "qwen38_shared_expert_gate_sigmoid", constants: groupConstants)
        self.sharedGateQ8Pipeline = try context.pipeline(
            "qwen38_shared_expert_gate_sigmoid_q8", constants: routerQ8Constants)
        self.sharedExpert = try QwenSharedExpertInt4(context: context)

        func makeBuffer(length: Int) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }
        self.routerLogits = try makeBuffer(
            length: geometry.expertCount * MemoryLayout<Float>.stride)
        self.routeIndices = try makeBuffer(
            length: geometry.topK * MemoryLayout<UInt32>.stride)
        self.routeWeights = try makeBuffer(
            length: geometry.topK * MemoryLayout<Float16>.stride)
        self.acts = try makeBuffer(
            length: geometry.topK * geometry.intermediateSize
                * MemoryLayout<Float16>.stride)
        self.sharedGate = try makeBuffer(length: MemoryLayout<Float>.stride)

        guard let phase1Function = context.library.makeFunction(
            name: "qwen38_mtp_moe_phase1_gate_up_silu") else {
            throw MetalError.noDevice
        }
        self.expertArgumentEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        self.expertArgumentBuffer = try makeBuffer(
            length: expertArgumentEncoder.encodedLength)
    }

    func encodeRouter(commandBuffer: MTLCommandBuffer,
                      weights: Qwen38MTPSwitchMoEWeights,
                      hidden: MTLBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var expertCount = UInt32(geometry.expertCount)
        if weights.router.quantization?.bits == 8 {
            encoder.setComputePipelineState(routerQ8Pipeline)
            encoder.setBuffer(weights.router.buffer,
                              offset: Int(weights.router.offset), index: 0)
            encoder.setBuffer(weights.router.buffer,
                              offset: Int(weights.router.scaleOffset), index: 1)
            encoder.setBuffer(weights.router.buffer,
                              offset: Int(weights.router.biasOffset), index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(routerLogits, offset: 0, index: 4)
            var hiddenSize = UInt32(geometry.hiddenSize)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&hiddenSize, length: MemoryLayout<UInt32>.stride, index: 6)
        } else {
            encoder.setComputePipelineState(routerPipeline)
            encoder.setBuffer(weights.router.buffer,
                              offset: Int(weights.router.offset), index: 0)
            encoder.setBuffer(hidden, offset: 0, index: 1)
            encoder.setBuffer(routerLogits, offset: 0, index: 2)
            var hiddenSize = UInt32(geometry.hiddenSize)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&hiddenSize, length: MemoryLayout<UInt32>.stride, index: 4)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: (geometry.expertCount + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()

        guard let selector = commandBuffer.makeComputeCommandEncoder() else { return }
        selector.setComputePipelineState(selectPipeline)
        selector.setBuffer(routerLogits, offset: 0, index: 0)
        selector.setBuffer(routeIndices, offset: 0, index: 1)
        selector.setBuffer(routeWeights, offset: 0, index: 2)
        selector.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 3)
        selector.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        selector.endEncoding()
    }

    func selectedExperts() -> [Int] {
        let pointer = routeIndices.contents().assumingMemoryBound(to: UInt32.self)
        return (0..<geometry.topK).map { Int(pointer[$0]) }
    }

    func selectedRouteWeights() -> [Float] {
        let pointer = routeWeights.contents().assumingMemoryBound(to: Float16.self)
        return (0..<geometry.topK).map { Float(pointer[$0]) }
    }

    func selectedRoutes() -> (experts: [Int], weights: [Float]) {
        (selectedExperts(), selectedRouteWeights())
    }

    func emitRouterDiagnostics(weights: Qwen38MTPSwitchMoEWeights,
                               hidden: MTLBuffer) {
        let hiddenPointer = hidden.contents().assumingMemoryBound(to: Float16.self)
        var expectedLogits = [Float](repeating: 0, count: geometry.expertCount)
        for expert in 0..<geometry.expertCount {
            expectedLogits[expert] = Self.affineQ4Dot(
                weights.router, row: expert, input: hiddenPointer,
                count: geometry.hiddenSize, groupSize: geometry.hiddenSize)
        }

        let actualLogits = routerLogits.contents()
            .assumingMemoryBound(to: Float.self)
        var logitMaxError: Float = 0
        var logitSquaredError: Float = 0
        for expert in 0..<geometry.expertCount {
            let error = abs(actualLogits[expert] - expectedLogits[expert])
            logitMaxError = max(logitMaxError, error)
            logitSquaredError += error * error
        }

        let maxLogit = expectedLogits.max() ?? -.infinity
        let scores = expectedLogits.map { exp($0 - maxLogit) }
        let expectedExperts = scores.indices.sorted {
            if scores[$0] == scores[$1] { return $0 < $1 }
            return scores[$0] > scores[$1]
        }.prefix(geometry.topK)
        let expectedWeights = expectedExperts.map {
            Float16(scores[$0] / expectedExperts.reduce(Float(0)) {
                $0 + scores[$1]
            })
        }.map(Float.init)
        let actualExperts = selectedExperts()
        let actualWeights = selectedRouteWeights()
        var weightMaxError: Float = 0
        var weightSquaredError: Float = 0
        for index in 0..<geometry.topK {
            let error = abs(actualWeights[index] - expectedWeights[index])
            weightMaxError = max(weightMaxError, error)
            weightSquaredError += error * error
        }
        let indicesMatch = actualExperts == Array(expectedExperts)
        let line = String(format: "mtp router logits_max_abs=%+.6e logits_rms=%.6e "
                          + "indices_match=%@ weights_max_abs=%+.6e weights_rms=%.6e "
                          + "expected=%@ actual=%@\\n",
                          logitMaxError,
                          sqrt(logitSquaredError / Float(geometry.expertCount)),
                          indicesMatch ? "true" : "false",
                          weightMaxError,
                          sqrt(weightSquaredError / Float(geometry.topK)),
                          Array(expectedExperts).map(String.init).joined(separator: ","),
                          actualExperts.map(String.init).joined(separator: ","))
        FileHandle.standardError.write(Data(line.utf8))
    }

    func emitSharedExpertDiagnostics(weights: Qwen38MTPSwitchMoEWeights,
                                     input: MTLBuffer,
                                     gate: MTLBuffer,
                                     up: MTLBuffer,
                                     act: MTLBuffer,
                                     output: MTLBuffer) {
        let inputPointer = input.contents().assumingMemoryBound(to: Float16.self)
        var expectedGate = [Float16](repeating: 0, count: geometry.intermediateSize)
        var expectedUp = [Float16](repeating: 0, count: geometry.intermediateSize)
        for row in 0..<geometry.intermediateSize {
            expectedGate[row] = Float16(Self.affineQ4Dot(
                weights.sharedExpertGate, row: row, input: inputPointer,
                count: geometry.hiddenSize, groupSize: geometry.groupSize))
            expectedUp[row] = Float16(Self.affineQ4Dot(
                weights.sharedExpertUp, row: row, input: inputPointer,
                count: geometry.hiddenSize, groupSize: geometry.groupSize))
        }
        var expectedAct = [Float16](repeating: 0, count: geometry.intermediateSize)
        for row in 0..<geometry.intermediateSize {
            let gateValue = Float(expectedGate[row])
            let upValue = Float(expectedUp[row])
            expectedAct[row] = Float16(
                (gateValue / (1 + exp(-gateValue))) * upValue)
        }
        var expectedOutput = [Float16](repeating: 0, count: geometry.hiddenSize)
        expectedAct.withUnsafeBufferPointer { actBuffer in
            for row in 0..<geometry.hiddenSize {
                expectedOutput[row] = Float16(Self.affineQ4Dot(
                    weights.sharedExpertDown, row: row,
                    input: actBuffer.baseAddress!,
                    count: geometry.intermediateSize,
                    groupSize: geometry.groupSize))
            }
        }
        emitSharedExpertError(label: "gate", actual: gate,
                              expected: expectedGate)
        emitSharedExpertError(label: "up", actual: up, expected: expectedUp)
        emitSharedExpertError(label: "act", actual: act, expected: expectedAct)
        emitSharedExpertError(label: "output", actual: output,
                              expected: expectedOutput)
    }

    private func emitSharedExpertError(label: String,
                                       actual: MTLBuffer,
                                       expected: [Float16]) {
        let actualPointer = actual.contents().assumingMemoryBound(to: Float16.self)
        var maxError: Float = 0
        var squaredError: Float = 0
        for index in expected.indices {
            let error = abs(Float(actualPointer[index]) - Float(expected[index]))
            maxError = max(maxError, error)
            squaredError += error * error
        }
        let rmsError = sqrt(squaredError / Float(expected.count))
        let line = String(format: "mtp shared_moe relation=%@ n=%d max_abs=%+.6e "
                          + "rms_error=%.6e\\n",
                          label, expected.count, maxError, rmsError)
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static func affineQ4Dot(_ view: TensorView,
                                    row: Int,
                                    input: UnsafePointer<Float16>,
                                    count: Int,
                                    groupSize: Int) -> Float {
        if view.quantization == nil {
            let weights = view.buffer.contents()
                .advanced(by: Int(view.offset))
                .assumingMemoryBound(to: UInt16.self)
            let rowBase = row * count
            var result: Float = 0
            for index in 0..<count {
                result += Quantization.bf16ToFloat(weights[rowBase + index])
                    * Float(input[index])
            }
            return result
        }

        let bits = view.quantization?.bits ?? 4
        let actualGroupSize = view.quantization?.groupSize ?? groupSize
        let weights = view.buffer.contents()
            .advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt8.self)
        let scales = view.buffer.contents()
            .advanced(by: Int(view.scaleOffset))
            .assumingMemoryBound(to: UInt16.self)
        let biases = view.buffer.contents()
            .advanced(by: Int(view.biasOffset))
            .assumingMemoryBound(to: UInt16.self)
        let groups = count / actualGroupSize
        let rowBytes = bits == 4 ? count / 2 : count
        var result: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(scales[row * groups + group])
            let bias = Quantization.bf16ToFloat(biases[row * groups + group])
            let byteBase = row * rowBytes
                + group * (bits == 4 ? actualGroupSize / 2 : actualGroupSize)
            for index in 0..<actualGroupSize {
                let packed = weights[byteBase + (bits == 4 ? index / 2 : index)]
                let quant = bits == 4
                    ? (index.isMultiple(of: 2) ? packed & 0x0F : packed >> 4)
                    : packed
                let value = Float(quant) * scale + bias
                result += value * Float(input[group * actualGroupSize + index])
            }
        }
        return result
    }

    func emitRoutedExpertDiagnostics(experts: [Qwen38MTPSwitchExpertViews],
                                     input: MTLBuffer,
                                     sharedOutput: MTLBuffer,
                                     output: MTLBuffer) {
        let inputPointer = input.contents().assumingMemoryBound(to: Float16.self)
        var expectedActs = [Float16](repeating: 0,
                                     count: geometry.topK * geometry.intermediateSize)
        for (slot, expert) in experts.enumerated() {
            for row in 0..<geometry.intermediateSize {
                let gate = Self.affineQ4Dot(
                    expert.gate, row: row, input: inputPointer,
                    count: geometry.hiddenSize, groupSize: geometry.groupSize)
                let up = Self.affineQ4Dot(
                    expert.up, row: row, input: inputPointer,
                    count: geometry.hiddenSize, groupSize: geometry.groupSize)
                expectedActs[slot * geometry.intermediateSize + row] = Float16(
                    (gate / (1 + exp(-gate))) * up)
            }
        }

        let sharedGatePointer = sharedGate.contents()
            .assumingMemoryBound(to: Float.self)
        let actualSharedOutput = sharedOutput.contents()
            .assumingMemoryBound(to: Float16.self)
        var expectedOutput = [Float16](repeating: 0, count: geometry.hiddenSize)
        let routeWeights = selectedRouteWeights()
        expectedActs.withUnsafeBufferPointer { actsBuffer in
            for row in 0..<geometry.hiddenSize {
                var value = Float(actualSharedOutput[row]) * sharedGatePointer[0]
                for slot in 0..<geometry.topK {
                    let expert = experts[slot]
                    let down = Self.affineQ4Dot(
                        expert.down, row: row,
                        input: actsBuffer.baseAddress!
                            .advanced(by: slot * geometry.intermediateSize),
                        count: geometry.intermediateSize,
                        groupSize: geometry.groupSize)
                    value += routeWeights[slot] * down
                }
                expectedOutput[row] = Float16(value)
            }
        }
        emitSharedExpertError(label: "routed_act", actual: self.acts,
                              expected: expectedActs)
        emitSharedExpertError(label: "routed_output", actual: output,
                              expected: expectedOutput)
    }

    func encodeMLP(commandBuffer: MTLCommandBuffer,
                   weights: Qwen38MTPSwitchMoEWeights,
                   input: MTLBuffer,
                   output: MTLBuffer,
                   sharedOutput: MTLBuffer,
                   scratchGate: MTLBuffer,
                   scratchUp: MTLBuffer,
                   scratchAct: MTLBuffer,
                   experts: [Qwen38MTPSwitchExpertViews]) throws {
        guard experts.count == geometry.topK else {
            throw ModelError.archMismatch(
                field: "mtp.switchMoE.experts",
                expected: "ten selected expert views",
                actual: "\(experts.count)")
        }
        let sharedGateProjection = Self.projection(
            weights.sharedExpertGate,
            rows: geometry.intermediateSize,
            columns: geometry.hiddenSize)
        let sharedUpProjection = Self.projection(
            weights.sharedExpertUp,
            rows: geometry.intermediateSize,
            columns: geometry.hiddenSize)
        let sharedDownProjection = Self.projection(
            weights.sharedExpertDown,
            rows: geometry.hiddenSize,
            columns: geometry.intermediateSize)
        if weights.sharedExpertGate.quantization?.bits == 8 {
            try sharedExpert.encodeAffine(
                commandBuffer: commandBuffer,
                x: input,
                gate: sharedGateProjection,
                up: sharedUpProjection,
                down: sharedDownProjection,
                y: sharedOutput,
                scratchGate: scratchGate,
                scratchUp: scratchUp,
                scratchAct: scratchAct)
        } else {
            try sharedExpert.encode(
                commandBuffer: commandBuffer,
                x: input,
                gate: sharedGateProjection,
                up: sharedUpProjection,
                down: sharedDownProjection,
                y: sharedOutput,
                scratchGate: scratchGate,
                scratchUp: scratchUp,
                scratchAct: scratchAct)
        }

        guard let gate = commandBuffer.makeComputeCommandEncoder() else { return }
        gate.setComputePipelineState(
            weights.sharedExpertMultiplier.quantization?.bits == 8
                ? sharedGateQ8Pipeline : sharedGatePipeline)
        let multiplier = weights.sharedExpertMultiplier
        gate.setBuffer(input, offset: 0, index: 0)
        gate.setBuffer(multiplier.buffer, offset: Int(multiplier.offset), index: 1)
        gate.setBuffer(multiplier.buffer, offset: Int(multiplier.scaleOffset), index: 2)
        gate.setBuffer(multiplier.buffer, offset: Int(multiplier.biasOffset), index: 3)
        gate.setBuffer(sharedGate, offset: 0, index: 4)
        var hiddenSize = UInt32(geometry.hiddenSize)
        gate.setBytes(&hiddenSize, length: MemoryLayout<UInt32>.stride, index: 5)
        gate.dispatchThreads(
            MTLSize(width: 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        gate.endEncoding()

        expertArgumentEncoder.setArgumentBuffer(expertArgumentBuffer, offset: 0)
        for (index, expert) in experts.enumerated() {
            expertArgumentEncoder.setBuffer(
                expert.gate.buffer, offset: Int(expert.gate.offset), index: index)
            expertArgumentEncoder.setBuffer(
                expert.gate.buffer, offset: Int(expert.gate.scaleOffset), index: 10 + index)
            expertArgumentEncoder.setBuffer(
                expert.gate.buffer, offset: Int(expert.gate.biasOffset), index: 20 + index)
            expertArgumentEncoder.setBuffer(
                expert.up.buffer, offset: Int(expert.up.offset), index: 30 + index)
            expertArgumentEncoder.setBuffer(
                expert.up.buffer, offset: Int(expert.up.scaleOffset), index: 40 + index)
            expertArgumentEncoder.setBuffer(
                expert.up.buffer, offset: Int(expert.up.biasOffset), index: 50 + index)
            expertArgumentEncoder.setBuffer(
                expert.down.buffer, offset: Int(expert.down.offset), index: 60 + index)
            expertArgumentEncoder.setBuffer(
                expert.down.buffer, offset: Int(expert.down.scaleOffset), index: 70 + index)
            expertArgumentEncoder.setBuffer(
                expert.down.buffer, offset: Int(expert.down.biasOffset), index: 80 + index)
        }

        guard let phase1 = commandBuffer.makeComputeCommandEncoder() else { return }
        phase1.setComputePipelineState(phase1Pipeline)
        phase1.setBuffer(expertArgumentBuffer, offset: 0, index: 0)
        phase1.setBuffer(input, offset: 0, index: 1)
        phase1.setBuffer(acts, offset: 0, index: 2)
        var hidden = UInt32(geometry.hiddenSize)
        var intermediate = UInt32(geometry.intermediateSize)
        phase1.setBytes(&hidden, length: MemoryLayout<UInt32>.stride, index: 3)
        phase1.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 4)
        phase1.dispatchThreadgroups(
            MTLSize(width: (geometry.topK * geometry.intermediateSize + 7) / 8,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        phase1.endEncoding()

        guard let phase2 = commandBuffer.makeComputeCommandEncoder() else { return }
        phase2.setComputePipelineState(phase2Pipeline)
        phase2.setBuffer(expertArgumentBuffer, offset: 0, index: 0)
        phase2.setBuffer(acts, offset: 0, index: 1)
        phase2.setBuffer(routeWeights, offset: 0, index: 2)
        phase2.setBuffer(sharedOutput, offset: 0, index: 3)
        phase2.setBuffer(output, offset: 0, index: 4)
        phase2.setBytes(&hidden, length: MemoryLayout<UInt32>.stride, index: 5)
        phase2.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 6)
        phase2.setBuffer(sharedGate, offset: 0, index: 7)
        phase2.dispatchThreadgroups(
            MTLSize(width: geometry.hiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: geometry.topK * 32,
                                            height: 1, depth: 1))
        phase2.endEncoding()
    }

    private static func projection(
        _ view: TensorView,
        rows: Int,
        columns: Int
    ) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: view.buffer,
            scales: view.buffer,
            biases: view.buffer,
            weightsOffset: Int(view.offset),
            scalesOffset: Int(view.scaleOffset),
            biasesOffset: Int(view.biasOffset),
            rows: UInt32(rows),
            cols: UInt32(columns))
    }
}
