import Darwin
import Metal

struct QwenGatedDeltaNetGeometry: Equatable {
    let keyHeads: Int
    let valueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let convolutionKernel: Int

    var keyDimension: Int { keyHeads * keyHeadDim }
    var valueDimension: Int { valueHeads * valueHeadDim }
    var qkvDimension: Int { keyDimension * 2 + valueDimension }
    var recurrentStateElements: Int {
        valueHeads * keyHeadDim * valueHeadDim
    }

    static let qwen = QwenGatedDeltaNetGeometry(
        keyHeads: 16,
        valueHeads: 32,
        keyHeadDim: 128,
        valueHeadDim: 128,
        convolutionKernel: 4)
}

struct QwenGatedDeltaNetSnapshot {
    let recurrentState: [UInt8]
    let convolutionState: [UInt8]
}

final class QwenGatedDeltaNetState {
    let geometry: QwenGatedDeltaNetGeometry
    let convolutionChannels: Int
    let recurrentBuffer: MTLBuffer
    let convolutionBuffer: MTLBuffer

    init(device: MTLDevice,
         geometry: QwenGatedDeltaNetGeometry,
         convolutionChannels: Int) throws {
        precondition(convolutionChannels > 0)
        self.geometry = geometry
        self.convolutionChannels = convolutionChannels
        let recurrentBytes = geometry.recurrentStateElements * MemoryLayout<Float>.stride
        let convolutionElements = convolutionChannels * (geometry.convolutionKernel - 1)
        guard let recurrent = device.makeBuffer(
            length: recurrentBytes, options: .storageModeShared),
            let convolution = device.makeBuffer(
                length: convolutionElements * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.recurrentBuffer = recurrent
        self.convolutionBuffer = convolution
        reset()
    }

    var recurrentStateBytes: Int {
        geometry.recurrentStateElements * MemoryLayout<Float>.stride
    }

    var convolutionStateBytes: Int {
        convolutionChannels * (geometry.convolutionKernel - 1) * MemoryLayout<UInt16>.stride
    }

    func reset() {
        memset(recurrentBuffer.contents(), 0, recurrentStateBytes)
        memset(convolutionBuffer.contents(), 0, convolutionStateBytes)
    }

    func snapshot() -> QwenGatedDeltaNetSnapshot {
        QwenGatedDeltaNetSnapshot(
            recurrentState: bytes(from: recurrentBuffer, count: recurrentStateBytes),
            convolutionState: bytes(from: convolutionBuffer, count: convolutionStateBytes))
    }

    func restore(_ snapshot: QwenGatedDeltaNetSnapshot) {
        precondition(snapshot.recurrentState.count == recurrentStateBytes)
        precondition(snapshot.convolutionState.count == convolutionStateBytes)
        copy(snapshot.recurrentState, to: recurrentBuffer)
        copy(snapshot.convolutionState, to: convolutionBuffer)
    }

    private func bytes(from buffer: MTLBuffer, count: Int) -> [UInt8] {
        let pointer = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func copy(_ bytes: [UInt8], to buffer: MTLBuffer) {
        _ = bytes.withUnsafeBytes { source in
            memcpy(buffer.contents(), source.baseAddress!, bytes.count)
        }
    }
}

final class QwenGatedDeltaNetStateManager {
    private(set) var states: [QwenGatedDeltaNetState]

    init(context: MetalContext,
         layerCount: Int,
         geometry: QwenGatedDeltaNetGeometry = .qwen,
         convolutionChannels: Int? = nil) throws {
        precondition(layerCount > 0)
        let channels = convolutionChannels ?? geometry.qkvDimension
        self.states = try (0..<layerCount).map { _ in
            try QwenGatedDeltaNetState(
                device: context.device,
                geometry: geometry,
                convolutionChannels: channels)
        }
    }

    func state(layer: Int) -> QwenGatedDeltaNetState {
        states[layer]
    }

    func reset() {
        states.forEach { $0.reset() }
    }

    func reset(layer: Int) {
        states[layer].reset()
    }
}

final class QwenGatedDeltaNet {
    private let convolutionPSO: MTLComputePipelineState
    private let recurrentPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.convolutionPSO = try context.pipeline("qwen_gated_delta_causal_conv")
        self.recurrentPSO = try context.pipeline("qwen_gated_delta_recurrent")
    }

    /// Apply Qwen's depthwise causal convolution and SiLU to one token.
    func encodeCausalConvolution(commandBuffer: MTLCommandBuffer,
                                 input: MTLBuffer,
                                 weights: MTLBuffer,
                                 output: MTLBuffer,
                                 state: QwenGatedDeltaNetState) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convolutionPSO)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(state.convolutionBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var channels = UInt32(state.convolutionChannels)
        var kernel = UInt32(state.geometry.convolutionKernel)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
        dispatch(encoder, pipeline: convolutionPSO, count: state.convolutionChannels)
        encoder.endEncoding()
    }

    /// Advance one token of the recurrent gated-delta rule.
    func encodeRecurrent(commandBuffer: MTLCommandBuffer,
                         query: MTLBuffer,
                         key: MTLBuffer,
                         value: MTLBuffer,
                         decay: MTLBuffer,
                         beta: MTLBuffer,
                         output: MTLBuffer,
                         state: QwenGatedDeltaNetState) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(recurrentPSO)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(decay, offset: 0, index: 3)
        encoder.setBuffer(beta, offset: 0, index: 4)
        encoder.setBuffer(state.recurrentBuffer, offset: 0, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)
        var keyHeads = UInt32(state.geometry.keyHeads)
        var valueHeads = UInt32(state.geometry.valueHeads)
        var keyDim = UInt32(state.geometry.keyHeadDim)
        var valueDim = UInt32(state.geometry.valueHeadDim)
        encoder.setBytes(&keyHeads, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&valueHeads, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 10)
        dispatch(encoder, pipeline: recurrentPSO, count: state.geometry.valueHeads)
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                           pipeline: MTLComputePipelineState,
                           count: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}