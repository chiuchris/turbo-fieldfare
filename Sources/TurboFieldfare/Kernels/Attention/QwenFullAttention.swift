import Foundation
import Metal

struct QwenFullAttentionGeometry: Sendable, Equatable {
    let queryHeads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let rotaryDimension: Int
    let ropeTheta: Float

    static let qwen = QwenFullAttentionGeometry(
        queryHeads: 16,
        keyValueHeads: 2,
        headDimension: 256,
        rotaryDimension: 64,
        ropeTheta: 10_000_000.0)

    var queryWidth: Int { queryHeads * headDimension }
    var keyValueWidth: Int { keyValueHeads * headDimension }
    var attentionScale: Float { 1.0 / sqrt(Float(headDimension)) }
    var rotaryPairs: Int { rotaryDimension / 2 }
}

struct QwenFullAttentionKVSnapshot {
    let key: [UInt8]
    let value: [UInt8]
    let count: Int
}

final class QwenFullAttentionKVCache {
    let geometry: QwenFullAttentionGeometry
    let capacity: Int
    let key: MTLBuffer
    let value: MTLBuffer
    private(set) var count = 0

    init(device: MTLDevice,
         capacity: Int,
         geometry: QwenFullAttentionGeometry = .qwen) throws {
        precondition(capacity > 0, "KV capacity must be positive")
        precondition(geometry.queryHeads % geometry.keyValueHeads == 0,
                     "query heads must be divisible by KV heads")
        self.geometry = geometry
        self.capacity = capacity
        let bytes = capacity * geometry.keyValueWidth
            * MemoryLayout<Float16>.stride
        guard let key = device.makeBuffer(length: bytes,
                                          options: .storageModeShared),
              let value = device.makeBuffer(length: bytes,
                                            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.key = key
        self.value = value
        reset()
    }

    var tokenBytes: Int {
        geometry.keyValueWidth * MemoryLayout<Float16>.stride
    }

    func snapshot() -> QwenFullAttentionKVSnapshot {
        let byteCount = count * tokenBytes
        return QwenFullAttentionKVSnapshot(
            key: bytes(from: key, count: byteCount),
            value: bytes(from: value, count: byteCount),
            count: count)
    }

    func restore(_ snapshot: QwenFullAttentionKVSnapshot) {
        precondition(snapshot.count >= 0 && snapshot.count <= capacity,
                     "KV snapshot count exceeds cache capacity")
        let bytes = snapshot.count * tokenBytes
        precondition(snapshot.key.count == bytes && snapshot.value.count == bytes,
                     "KV snapshot bytes do not match snapshot count")
        copy(snapshot.key, to: key)
        copy(snapshot.value, to: value)
        count = snapshot.count
    }

    func append(commandBuffer: MTLCommandBuffer,
                key sourceKey: MTLBuffer,
                keyOffset: Int = 0,
                value sourceValue: MTLBuffer,
                valueOffset: Int = 0) {
        precondition(count < capacity, "KV cache capacity exceeded")
        precondition(keyOffset % 2 == 0 && valueOffset % 2 == 0,
                     "FP16 KV offsets must be two-byte aligned")
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        let destinationOffset = count * tokenBytes
        blit.copy(from: sourceKey,
                  sourceOffset: keyOffset,
                  to: key,
                  destinationOffset: destinationOffset,
                  size: tokenBytes)
        blit.copy(from: sourceValue,
                  sourceOffset: valueOffset,
                  to: value,
                  destinationOffset: destinationOffset,
                  size: tokenBytes)
        blit.endEncoding()
        count += 1
    }

    func reset() {
        memset(key.contents(), 0, key.length)
        memset(value.contents(), 0, value.length)
        count = 0
    }

    private func bytes(from buffer: MTLBuffer, count: Int) -> [UInt8] {
        let pointer = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func copy(_ bytes: [UInt8], to buffer: MTLBuffer) {
        guard !bytes.isEmpty else { return }
        _ = bytes.withUnsafeBytes { source in
            memcpy(buffer.contents(), source.baseAddress!, bytes.count)
        }
    }
}

final class QwenAttentionOutputGate {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline("qwen_attention_output_gate")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                attention: MTLBuffer,
                gate: MTLBuffer,
                output: MTLBuffer,
                count: UInt32) {
        precondition(count > 0, "output gate count must be positive")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(attention, offset: 0, index: 0)
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var countValue = count
        encoder.setBytes(&countValue,
                         length: MemoryLayout<UInt32>.stride,
                         index: 3)
        let width = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

final class QwenFullAttention {
    let geometry: QwenFullAttentionGeometry
    private let attention: Attention
    private let rmsNorm: RMSNorm
    private let rope: RoPE
    private let outputGate: QwenAttentionOutputGate

    init(context: MetalContext,
         geometry: QwenFullAttentionGeometry = .qwen) throws {
        precondition(geometry.rotaryDimension.isMultiple(of: 2),
                     "rotary dimension must be even")
        precondition(geometry.rotaryDimension <= geometry.headDimension,
                     "rotary dimension cannot exceed head dimension")
        self.geometry = geometry
        self.attention = try Attention(context: context)
        self.rmsNorm = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.outputGate = try QwenAttentionOutputGate(context: context)
    }

    func encodeQueryKey(commandBuffer: MTLCommandBuffer,
                        query: MTLBuffer,
                        key: MTLBuffer,
                        queryNorm: MTLBuffer,
                        keyNorm: MTLBuffer,
                        normalizedQuery: MTLBuffer,
                        normalizedKey: MTLBuffer,
                        position: UInt32,
                        epsilon: Float) {
        rmsNorm.encodeBF16WPerHead(
            commandBuffer: commandBuffer,
            x: query,
            weight: queryNorm,
            out: normalizedQuery,
            headDim: UInt32(geometry.headDimension),
            numHeads: geometry.queryHeads,
            eps: epsilon)
        rmsNorm.encodeBF16WPerHead(
            commandBuffer: commandBuffer,
            x: key,
            weight: keyNorm,
            out: normalizedKey,
            headDim: UInt32(geometry.headDimension),
            numHeads: geometry.keyValueHeads,
            eps: epsilon)
        rope.encodeProportionalNeox(
            commandBuffer: commandBuffer,
            data: normalizedQuery,
            position: position,
            headDim: UInt32(geometry.headDimension),
            numHeads: UInt32(geometry.queryHeads),
            rotatedPairs: UInt32(geometry.rotaryPairs),
            theta: geometry.ropeTheta)
        rope.encodeProportionalNeox(
            commandBuffer: commandBuffer,
            data: normalizedKey,
            position: position,
            headDim: UInt32(geometry.headDimension),
            numHeads: UInt32(geometry.keyValueHeads),
            rotatedPairs: UInt32(geometry.rotaryPairs),
            theta: geometry.ropeTheta)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                query: MTLBuffer,
                keyValueCache: QwenFullAttentionKVCache,
                output: MTLBuffer) {
        precondition(keyValueCache.geometry == geometry,
                     "KV cache geometry does not match attention")
        precondition(keyValueCache.count > 0,
                     "full attention requires a populated KV cache")
        attention.encodeFull(
            commandBuffer: commandBuffer,
            q: query,
            k: keyValueCache.key,
            v: keyValueCache.value,
            out: output,
            headDim: UInt32(geometry.headDimension),
            numQHeads: UInt32(geometry.queryHeads),
            numKVHeads: UInt32(geometry.keyValueHeads),
            seqLen: UInt32(keyValueCache.count),
            scale: geometry.attentionScale)
    }

    func encodeOutputGate(commandBuffer: MTLCommandBuffer,
                          attention: MTLBuffer,
                          gate: MTLBuffer,
                          output: MTLBuffer) {
        outputGate.encode(commandBuffer: commandBuffer,
                          attention: attention,
                          gate: gate,
                          output: output,
                          count: UInt32(geometry.queryWidth))
    }
}