import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QwenFullAttentionTests {
    @Test func productionGeometryMatchesQwenContract() {
        let geometry = QwenFullAttentionGeometry.qwen

        #expect(geometry.queryHeads == 16)
        #expect(geometry.keyValueHeads == 2)
        #expect(geometry.headDimension == 256)
        #expect(geometry.rotaryDimension == 64)
        #expect(geometry.ropeTheta == 10_000_000.0)
        #expect(geometry.rotaryPairs == 32)
        #expect(geometry.queryWidth == 4_096)
        #expect(geometry.keyValueWidth == 512)
        #expect(abs(geometry.attentionScale - 0.0625) < 0.000_001)
    }

    @Test func productionHeadIsExplicitlyUntied() throws {
        let context = try MetalContext()
        let attention = try QwenFullAttention(context: context)
        let head = try QwenUntiedLMHead(context: context)

        #expect(attention.geometry == .qwen)
        #expect(head.geometry == .qwen)
    }

    @Test func kvCacheAppendsAndResetsBoundedTokens() throws {
        let context = try MetalContext()
        let cache = try QwenFullAttentionKVCache(
            device: context.device,
            capacity: 2)
        let width = QwenFullAttentionGeometry.qwen.keyValueWidth
        let firstKey = try #require(
            Fp16Buffer.make(context.device,
                            values: [Float](repeating: 1, count: width)))
        let firstValue = try #require(
            Fp16Buffer.make(context.device,
                            values: [Float](repeating: 2, count: width)))
        let secondKey = try #require(
            Fp16Buffer.make(context.device,
                            values: [Float](repeating: 3, count: width)))
        let secondValue = try #require(
            Fp16Buffer.make(context.device,
                            values: [Float](repeating: 4, count: width)))

        for pair in [(firstKey, firstValue), (secondKey, secondValue)] {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            cache.append(commandBuffer: commandBuffer,
                         key: pair.0,
                         value: pair.1)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
        }

        #expect(cache.count == 2)
        let cachedKey = Fp16Buffer.read(cache.key, count: width * 2)
        let cachedValue = Fp16Buffer.read(cache.value, count: width * 2)
        #expect(cachedKey[0] == 1)
        #expect(cachedKey[width] == 3)
        #expect(cachedValue[0] == 2)
        #expect(cachedValue[width] == 4)

        cache.reset()
        #expect(cache.count == 0)
        #expect(Fp16Buffer.read(cache.key, count: width * 2).allSatisfy { $0 == 0 })
        #expect(Fp16Buffer.read(cache.value, count: width * 2).allSatisfy { $0 == 0 })
    }

    @Test func outputGateMatchesSigmoidReference() throws {
        let context = try MetalContext()
        let gate = try QwenAttentionOutputGate(context: context)
        let input = try #require(
            Fp16Buffer.make(context.device, values: [1, 1, 1]))
        let gateInput = try #require(
            Fp16Buffer.make(context.device, values: [-2, 0, 2]))
        let output = try #require(Fp16Buffer.make(context.device, count: 3))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        gate.encode(commandBuffer: commandBuffer,
                    attention: input,
                    gate: gateInput,
                    output: output,
                    count: 3)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = Fp16Buffer.read(output, count: 3)
        let expected = [
            1 / (1 + exp(Float(2))),
            0.5,
            1 / (1 + exp(Float(-2))),
        ]
        for (actualValue, expectedValue) in zip(actual, expected) {
            #expect(abs(actualValue - expectedValue) < 0.002)
        }
    }
}