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

    @Test func kvCacheRestoresSnapshotAfterAdditionalAppend() throws {
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

        for pair in [(firstKey, firstValue)] {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            cache.append(commandBuffer: commandBuffer,
                         key: pair.0,
                         value: pair.1)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let snapshot = cache.snapshot()

        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        cache.append(commandBuffer: commandBuffer,
                     key: secondKey,
                     value: secondValue)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(cache.count == 2)

        cache.restore(snapshot)

        #expect(cache.count == 1)
        #expect(Fp16Buffer.read(cache.key, count: width)[0] == 1)
        #expect(Fp16Buffer.read(cache.value, count: width)[0] == 2)
        let restoredKey = Array(UnsafeBufferPointer(
            start: cache.key.contents().assumingMemoryBound(to: UInt8.self),
            count: snapshot.key.count))
        let restoredValue = Array(UnsafeBufferPointer(
            start: cache.value.contents().assumingMemoryBound(to: UInt8.self),
            count: snapshot.value.count))
        #expect(restoredKey == snapshot.key)
        #expect(restoredValue == snapshot.value)
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

    @Test func queryAndGateSplitWithinEachHead() throws {
        let context = try MetalContext()
        let gate = try QwenAttentionOutputGate(context: context)
        let projection = try #require(Fp16Buffer.make(
            context.device,
            values: [10, 11, 20, 21, 30, 31, 40, 41]))
        let query = try #require(Fp16Buffer.make(context.device, count: 4))
        let gateOutput = try #require(Fp16Buffer.make(context.device, count: 4))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        gate.encodeSplit(commandBuffer: commandBuffer,
                         projection: projection,
                         query: query,
                         gate: gateOutput,
                         headDimension: 2,
                         headCount: 2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(Fp16Buffer.read(query, count: 4) == [10, 11, 30, 31])
        #expect(Fp16Buffer.read(gateOutput, count: 4) == [20, 21, 40, 41])
    }

    @Test func batchedQueryAndGateSplitPreservesTokenRows() throws {
        let context = try MetalContext()
        let gate = try QwenAttentionOutputGate(context: context)
        let projection = try #require(Fp16Buffer.make(
            context.device,
            values: [10, 11, 20, 21, 30, 31, 40, 41,
                     50, 51, 60, 61, 70, 71, 80, 81]))
        let query = try #require(Fp16Buffer.make(context.device, count: 8))
        let gateOutput = try #require(Fp16Buffer.make(context.device, count: 8))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        gate.encodeSplitBatch(commandBuffer: commandBuffer,
                              projection: projection,
                              query: query,
                              gate: gateOutput,
                              tokenCount: 2,
                              headDimension: 2,
                              headCount: 2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(Fp16Buffer.read(query, count: 8) == [10, 11, 30, 31, 50, 51, 70, 71])
        #expect(Fp16Buffer.read(gateOutput, count: 8) == [20, 21, 40, 41, 60, 61, 80, 81])
    }

    @Test func batchedCausalAttentionMatchesSingleRowPasses() throws {
        let context = try MetalContext()
        let qwenAttention = try QwenFullAttention(context: context)
        let scalarAttention = try PrefillAttention(context: context)
        let geometry = QwenFullAttentionGeometry.qwen
        let tokenCount = 2
        let queries = (0..<(tokenCount * geometry.queryWidth)).map {
            Float16(Float(($0 % 19) - 9) / 10)
        }
        let keys = (0..<(tokenCount * geometry.keyValueWidth)).map {
            Float16(Float(($0 % 13) - 6) / 8)
        }
        let values = (0..<(tokenCount * geometry.keyValueWidth)).map {
            Float16(Float(($0 % 11) - 5) / 6)
        }
        let queryBuffer = try #require(Fp16Buffer.make(context.device, halves: queries))
        let keyBuffer = try #require(Fp16Buffer.make(context.device, halves: keys))
        let valueBuffer = try #require(Fp16Buffer.make(context.device, halves: values))
        let batchOutput = try #require(
            Fp16Buffer.make(context.device, count: tokenCount * geometry.queryWidth))
        let scalarOutput = try #require(
            Fp16Buffer.make(context.device, count: tokenCount * geometry.queryWidth))
        let cache = try QwenFullAttentionKVCache(device: context.device, capacity: tokenCount)
        let appendBuffer = try #require(context.queue.makeCommandBuffer())
        cache.appendBatch(commandBuffer: appendBuffer,
                          key: keyBuffer,
                          value: valueBuffer,
                          tokenCount: tokenCount)
        appendBuffer.commit()
        appendBuffer.waitUntilCompleted()
        #expect(appendBuffer.error == nil)

        let batchBuffer = try #require(context.queue.makeCommandBuffer())
        qwenAttention.encodeBatch(commandBuffer: batchBuffer,
                                  query: queryBuffer,
                                  cache: cache,
                                  output: batchOutput,
                                  startPosition: 0,
                                  tokenCount: UInt32(tokenCount))
        batchBuffer.commit()
        batchBuffer.waitUntilCompleted()
        #expect(batchBuffer.error == nil)

        let scalarBuffer = try #require(context.queue.makeCommandBuffer())
        for row in 0..<tokenCount {
            scalarAttention.encodeCausal(
                commandBuffer: scalarBuffer,
                q: queryBuffer,
                qOffset: row * geometry.queryWidth * MemoryLayout<Float16>.stride,
                k: cache.key,
                v: cache.value,
                out: scalarOutput,
                outOffset: row * geometry.queryWidth * MemoryLayout<Float16>.stride,
                params: PrefillAttentionParams(
                    startPosition: UInt32(row),
                    queryCount: 1,
                    headDim: UInt32(geometry.headDimension),
                    numQHeads: UInt32(geometry.queryHeads),
                    numKVHeads: UInt32(geometry.keyValueHeads),
                    kvValidCount: UInt32(row + 1),
                    slidingWindow: 0,
                    kvTokenStrideElements: UInt32(geometry.keyValueWidth),
                    qTokenStrideElements: UInt32(geometry.queryWidth),
                    oTokenStrideElements: UInt32(geometry.queryWidth),
                    scale: geometry.attentionScale))
        }
        scalarBuffer.commit()
        scalarBuffer.waitUntilCompleted()
        #expect(scalarBuffer.error == nil)
        let maxAbs = RelError.maxAbsDiff(
            Fp16Buffer.read(batchOutput, count: tokenCount * geometry.queryWidth),
            Fp16Buffer.read(scalarOutput, count: tokenCount * geometry.queryWidth))
        #expect(maxAbs <= 1e-3, "maxAbs=\(maxAbs)")
    }
}