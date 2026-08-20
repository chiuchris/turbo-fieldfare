import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QwenGatedDeltaNetTests {
    private static let geometry = QwenGatedDeltaNetGeometry(
        keyHeads: 2,
        valueHeads: 4,
        keyHeadDim: 4,
        valueHeadDim: 3,
        convolutionKernel: 4)

    @Test func recurrentStepMatchesReferenceAcrossTokens() throws {
        let context = try MetalContext()
        let kernel = try QwenGatedDeltaNet(context: context)
        let state = try QwenGatedDeltaNetState(
            device: context.device,
            geometry: Self.geometry,
            convolutionChannels: 1)
        var reference = [Float](repeating: 0, count: Self.geometry.recurrentStateElements)
        var rng = SplitMix64(seed: 0x7F5)

        for _ in 0..<5 {
            let query = Self.randomValues(count: Self.geometry.keyDimension, rng: &rng)
            let key = Self.randomValues(count: Self.geometry.keyDimension, rng: &rng)
            let value = Self.randomValues(count: Self.geometry.valueDimension, rng: &rng)
            let decay = (0..<Self.geometry.valueHeads).map { _ in rng.uniform(-1.2, -0.1) }
            let beta = (0..<Self.geometry.valueHeads).map { _ in rng.uniform(0.1, 0.9) }
            let expected = Self.referenceStep(
                query: query, key: key, value: value, decay: decay, beta: beta,
                state: &reference)
            let actual = try Self.runRecurrent(
                kernel: kernel, context: context, state: state,
                query: query, key: key, value: value, decay: decay, beta: beta)
            let maxError = zip(actual, expected)
                .map { abs($0 - Float(Float16($1))) }
                .max() ?? 0
            #expect(maxError < 0.01)
        }
    }

    @Test func causalConvolutionMatchesReferenceAndReset() throws {
        let context = try MetalContext()
        let kernel = try QwenGatedDeltaNet(context: context)
        let state = try QwenGatedDeltaNetState(
            device: context.device,
            geometry: Self.geometry,
            convolutionChannels: 5)
        var rng = SplitMix64(seed: 0xC0A)
        let inputs = (0..<4).map { _ in Self.randomValues(count: 5, rng: &rng) }
        let weights = Self.randomValues(count: 5 * Self.geometry.convolutionKernel, rng: &rng)
        var history = [Float](repeating: 0, count: 5 * (Self.geometry.convolutionKernel - 1))

        for input in inputs {
            let expected = Self.referenceConvolution(
                input: input, weights: weights, history: &history)
            let actual = try Self.runConvolution(
                kernel: kernel, context: context, state: state,
                input: input, weights: weights)
            let maxError = zip(actual, expected)
                .map { abs($0 - Float(Float16($1))) }
                .max() ?? 0
            #expect(maxError < 0.01)
        }

        state.reset()
        var resetHistory = [Float](repeating: 0, count: history.count)
        let expected = Self.referenceConvolution(
            input: inputs[0], weights: weights, history: &resetHistory)
        let actual = try Self.runConvolution(
            kernel: kernel, context: context, state: state,
            input: inputs[0], weights: weights)
        #expect(actual == expected.map { Float(Float16($0)) })
    }

    @Test func managerKeepsBoundedStateAndRestoresSnapshots() throws {
        let context = try MetalContext()
        let manager = try QwenGatedDeltaNetStateManager(
            context: context, layerCount: 30, geometry: .qwen)
        let first = manager.state(layer: 0)
        let last = manager.state(layer: 29)
        #expect(manager.states.count == 30)
        #expect(first.recurrentStateBytes == 32 * 128 * 128 * MemoryLayout<Float>.stride)
        #expect(first.convolutionStateBytes == 8192 * 3 * MemoryLayout<UInt16>.stride)
        #expect(first.recurrentBuffer !== last.recurrentBuffer)

        let snapshot = first.snapshot()
        first.recurrentBuffer.contents().assumingMemoryBound(to: UInt8.self)[0] = 0x7f
        first.restore(snapshot)
        #expect(first.recurrentBuffer.contents().assumingMemoryBound(to: UInt8.self)[0] == 0)

        manager.reset()
        #expect(first.recurrentBuffer.contents().assumingMemoryBound(to: UInt8.self)[0] == 0)
    }

    private static func runRecurrent(
        kernel: QwenGatedDeltaNet,
        context: MetalContext,
        state: QwenGatedDeltaNetState,
        query: [Float],
        key: [Float],
        value: [Float],
        decay: [Float],
        beta: [Float]
    ) throws -> [Float] {
        let queryBuffer = try #require(Fp16Buffer.make(context.device, values: query))
        let keyBuffer = try #require(Fp16Buffer.make(context.device, values: key))
        let valueBuffer = try #require(Fp16Buffer.make(context.device, values: value))
        let decayBuffer = try #require(context.device.makeBuffer(
            bytes: decay, length: decay.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let betaBuffer = try #require(context.device.makeBuffer(
            bytes: beta, length: beta.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let output = try #require(Fp16Buffer.make(
            context.device, count: Self.geometry.valueDimension))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeRecurrent(commandBuffer: commandBuffer,
                               query: queryBuffer,
                               key: keyBuffer,
                               value: valueBuffer,
                               decay: decayBuffer,
                               beta: betaBuffer,
                               output: output,
                               state: state)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Fp16Buffer.read(output, count: Self.geometry.valueDimension)
    }

    private static func runConvolution(
        kernel: QwenGatedDeltaNet,
        context: MetalContext,
        state: QwenGatedDeltaNetState,
        input: [Float],
        weights: [Float]
    ) throws -> [Float] {
        let inputBuffer = try #require(Fp16Buffer.make(context.device, values: input))
        let weightBuffer = try #require(Fp16Buffer.make(context.device, values: weights))
        let output = try #require(Fp16Buffer.make(context.device, count: input.count))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeCausalConvolution(commandBuffer: commandBuffer,
                                       input: inputBuffer,
                                       weights: weightBuffer,
                                       output: output,
                                       state: state)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Fp16Buffer.read(output, count: input.count)
    }

    private static func referenceStep(
        query: [Float],
        key: [Float],
        value: [Float],
        decay: [Float],
        beta: [Float],
        state: inout [Float]
    ) -> [Float] {
        let keyDim = Self.geometry.keyHeadDim
        let valueDim = Self.geometry.valueHeadDim
        let scale = 1 / sqrt(Float(keyDim))
        var output = [Float](repeating: 0, count: Self.geometry.valueDimension)
        for head in 0..<Self.geometry.valueHeads {
            let keyHead = head * Self.geometry.keyHeads / Self.geometry.valueHeads
            let qBase = keyHead * keyDim
            let valueBase = head * valueDim
            let stateBase = head * keyDim * valueDim
            let qRange = qBase..<(qBase + keyDim)
            let kNorm = sqrt(key[qRange].reduce(0) { $0 + $1 * $1 } + 1e-6)
            let qNorm = sqrt(query[qRange].reduce(0) { $0 + $1 * $1 } + 1e-6)
            for index in stateBase..<(stateBase + keyDim * valueDim) {
                state[index] *= exp(decay[head])
            }
            for valueIndex in 0..<valueDim {
                var memory: Float = 0
                for keyIndex in 0..<keyDim {
                    memory += state[stateBase + keyIndex * valueDim + valueIndex]
                        * key[qBase + keyIndex] / kNorm
                }
                let delta = (value[valueBase + valueIndex] - memory) * beta[head]
                for keyIndex in 0..<keyDim {
                    state[stateBase + keyIndex * valueDim + valueIndex] +=
                        key[qBase + keyIndex] / kNorm * delta
                }
            }
            for valueIndex in 0..<valueDim {
                var result: Float = 0
                for keyIndex in 0..<keyDim {
                    result += state[stateBase + keyIndex * valueDim + valueIndex]
                        * query[qBase + keyIndex] / qNorm * scale
                }
                output[valueBase + valueIndex] = result
            }
        }
        return output
    }

    private static func referenceConvolution(
        input: [Float],
        weights: [Float],
        history: inout [Float]
    ) -> [Float] {
        let kernel = Self.geometry.convolutionKernel
        var output = [Float](repeating: 0, count: input.count)
        for channel in input.indices {
            let stateBase = channel * (kernel - 1)
            let weightBase = channel * kernel
            var value: Float = 0
            for tap in 0..<(kernel - 1) {
                value += Float(Float16(weights[weightBase + tap]))
                    * history[stateBase + tap]
            }
            value += Float(Float16(weights[weightBase + kernel - 1]))
                * Float(Float16(input[channel]))
            for tap in 0..<(kernel - 2) {
                history[stateBase + tap] = history[stateBase + tap + 1]
            }
            history[stateBase + kernel - 2] = Float(Float16(input[channel]))
            output[channel] = value / (1 + exp(-value))
        }
        return output.map { Float(Float16($0)) }
    }

    private static func randomValues(
        count: Int,
        rng: inout SplitMix64
    ) -> [Float] {
        (0..<count).map { _ in rng.uniform(-0.8, 0.8) }
    }
}