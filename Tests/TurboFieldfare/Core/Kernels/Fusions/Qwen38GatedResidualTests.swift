import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareFormat
import TurboFieldfareValidationSupport

@Suite struct Qwen38GatedResidualTests {
    @Test func rmsNormUsesSignedCheckpointWeights() throws {
        let context = try MetalContext()
        let kernels = try Qwen38GatedResidual(context: context)
        let inputValues: [Float] = [1, -2, 3, -4]
        let weights: [Float] = [-0.5, 0.25, -1, 1.5]
        let input = try #require(Fp16Buffer.make(context.device, values: inputValues))
        let weightBits = weights.map(Quantization.bf16Bits)
        let weight = try #require(context.device.makeBuffer(
            bytes: weightBits,
            length: weightBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let output = try #require(Fp16Buffer.make(context.device, count: inputValues.count))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernels.encodeRMSNorm(
            commandBuffer: commandBuffer,
            input: input,
            weight: weight,
            output: output,
            tokenCount: 1,
            width: UInt32(inputValues.count),
            epsilon: 1e-6)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let sum = inputValues.reduce(Float(0)) { $0 + $1 * $1 }
        let inverse = 1 / sqrt(sum / Float(inputValues.count) + 1e-6)
        let expected = zip(inputValues, weights).map { value, weight in
            value * inverse * weight
        }
        expectClose(Fp16Buffer.read(output, count: inputValues.count), expected)
    }

    @Test func fourStreamOperationsMatchReferenceAcrossRows() throws {
        let context = try MetalContext()
        let kernels = try Qwen38GatedResidual(context: context)
        let tokenCount = 2
        let streamCount = 4
        let hiddenSize = 3
        let hyperCount = tokenCount * streamCount * hiddenSize
        let inputValues = (0..<hyperCount).map { Float(($0 % 11) - 5) / 4 }
        let zeroCenteredWeights = (0..<(streamCount * hiddenSize)).map {
            Float(($0 % 5) - 2) / 10
        }
        let mixLogits = (0..<hyperCount).map { Float(($0 % 7) - 3) / 3 }
        let blockValues = (0..<(tokenCount * hiddenSize)).map {
            Float($0 + 1) / 5
        }
        let injectionLogits = (0..<(tokenCount * streamCount)).map {
            Float(($0 % 5) - 2) / 2
        }
        let lowRankValues: [Float] = [-2, -1, 0, 1, 2, 3]

        let input = try #require(Fp16Buffer.make(context.device, values: inputValues))
        let weightBits = [Quantization.bf16Bits(99)]
            + zeroCenteredWeights.map(Quantization.bf16Bits)
        let weight = try #require(context.device.makeBuffer(
            bytes: weightBits,
            length: weightBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let normalized = try #require(Fp16Buffer.make(context.device, count: hyperCount))
        let collapsed = try #require(Fp16Buffer.make(
            context.device, count: tokenCount * hiddenSize))
        let mix = try #require(Fp16Buffer.make(context.device, values: mixLogits))
        let mixed = try #require(Fp16Buffer.make(
            context.device, count: tokenCount * hiddenSize))
        let block = try #require(Fp16Buffer.make(context.device, values: blockValues))
        let injectionInput = try #require(Fp16Buffer.make(
            context.device, values: injectionLogits))
        let injectionWeights = try #require(Fp16Buffer.make(
            context.device, count: tokenCount * streamCount))
        let injected = try #require(Fp16Buffer.make(context.device, count: hyperCount))
        let repeated = try #require(Fp16Buffer.make(context.device, count: hyperCount))
        let lowRankInput = try #require(Fp16Buffer.make(
            context.device, values: lowRankValues))
        let lowRankOutput = try #require(Fp16Buffer.make(
            context.device, count: lowRankValues.count))

        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernels.encodeGroupedNorm(
            commandBuffer: commandBuffer,
            input: input,
            weight: weight,
            weightOffset: MemoryLayout<UInt16>.stride,
            output: normalized,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize),
            epsilon: 1e-6)
        kernels.encodeCollapseStreams(
            commandBuffer: commandBuffer,
            input: input,
            output: collapsed,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize))
        kernels.encodeMixStreams(
            commandBuffer: commandBuffer,
            normalized: normalized,
            mixLogits: mix,
            output: mixed,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize))
        kernels.encodeInjectionWeights(
            commandBuffer: commandBuffer,
            logits: injectionInput,
            output: injectionWeights,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount))
        kernels.encodeInjectStreams(
            commandBuffer: commandBuffer,
            hyperInput: input,
            blockOutput: block,
            injectionWeights: injectionWeights,
            output: injected,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize))
        kernels.encodeRepeatStreams(
            commandBuffer: commandBuffer,
            input: block,
            output: repeated,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize))
        kernels.encodeLowRankSiLU(
            commandBuffer: commandBuffer,
            input: lowRankInput,
            output: lowRankOutput,
            count: UInt32(lowRankValues.count),
            streamCount: UInt32(streamCount))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let expectedNorm = groupedNormReference(
            input: inputValues,
            weight: zeroCenteredWeights,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize)
        expectClose(Fp16Buffer.read(normalized, count: hyperCount), expectedNorm)

        let expectedCollapsed = (0..<(tokenCount * hiddenSize)).map { index in
            let tokenBase = (index / hiddenSize) * streamCount * hiddenSize
            let feature = index % hiddenSize
            let sum = (0..<streamCount).reduce(Float(0)) { partial, stream in
                partial + inputValues[tokenBase + stream * hiddenSize + feature]
            }
            return sum / Float(streamCount)
        }
        expectClose(
            Fp16Buffer.read(collapsed, count: expectedCollapsed.count),
            expectedCollapsed)

        let expectedMix = mixReference(
            normalized: expectedNorm,
            logits: mixLogits,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize)
        expectClose(Fp16Buffer.read(mixed, count: expectedMix.count), expectedMix)

        let expectedWeights = injectionLogits.map {
            Float(2 / (1 + exp(-Double($0) / Double(streamCount))))
        }
        expectClose(
            Fp16Buffer.read(injectionWeights, count: expectedWeights.count),
            expectedWeights)

        let expectedInjected = (0..<hyperCount).map { index in
            let feature = index % hiddenSize
            let streamRow = index / hiddenSize
            let token = streamRow / streamCount
            return inputValues[index]
                + blockValues[token * hiddenSize + feature] * expectedWeights[streamRow]
        }
        expectClose(Fp16Buffer.read(injected, count: hyperCount), expectedInjected)

        let expectedRepeated = (0..<hyperCount).map { index in
            let feature = index % hiddenSize
            let token = index / (streamCount * hiddenSize)
            return blockValues[token * hiddenSize + feature]
        }
        expectClose(Fp16Buffer.read(repeated, count: hyperCount), expectedRepeated)

        let expectedLowRank = lowRankValues.map { value in
            let scaled = value / Float(streamCount)
            return scaled / (1 + exp(-scaled))
        }
        expectClose(
            Fp16Buffer.read(lowRankOutput, count: lowRankValues.count),
            expectedLowRank)
    }

    @Test func composedHyperConnectionMatchesReferenceAndOptionalInjection() throws {
        let context = try MetalContext()
        let geometry = Qwen38HyperConnectionGeometry(
            streamCount: 2,
            hiddenSize: 32,
            lowRankSize: 32)
        let pipeline = try Qwen38HyperConnection(
            context: context,
            geometry: geometry)
        let tokenCount = 2
        let hyperWidth = Int(geometry.hyperWidth)
        let hiddenSize = Int(geometry.hiddenSize)
        let streamCount = Int(geometry.streamCount)
        let lowRankSize = Int(geometry.lowRankSize)
        let hyperInput = (0..<(tokenCount * hyperWidth)).map {
            Float(($0 * 7) % 19 - 9) / 8
        }
        let normWeights = (0..<hyperWidth).map { Float(($0 % 7) - 3) / 16 }
        let normBits = normWeights.map(Quantization.bf16Bits)
        let norm = try #require(context.device.makeBuffer(
            bytes: normBits,
            length: normBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let down = try makeProjection(
            device: context.device,
            outputWidth: lowRankSize,
            inputWidth: hyperWidth,
            seed: 1)
        let up = try makeProjection(
            device: context.device,
            outputWidth: hyperWidth,
            inputWidth: lowRankSize,
            seed: 3)
        let inject = try makeProjection(
            device: context.device,
            outputWidth: streamCount,
            inputWidth: hyperWidth,
            seed: 5)
        let inputBuffer = try #require(Fp16Buffer.make(
            context.device,
            values: hyperInput))
        let branchOutput = (0..<(tokenCount * hiddenSize)).map {
            Float(($0 * 5) % 13 - 6) / 7
        }
        let branchBuffer = try #require(Fp16Buffer.make(
            context.device,
            values: branchOutput))
        let mixed = try #require(Fp16Buffer.make(
            context.device,
            count: tokenCount * hiddenSize))
        let injected = try #require(Fp16Buffer.make(
            context.device,
            count: tokenCount * hyperWidth))
        let scratch = try makeHyperConnectionScratch(
            device: context.device,
            tokenCount: tokenCount,
            geometry: geometry)
        let weights = Qwen38HyperConnectionWeights(
            norm: norm,
            inputMixDown: down.projection,
            inputMixUp: up.projection,
            blockInject: inject.projection)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        pipeline.encodePrepare(
            commandBuffer: commandBuffer,
            hyperInput: inputBuffer,
            weights: weights,
            scratch: scratch,
            mixedInput: mixed,
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6)
        pipeline.encodeInject(
            commandBuffer: commandBuffer,
            hyperInput: inputBuffer,
            branchOutput: branchBuffer,
            injectionWeights: scratch.injectionWeights,
            output: injected,
            tokenCount: UInt32(tokenCount))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let normalized = groupedNormReference(
            input: hyperInput,
            weight: normWeights,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize,
            oneCentered: false)
        let lowRank = projectionReference(
            input: normalized,
            matrix: down.matrix,
            tokenCount: tokenCount,
            outputWidth: lowRankSize,
            inputWidth: hyperWidth).map {
                let scaled = $0 / Float(streamCount)
                return scaled / (1 + exp(-scaled))
            }
        let mixLogits = projectionReference(
            input: lowRank,
            matrix: up.matrix,
            tokenCount: tokenCount,
            outputWidth: hyperWidth,
            inputWidth: lowRankSize)
        let expectedMixed = mixReference(
            normalized: normalized,
            logits: mixLogits,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize)
        let injectionLogits = projectionReference(
            input: normalized,
            matrix: inject.matrix,
            tokenCount: tokenCount,
            outputWidth: streamCount,
            inputWidth: hyperWidth)
        let expectedWeights = injectionLogits.map {
            Float(2 / (1 + exp(-Double($0) / Double(streamCount))))
        }
        let expectedInjected = (0..<(tokenCount * hyperWidth)).map { index in
            let streamRow = index / hiddenSize
            let token = streamRow / streamCount
            let feature = index % hiddenSize
            return hyperInput[index]
                + branchOutput[token * hiddenSize + feature] * expectedWeights[streamRow]
        }
        expectClose(Fp16Buffer.read(mixed, count: expectedMixed.count), expectedMixed)
        expectClose(
            Fp16Buffer.read(scratch.injectionWeights, count: expectedWeights.count),
            expectedWeights)
        expectClose(
            Fp16Buffer.read(injected, count: expectedInjected.count),
            expectedInjected)

        let finalScratch = try makeHyperConnectionScratch(
            device: context.device,
            tokenCount: tokenCount,
            geometry: geometry,
            injectionSentinel: 7)
        let finalMixed = try #require(Fp16Buffer.make(
            context.device,
            count: tokenCount * hiddenSize))
        let finalCommandBuffer = try #require(context.queue.makeCommandBuffer())
        pipeline.encodePrepare(
            commandBuffer: finalCommandBuffer,
            hyperInput: inputBuffer,
            weights: Qwen38HyperConnectionWeights(
                norm: norm,
                inputMixDown: down.projection,
                inputMixUp: up.projection,
                blockInject: nil),
            scratch: finalScratch,
            mixedInput: finalMixed,
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6)
        finalCommandBuffer.commit()
        finalCommandBuffer.waitUntilCompleted()
        #expect(finalCommandBuffer.error == nil)
        expectClose(
            Fp16Buffer.read(finalMixed, count: expectedMixed.count),
            expectedMixed)
        #expect(Fp16Buffer.read(
            finalScratch.injectionWeights,
            count: tokenCount * streamCount).allSatisfy { $0 == 7 })
    }

    @Test func tensorViewsValidateExactHyperConnectionSchema() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = Qwen38HyperConnectionGeometry(
            streamCount: 2,
            hiddenSize: 32,
            lowRankSize: 32)
        let normBuffer = try #require(device.makeBuffer(
            length: Int(geometry.hyperWidth) * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let norm = TensorView(
            buffer: normBuffer,
            offset: 0,
            length: UInt64(normBuffer.length),
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (geometry.hyperWidth, 0, 0, 0),
            dtype: GTurboFormatV1.DType.bf16.rawValue)
        let down = try makeProjectionView(
            device: device,
            rows: geometry.lowRankSize,
            columns: geometry.hyperWidth)
        let up = try makeProjectionView(
            device: device,
            rows: geometry.hyperWidth,
            columns: geometry.lowRankSize)
        let inject = try makeProjectionView(
            device: device,
            rows: geometry.streamCount,
            columns: geometry.hyperWidth)

        let weights = try Qwen38HyperConnectionWeights(
            norm: norm,
            inputMixDown: down,
            inputMixUp: up,
            blockInject: inject,
            geometry: geometry)
        #expect(weights.norm === normBuffer)
        #expect(weights.blockInject != nil)

        let malformedUp = TensorView(
            buffer: up.buffer,
            offset: up.offset,
            length: up.length,
            scaleOffset: up.scaleOffset,
            scaleLength: up.scaleLength,
            biasOffset: up.biasOffset,
            biasLength: up.biasLength,
            shape: (geometry.lowRankSize, geometry.hyperWidth, 0, 0),
            dtype: up.dtype)
        #expect(throws: ModelError.self) {
            _ = try Qwen38HyperConnectionWeights(
                norm: norm,
                inputMixDown: down,
                inputMixUp: malformedUp,
                blockInject: inject,
                geometry: geometry)
        }
    }
}

private struct TestProjection {
    let projection: Qwen38PLEQuantizedProjection
    let matrix: [Float]
}

private func makeProjectionView(
    device: MTLDevice,
    rows: UInt32,
    columns: UInt32
) throws -> TensorView {
    let weightBytes = UInt64(rows) * UInt64(columns) / 2
    let auxiliaryBytes = UInt64(rows) * UInt64(columns / 32)
        * UInt64(MemoryLayout<UInt16>.stride)
    let scaleOffset = weightBytes
    let biasOffset = scaleOffset + auxiliaryBytes
    let buffer = try #require(device.makeBuffer(
        length: Int(biasOffset + auxiliaryBytes),
        options: .storageModeShared))
    return TensorView(
        buffer: buffer,
        offset: 0,
        length: weightBytes,
        scaleOffset: scaleOffset,
        scaleLength: auxiliaryBytes,
        biasOffset: biasOffset,
        biasLength: auxiliaryBytes,
        shape: (rows, columns, 0, 0),
        dtype: GTurboFormatV1.DType.u32.rawValue)
}

private func makeProjection(device: MTLDevice,
                            outputWidth: Int,
                            inputWidth: Int,
                            seed: Int) throws -> TestProjection {
    let groupSize = 32
    let groupCount = inputWidth / groupSize
    var weights: [UInt8] = []
    var scales: [UInt16] = []
    var biases: [UInt16] = []
    var matrix = [Float](repeating: 0, count: outputWidth * inputWidth)
    for row in 0..<outputWidth {
        var rowWeights: [UInt8] = []
        for group in 0..<groupCount {
            let scale = Float((row + seed) % 5 + 1) / 32
            let bias = Float((group + seed) % 3 - 1) / 64
            scales.append(Quantization.bf16Bits(scale))
            biases.append(Quantization.bf16Bits(bias))
            for lane in 0..<(groupSize / 2) {
                let low = UInt8((row + group + lane + seed) % 7)
                let high = UInt8((row * 2 + group + lane + seed) % 7)
                rowWeights.append(low | (high << 4))
            }
        }
        weights.append(contentsOf: rowWeights)
        for feature in 0..<inputWidth {
            let packed = rowWeights[feature / 2]
            let quantized = feature.isMultiple(of: 2) ? packed & 0x0F : packed >> 4
            let metadata = row * groupCount + feature / groupSize
            matrix[row * inputWidth + feature] = Float(quantized)
                * Quantization.bf16ToFloat(scales[metadata])
                + Quantization.bf16ToFloat(biases[metadata])
        }
    }
    let weightBuffer = try #require(device.makeBuffer(
        bytes: weights,
        length: weights.count,
        options: .storageModeShared))
    let scaleBuffer = try #require(device.makeBuffer(
        bytes: scales,
        length: scales.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared))
    let biasBuffer = try #require(device.makeBuffer(
        bytes: biases,
        length: biases.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared))
    return TestProjection(
        projection: Qwen38PLEQuantizedProjection(
            weights: weightBuffer,
            scales: scaleBuffer,
            biases: biasBuffer),
        matrix: matrix)
}

private func makeHyperConnectionScratch(
    device: MTLDevice,
    tokenCount: Int,
    geometry: Qwen38HyperConnectionGeometry,
    injectionSentinel: Float? = nil
) throws -> Qwen38HyperConnectionScratch {
    let hyperCount = tokenCount * Int(geometry.hyperWidth)
    let lowRankCount = tokenCount * Int(geometry.lowRankSize)
    let streamCount = tokenCount * Int(geometry.streamCount)
    let injectionValues = [Float](
        repeating: injectionSentinel ?? 0,
        count: streamCount)
    return try Qwen38HyperConnectionScratch(
        normalized: #require(Fp16Buffer.make(device, count: hyperCount)),
        lowRank: #require(Fp16Buffer.make(device, count: lowRankCount)),
        activatedLowRank: #require(Fp16Buffer.make(device, count: lowRankCount)),
        mixLogits: #require(Fp16Buffer.make(device, count: hyperCount)),
        injectionLogits: #require(Fp16Buffer.make(device, values: injectionValues)),
        injectionWeights: #require(Fp16Buffer.make(device, values: injectionValues)))
}

private func projectionReference(input: [Float],
                                 matrix: [Float],
                                 tokenCount: Int,
                                 outputWidth: Int,
                                 inputWidth: Int) -> [Float] {
    (0..<(tokenCount * outputWidth)).map { index in
        let token = index / outputWidth
        let row = index % outputWidth
        return (0..<inputWidth).reduce(Float(0)) { sum, feature in
            sum + input[token * inputWidth + feature]
                * matrix[row * inputWidth + feature]
        }
    }
}

private func groupedNormReference(input: [Float],
                                  weight: [Float],
                                  tokenCount: Int,
                                  streamCount: Int,
                                  hiddenSize: Int,
                                  oneCentered: Bool = true) -> [Float] {
    var output = [Float](repeating: 0, count: input.count)
    for token in 0..<tokenCount {
        for stream in 0..<streamCount {
            let base = (token * streamCount + stream) * hiddenSize
            let sum = (0..<hiddenSize).reduce(Float(0)) { partial, feature in
                let value = input[base + feature]
                return partial + value * value
            }
            let inverse = 1 / sqrt(sum / Float(hiddenSize) + 1e-6)
            for feature in 0..<hiddenSize {
                output[base + feature] = input[base + feature] * inverse
                    * (oneCentered ? 1 + weight[stream * hiddenSize + feature]
                       : weight[stream * hiddenSize + feature])
            }
        }
    }
    return output
}

private func mixReference(normalized: [Float],
                          logits: [Float],
                          tokenCount: Int,
                          streamCount: Int,
                          hiddenSize: Int) -> [Float] {
    var output = [Float](repeating: 0, count: tokenCount * hiddenSize)
    for token in 0..<tokenCount {
        for feature in 0..<hiddenSize {
            var sum: Float = 0
            for stream in 0..<streamCount {
                let index = (token * streamCount + stream) * hiddenSize + feature
                let gate = Float(1 / (1 + exp(-Double(logits[index]))))
                sum += gate * normalized[index]
            }
            output[token * hiddenSize + feature] = sum / Float(streamCount)
        }
    }
    return output
}

private func expectClose(_ actual: [Float],
                         _ expected: [Float],
                         tolerance: Float = 0.015) {
    #expect(actual.count == expected.count)
    for index in actual.indices {
        #expect(abs(actual[index] - Float(Float16(expected[index]))) < tolerance)
    }
}
