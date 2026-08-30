import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct Qwen38PLETests {
    @Test func addressingMatchesQwen4ExpReferenceAndEOSBoundaries() {
        let addressing = Qwen38PLEAddressing()
        #expect(addressing.layerMultipliers == [
            23_703_573_157_769,
            20_109_073_645_365,
            8_052_911_324_071,
        ])
        #expect(addressing.headVocabSizes == [
            20_000_003, 20_000_023, 20_000_033, 20_000_047,
            20_000_059, 20_000_063, 20_000_069, 20_000_077,
            20_000_081, 20_000_093, 20_000_107, 20_000_147,
            20_000_153, 20_000_159, 20_000_161, 20_000_171,
        ])
        #expect(addressing.paddedVocabSize == 320_001_536)

        let addresses = addressing.addresses(tokens: [1, 2, 3, 248_044, 4, 5])
        #expect(addresses[0] == [
            16_121_432, 28_938_500, 59_087_997, 73_487_090,
            81_148_277, 104_500_129, 120_276_032, 149_373_875,
            176_283_436, 184_305_849, 216_528_839, 231_080_079,
            257_961_536, 266_068_568, 289_043_455, 305_959_965,
        ])
        #expect(addresses[2] == [
            14_605_717, 24_410_875, 49_313_567, 72_177_428,
            86_060_820, 104_022_010, 130_963_819, 146_886_202,
            161_523_077, 195_939_126, 219_424_565, 220_811_782,
            248_020_141, 275_228_530, 284_297_999, 309_645_223,
        ])
        #expect(addresses[4] == [
            16_786_187, 37_399_507, 51_447_157, 75_303_773,
            99_642_929, 108_554_057, 122_668_943, 142_885_423,
            178_075_680, 189_995_935, 213_432_942, 234_309_713,
            243_139_806, 273_195_768, 283_486_804, 316_984_683,
        ])

        var context: [Int64] = []
        let first = addressing.addresses(tokens: [1, 2, 3], context: &context)
        let second = addressing.addresses(tokens: [248_044, 4, 5], context: &context)
        #expect(first + second == addresses)
        #expect(context == [4, 5])
    }

    @Test func modelMetadataAddressingUsesResidentValues() throws {
        let context = try MetalContext()
        let multipliers = try Self.metadataView(
            context: context,
            values: [3, 5, 7])
        let headVocabSizes = try Self.metadataView(
            context: context,
            values: Array(repeating: 11, count: 16))
        let headOffsets = try Self.metadataView(
            context: context,
            values: (0..<16).map { Int64($0 * 11) })

        let addressing = try Qwen38PLEAddressing(
            layerMultipliers: multipliers,
            headOffsets: headOffsets,
            headVocabSizes: headVocabSizes,
            ngramSize: 3,
            headsPerNgram: 8,
            unigramVocabSize: 32,
            ngramVocabSizeBase: 11,
            vocabDivisor: 16)

        #expect(addressing.layerMultipliers == [3, 5, 7])
        #expect(addressing.headVocabSizes == Array(repeating: 11, count: 16))
        #expect(addressing.headOffsets == (0..<16).map { Int64($0 * 11) })
        #expect(addressing.paddedVocabSize == 176)
    }

    @Test func rejectsWrongPLEMetadataDtype() throws {
        let context = try MetalContext()
        let multipliers = try Self.metadataView(
            context: context,
            values: [3, 5, 7],
            dtype: 0)
        let headVocabSizes = try Self.metadataView(
            context: context,
            values: Array(repeating: 11, count: 16))
        let headOffsets = try Self.metadataView(
            context: context,
            values: (0..<16).map { Int64($0 * 11) })

        #expect(throws: ModelError.self) {
            try Qwen38PLEAddressing(
                layerMultipliers: multipliers,
                headOffsets: headOffsets,
                headVocabSizes: headVocabSizes,
                ngramSize: 3,
                headsPerNgram: 8,
                unigramVocabSize: 32,
                ngramVocabSizeBase: 11,
                vocabDivisor: 16)
        }
    }

    @Test func rejectsNonContiguousPLEHeadOffsets() throws {
        let context = try MetalContext()
        let multipliers = try Self.metadataView(
            context: context,
            values: [3, 5, 7])
        let headVocabSizes = try Self.metadataView(
            context: context,
            values: Array(repeating: 11, count: 16))
        var offsets = (0..<16).map { Int64($0 * 11) }
        offsets[1] += 1
        let headOffsets = try Self.metadataView(context: context, values: offsets)

        #expect(throws: ModelError.self) {
            try Qwen38PLEAddressing(
                layerMultipliers: multipliers,
                headOffsets: headOffsets,
                headVocabSizes: headVocabSizes,
                ngramSize: 3,
                headsPerNgram: 8,
                unigramVocabSize: 32,
                ngramVocabSizeBase: 11,
                vocabDivisor: 16)
        }
    }

    private static func metadataView(
        context: MetalContext,
        values: [Int64],
        dtype: UInt8 = 4
    ) throws -> TensorView {
        let buffer = try #require(context.device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Int64>.stride,
            options: .storageModeShared))
        return TensorView(
            buffer: buffer,
            offset: 0,
            length: UInt64(values.count * MemoryLayout<Int64>.stride),
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (UInt32(values.count), 0, 0, 0),
            dtype: dtype)
    }

    @Test func group32ProjectionMatchesAffineReference() throws {
        let context = try MetalContext()
        let projection = try Qwen38PLEProjection(context: context)
        let tokenCount = 2
        let outputWidth = 5
        let inputWidth = 64
        let groupSize = 32
        let groupCount = inputWidth / groupSize
        var weights: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in 0..<outputWidth {
            for group in 0..<groupCount {
                for lane in 0..<(groupSize / 2) {
                    let low = UInt8((row * 3 + group + lane) % 15)
                    weights.append(low | ((low + 1) << 4))
                }
            }
            for group in 0..<groupCount {
                scales.append(Quantization.bf16Bits(
                    Float(row + 1) / 16 + Float(group + 1) / 4))
                biases.append(Quantization.bf16Bits(Float(row - group) / 8))
            }
        }
        let input = (0..<(tokenCount * inputWidth)).map { index in
            let token = index / inputWidth
            let feature = index % inputWidth
            return Float((token + 1) * (feature % 7 - 3)) / 8
        }
        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: weights,
            length: weights.count,
            options: .storageModeShared))
        let scaleBuffer = try #require(context.device.makeBuffer(
            bytes: scales,
            length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let biasBuffer = try #require(context.device.makeBuffer(
            bytes: biases,
            length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let inputBuffer = try #require(Fp16Buffer.make(context.device, values: input))
        let outputBuffer = try #require(Fp16Buffer.make(
            context.device, count: tokenCount * outputWidth))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        projection.encode(
            commandBuffer: commandBuffer,
            weights: weightBuffer,
            scales: scaleBuffer,
            biases: biasBuffer,
            input: inputBuffer,
            output: outputBuffer,
            tokenCount: UInt32(tokenCount),
            outputWidth: UInt32(outputWidth),
            inputWidth: UInt32(inputWidth))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        var expected: [Float] = []
        for token in 0..<tokenCount {
            for row in 0..<outputWidth {
                var sum: Float = 0
                for feature in 0..<inputWidth {
                    let packed = weights[row * (inputWidth / 2) + feature / 2]
                    let quantized = feature.isMultiple(of: 2)
                        ? packed & 0x0F
                        : packed >> 4
                    let metadata = row * groupCount + feature / groupSize
                    let scale = Quantization.bf16ToFloat(scales[metadata])
                    let bias = Quantization.bf16ToFloat(biases[metadata])
                    sum += (Float(quantized) * scale + bias)
                        * input[token * inputWidth + feature]
                }
                expected.append(sum)
            }
        }
        expectClose(
            Fp16Buffer.read(outputBuffer, count: expected.count),
            expected,
            tolerance: 0.15)
    }

    @Test func pipelineMatchesReferenceAcrossDecodeCalls() throws {
        let context = try MetalContext()
        let pipeline = try Qwen38PLEPipeline(context: context)
        let tokenCount = 2
        let streamCount = 2
        let hiddenSize = 4
        let embeddingSize = 32
        let channels = streamCount * hiddenSize
        let epsilon: Float = 1e-6
        let embeddings = (0..<(tokenCount * embeddingSize)).map {
            Float($0 % 9 - 4) / 8
        }
        let hiddenStates = (0..<(tokenCount * channels)).map {
            Float(($0 * 5) % 13 - 6) / 7
        }

        func makeProjection(rowBiases: [Float]) throws
            -> (Qwen38PLEQuantizedProjection, [Float]) {
            let packed = [UInt8](
                repeating: 0x21, count: rowBiases.count * embeddingSize / 2)
            let scaleBits = [UInt16](
                repeating: Quantization.bf16Bits(0), count: rowBiases.count)
            let biasBits = rowBiases.map(Quantization.bf16Bits)
            let weights = try #require(context.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared))
            let scales = try #require(context.device.makeBuffer(
                bytes: scaleBits,
                length: scaleBits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            let biases = try #require(context.device.makeBuffer(
                bytes: biasBits,
                length: biasBits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            return (
                Qwen38PLEQuantizedProjection(
                    weights: weights, scales: scales, biases: biases),
                biasBits.map(Quantization.bf16ToFloat))
        }

        let (keyProjection, keyBiases) = try makeProjection(
            rowBiases: (0..<channels).map { Float($0 % 5 + 1) / 32 })
        let (valueProjection, valueBiases) = try makeProjection(
            rowBiases: (0..<hiddenSize).map { Float($0 + 2) / 24 })
        let keyNormValues = (0..<channels).map { Float($0 % 3 - 1) / 16 }
        let queryNormValues = (0..<channels).map { Float($0 % 4 - 2) / 20 }
        let convolutionNormValues = (0..<channels).map { Float($0 % 5 - 2) / 24 }
        let convolutionValues = (0..<(channels * 4)).map {
            Float(($0 * 3) % 7 - 3) / 16
        }
        func makeBF16Buffer(_ values: [Float]) throws -> (MTLBuffer, [Float]) {
            let bits = values.map(Quantization.bf16Bits)
            let buffer = try #require(context.device.makeBuffer(
                bytes: bits,
                length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            return (buffer, bits.map(Quantization.bf16ToFloat))
        }
        let (keyNorm, keyNormReference) = try makeBF16Buffer(keyNormValues)
        let (queryNorm, queryNormReference) = try makeBF16Buffer(queryNormValues)
        let (convolutionNorm, convolutionNormReference) = try makeBF16Buffer(
            convolutionNormValues)
        let (convolutionWeights, convolutionReference) = try makeBF16Buffer(
            convolutionValues)
        let weights = Qwen38PLEWeights(
            keyProjection: keyProjection,
            valueProjection: valueProjection,
            keyNorm: keyNorm,
            queryNorm: queryNorm,
            convolutionNorm: convolutionNorm,
            convolution: convolutionWeights)
        let state = try Qwen38PLEConvolutionState(
            device: context.device, channels: channels)

        func groupedNorm(_ input: [Float], weights: [Float]) -> [Float] {
            var result: [Float] = []
            for stream in 0..<streamCount {
                let start = stream * hiddenSize
                let values = Array(input[start..<(start + hiddenSize)])
                let meanSquare = values.reduce(Float(0)) { $0 + $1 * $1 }
                    / Float(hiddenSize)
                let inverse = 1 / sqrt(meanSquare + epsilon)
                for feature in 0..<hiddenSize {
                    result.append(Float(Float16(
                        values[feature] * inverse * (1 + weights[start + feature]))))
                }
            }
            return result
        }

        var normalizedRows: [[Float]] = []
        var gatedRows: [[Float]] = []
        for token in 0..<tokenCount {
            let embedding = Array(embeddings[
                (token * embeddingSize)..<((token + 1) * embeddingSize)])
            let embeddingSum = embedding.reduce(Float(0), +)
            let key = keyBiases.map { Float(Float16($0 * embeddingSum)) }
            let value = valueBiases.map { Float(Float16($0 * embeddingSum)) }
            let query = Array(hiddenStates[(token * channels)..<((token + 1) * channels)])
            let normalizedKey = groupedNorm(key, weights: keyNormReference)
            let normalizedQuery = groupedNorm(query, weights: queryNormReference)
            var gated: [Float] = []
            for stream in 0..<streamCount {
                let start = stream * hiddenSize
                let dot = (0..<hiddenSize).reduce(Float(0)) {
                    $0 + normalizedKey[start + $1] * normalizedQuery[start + $1]
                }
                let gate = dot / sqrt(Float(hiddenSize))
                let transformed = gate == 0
                    ? 0
                    : copysign(sqrt(max(abs(gate), 1e-6)), gate)
                let weight = 1 / (1 + exp(-transformed))
                gated += value.map { Float(Float16(weight * $0)) }
            }
            gatedRows.append(gated)
            normalizedRows.append(groupedNorm(
                gated, weights: convolutionNormReference))
        }
        var history = [Float](repeating: 0, count: channels * 9)
        let convolution = referenceConvolution(
            rows: normalizedRows,
            weights: convolutionReference,
            channels: channels,
            history: &history)
        let expected = zip(gatedRows.flatMap { $0 }, convolution).map {
            Float(Float16($0 + Float(Float16($1))))
        }

        func runToken(_ token: Int) throws -> [Float] {
            let embedding = try #require(Fp16Buffer.make(
                context.device,
                values: Array(embeddings[
                    (token * embeddingSize)..<((token + 1) * embeddingSize)])))
            let hidden = try #require(Fp16Buffer.make(
                context.device,
                values: Array(hiddenStates[
                    (token * channels)..<((token + 1) * channels)])))
            func scratchBuffer(_ count: Int) throws -> MTLBuffer {
                try #require(Fp16Buffer.make(context.device, count: count))
            }
            let scratch = Qwen38PLEScratch(
                projectedKey: try scratchBuffer(channels),
                value: try scratchBuffer(hiddenSize),
                normalizedKey: try scratchBuffer(channels),
                normalizedQuery: try scratchBuffer(channels),
                gatedValue: try scratchBuffer(channels),
                normalizedGatedValue: try scratchBuffer(channels),
                convolution: try scratchBuffer(channels))
            let output = try scratchBuffer(channels)
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            pipeline.encode(
                commandBuffer: commandBuffer,
                embedding: embedding,
                hiddenStates: hidden,
                weights: weights,
                scratch: scratch,
                state: state,
                output: output,
                tokenCount: 1,
                streamCount: UInt32(streamCount),
                hiddenSize: UInt32(hiddenSize),
                embeddingSize: UInt32(embeddingSize),
                epsilon: epsilon)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            return Fp16Buffer.read(output, count: channels)
        }

        let actual = try runToken(0) + runToken(1)
        expectClose(actual, expected, tolerance: 0.03)

        let batchState = try Qwen38PLEConvolutionState(
            device: context.device, channels: channels)
        let batchEmbedding = try #require(Fp16Buffer.make(
            context.device, values: embeddings))
        let batchHidden = try #require(Fp16Buffer.make(
            context.device, values: hiddenStates))
        func batchScratchBuffer(_ count: Int) throws -> MTLBuffer {
            try #require(Fp16Buffer.make(context.device, count: count))
        }
        let batchScratch = Qwen38PLEScratch(
            projectedKey: try batchScratchBuffer(tokenCount * channels),
            value: try batchScratchBuffer(tokenCount * hiddenSize),
            normalizedKey: try batchScratchBuffer(tokenCount * channels),
            normalizedQuery: try batchScratchBuffer(tokenCount * channels),
            gatedValue: try batchScratchBuffer(tokenCount * channels),
            normalizedGatedValue: try batchScratchBuffer(tokenCount * channels),
            convolution: try batchScratchBuffer(tokenCount * channels))
        let batchOutput = try batchScratchBuffer(tokenCount * channels)
        let batchCommandBuffer = try #require(context.queue.makeCommandBuffer())
        pipeline.encode(
            commandBuffer: batchCommandBuffer,
            embedding: batchEmbedding,
            hiddenStates: batchHidden,
            weights: weights,
            scratch: batchScratch,
            state: batchState,
            output: batchOutput,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize),
            embeddingSize: UInt32(embeddingSize),
            epsilon: epsilon)
        batchCommandBuffer.commit()
        batchCommandBuffer.waitUntilCompleted()
        #expect(batchCommandBuffer.error == nil)
        expectClose(
            Fp16Buffer.read(batchOutput, count: tokenCount * channels),
            actual,
            tolerance: 0.03)
    }

    @Test func streamGateAndResidualMergeMatchReference() throws {
        let context = try MetalContext()
        let gate = try Qwen38PLEGate(context: context)
        let tokenCount = 2
        let streamCount = 3
        let hiddenSize = 3
        let key: [Float] = [
            1, 0, 0, 1, 0, 0, 1, 0, 0,
            1, 2, 3, 1, 2, 3, 1, 2, 3,
        ]
        let query: [Float] = [
            1, 0, 0, -1, 0, 0, 0, 1, 0,
            3, -2, 1, -3, 1, -1, 1, 1, -1,
        ]
        let value: [Float] = [2, -4, 1, -3, 5, 2]
        let convolution = (0..<(tokenCount * streamCount * hiddenSize)).map {
            Float($0 - 8) / 16
        }
        let keyBuffer = try #require(Fp16Buffer.make(context.device, values: key))
        let queryBuffer = try #require(Fp16Buffer.make(context.device, values: query))
        let valueBuffer = try #require(Fp16Buffer.make(context.device, values: value))
        let convolutionBuffer = try #require(Fp16Buffer.make(
            context.device, values: convolution))
        let gatedBuffer = try #require(Fp16Buffer.make(
            context.device, count: convolution.count))
        let outputBuffer = try #require(Fp16Buffer.make(
            context.device, count: convolution.count))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        gate.encodeGate(
            commandBuffer: commandBuffer,
            normalizedKey: keyBuffer,
            normalizedQuery: queryBuffer,
            value: valueBuffer,
            output: gatedBuffer,
            tokenCount: UInt32(tokenCount),
            streamCount: UInt32(streamCount),
            hiddenSize: UInt32(hiddenSize))
        gate.encodeResidualMerge(
            commandBuffer: commandBuffer,
            gatedValue: gatedBuffer,
            convolution: convolutionBuffer,
            output: outputBuffer,
            count: UInt32(convolution.count))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        var expectedGated: [Float] = []
        for token in 0..<tokenCount {
            for stream in 0..<streamCount {
                let base = (token * streamCount + stream) * hiddenSize
                let dot = (0..<hiddenSize).reduce(Float(0)) {
                    $0 + key[base + $1] * query[base + $1]
                }
                let gateValue = dot / sqrt(Float(hiddenSize))
                let transformed = gateValue == 0
                    ? 0
                    : copysign(sqrt(max(abs(gateValue), 1e-6)), gateValue)
                let weight = 1 / (1 + exp(-transformed))
                expectedGated += value[(token * hiddenSize)..<((token + 1) * hiddenSize)]
                    .map { weight * $0 }
            }
        }
        let expectedOutput = zip(expectedGated, convolution).map(+)
        expectClose(
            Fp16Buffer.read(gatedBuffer, count: expectedGated.count),
            expectedGated)
        expectClose(
            Fp16Buffer.read(outputBuffer, count: expectedOutput.count),
            expectedOutput)
    }

    @Test func dilatedConvolutionMatchesReferenceAcrossCallsAndReset() throws {
        let context = try MetalContext()
        let convolution = try Qwen38PLEConvolution(context: context)
        let channels = 5
        let state = try Qwen38PLEConvolutionState(
            device: context.device, channels: channels)
        var rows: [[Float]] = []
        for token in 0..<12 {
            var row: [Float] = []
            for channel in 0..<channels {
                let value = ((token + 1) * (channel + 2)) % 13 - 6
                row.append(Float(value) / 5)
            }
            rows.append(row)
        }
        var weights: [Float] = []
        for index in 0..<(channels * 4) {
            let value = Float((index * 7) % 11 - 5) / 8
            weights.append(Float(bitPattern: UInt32(value.bitPattern >> 16) << 16))
        }
        var history = [Float](repeating: 0, count: channels * 9)
        let expected = referenceConvolution(
            rows: rows, weights: weights, channels: channels, history: &history)

        let first = try runConvolution(
            context: context, convolution: convolution, state: state,
            rows: Array(rows.prefix(7)), weights: weights)
        let second = try runConvolution(
            context: context, convolution: convolution, state: state,
            rows: Array(rows.dropFirst(7)), weights: weights)
        expectClose(first + second, expected)

        state.reset()
        var resetHistory = [Float](repeating: 0, count: channels * 9)
        let resetExpected = referenceConvolution(
            rows: Array(rows.prefix(3)), weights: weights,
            channels: channels, history: &resetHistory)
        let resetActual = try runConvolution(
            context: context, convolution: convolution, state: state,
            rows: Array(rows.prefix(3)), weights: weights)
        expectClose(resetActual, resetExpected)
    }

    private func runConvolution(context: MetalContext,
                                convolution: Qwen38PLEConvolution,
                                state: Qwen38PLEConvolutionState,
                                rows: [[Float]],
                                weights: [Float]) throws -> [Float] {
        let input = try #require(Fp16Buffer.make(
            context.device, values: rows.flatMap { $0 }))
        let weightBits = weights.map(Quantization.bf16Bits)
        let weight = try #require(context.device.makeBuffer(
            bytes: weightBits,
            length: weightBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let output = try #require(Fp16Buffer.make(
            context.device, count: rows.count * state.channels))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        convolution.encode(
            commandBuffer: commandBuffer,
            input: input,
            weights: weight,
            output: output,
            state: state,
            tokenCount: UInt32(rows.count))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Fp16Buffer.read(output, count: rows.count * state.channels)
    }

    private func referenceConvolution(rows: [[Float]],
                                      weights: [Float],
                                      channels: Int,
                                      history: inout [Float]) -> [Float] {
        var output: [Float] = []
        for row in rows {
            for channel in 0..<channels {
                let stateBase = channel * 9
                let weightBase = channel * 4
                var value = weights[weightBase] * history[stateBase]
                value += weights[weightBase + 1] * history[stateBase + 3]
                value += weights[weightBase + 2] * history[stateBase + 6]
                value += weights[weightBase + 3] * row[channel]
                output.append(value / (1 + exp(-value)))
                for index in 0..<8 {
                    history[stateBase + index] = history[stateBase + index + 1]
                }
                history[stateBase + 8] = row[channel]
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
}
