import Foundation
import Metal
import TurboFieldfareFormat

struct Qwen38PLEAddressing: Sendable, Equatable {
    static let defaultSeed: UInt64 = 1_234

    let ngramSize: Int
    let headsPerNgram: Int
    let unigramVocabSize: Int64
    let ngramVocabSizeBase: Int64
    let eosTokenID: Int64
    let layerMultipliers: [Int64]
    let headVocabSizes: [Int64]
    let headOffsets: [Int64]
    let paddedVocabSize: Int64

    init(ngramSize: Int = 3,
         headsPerNgram: Int = 8,
         unigramVocabSize: Int64 = 248_320,
         ngramVocabSizeBase: Int64 = 20_000_000,
         eosTokenID: Int64 = 248_044,
         pleLayerIndex: Int = 0,
         seed: UInt64 = defaultSeed,
         vocabDivisor: Int64 = 128) {
        precondition(ngramSize > 1 && headsPerNgram > 0)
        precondition(unigramVocabSize > 0 && ngramVocabSizeBase > 1)
        precondition(pleLayerIndex >= 0 && vocabDivisor > 0)
        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.unigramVocabSize = unigramVocabSize
        self.ngramVocabSizeBase = ngramVocabSizeBase
        self.eosTokenID = eosTokenID
        self.layerMultipliers = Self.buildLayerMultipliers(
            unigramVocabSize: unigramVocabSize,
            ngramSize: ngramSize,
            pleLayerIndex: pleLayerIndex,
            seed: seed)

        let headCount = (ngramSize - 1) * headsPerNgram
        var sizes: [Int64] = []
        var offsets: [Int64] = []
        var total: Int64 = 0
        sizes.reserveCapacity(headCount)
        offsets.reserveCapacity(headCount)
        for head in 0..<headCount {
            let globalHead = pleLayerIndex * headCount + head
            let size = Self.findNthPrime(after: ngramVocabSizeBase - 1,
                                         count: globalHead + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        self.headVocabSizes = sizes
        self.headOffsets = offsets
        self.paddedVocabSize = ((total + vocabDivisor - 1) / vocabDivisor) * vocabDivisor
    }

    init(model: Model, layer: Int, architecture: Qwen38Architecture) throws {
        try self.init(
            layerMultipliers: model.qwen38PLE(layer: layer, tensor: .layerMultipliers),
            headOffsets: model.qwen38PLE(layer: layer, tensor: .ngramHeadOffsets),
            headVocabSizes: model.qwen38PLE(layer: layer, tensor: .ngramHeadVocabSizes),
            ngramSize: architecture.ngramSize,
            headsPerNgram: architecture.headsPerNgram,
            unigramVocabSize: Int64(model.config.vocabSize),
            ngramVocabSizeBase: Int64(architecture.ngramVocabSizeBase),
            vocabDivisor: Int64(architecture.ngramVocabSizeDivisor))
    }

    init(layerMultipliers: TensorView,
         headOffsets: TensorView,
         headVocabSizes: TensorView,
         ngramSize: Int,
         headsPerNgram: Int,
         unigramVocabSize: Int64,
         ngramVocabSizeBase: Int64,
         eosTokenID: Int64 = 248_044,
         vocabDivisor: Int64) throws {
        let multiplierValues = try Self.readMetadata(
            layerMultipliers,
            name: Qwen38TensorNames.PLETensor.layerMultipliers.rawValue,
            count: ngramSize)
        let headCount = (ngramSize - 1) * headsPerNgram
        let offsetValues = try Self.readMetadata(
            headOffsets,
            name: Qwen38TensorNames.PLETensor.ngramHeadOffsets.rawValue,
            count: headCount)
        let vocabValues = try Self.readMetadata(
            headVocabSizes,
            name: Qwen38TensorNames.PLETensor.ngramHeadVocabSizes.rawValue,
            count: headCount)
        try self.init(
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            unigramVocabSize: unigramVocabSize,
            ngramVocabSizeBase: ngramVocabSizeBase,
            eosTokenID: eosTokenID,
            layerMultipliers: multiplierValues,
            headVocabSizes: vocabValues,
            headOffsets: offsetValues,
            vocabDivisor: vocabDivisor)
    }

    private init(ngramSize: Int,
                 headsPerNgram: Int,
                 unigramVocabSize: Int64,
                 ngramVocabSizeBase: Int64,
                 eosTokenID: Int64,
                 layerMultipliers: [Int64],
                 headVocabSizes: [Int64],
                 headOffsets: [Int64],
                 vocabDivisor: Int64) throws {
        guard ngramSize > 1, headsPerNgram > 0,
              unigramVocabSize > 0, ngramVocabSizeBase > 1,
              vocabDivisor > 0 else {
            throw ModelError.indexCorrupt(detail: "invalid Qwen3.8 PLE addressing geometry")
        }
        let headCount = (ngramSize - 1) * headsPerNgram
        guard layerMultipliers.count == ngramSize,
              headVocabSizes.count == headCount,
              headOffsets.count == headCount,
              layerMultipliers.allSatisfy({ $0 > 0 }) else {
            throw ModelError.indexCorrupt(detail: "Qwen3.8 PLE metadata has invalid counts or multipliers")
        }

        var nextOffset: Int64 = 0
        for (offset, size) in zip(headOffsets, headVocabSizes) {
            guard size > 0, offset == nextOffset else {
                throw ModelError.indexCorrupt(
                    detail: "Qwen3.8 PLE metadata has non-contiguous head offsets")
            }
            let (end, overflow) = offset.addingReportingOverflow(size)
            guard !overflow else {
                throw ModelError.indexCorrupt(
                    detail: "Qwen3.8 PLE metadata head vocabulary overflows Int64")
            }
            nextOffset = end
        }
        let (paddedBase, baseOverflow) = nextOffset.addingReportingOverflow(vocabDivisor - 1)
        guard !baseOverflow else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 PLE metadata padded vocabulary overflows Int64")
        }
        let paddedVocabSize = (paddedBase / vocabDivisor) * vocabDivisor

        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.unigramVocabSize = unigramVocabSize
        self.ngramVocabSizeBase = ngramVocabSizeBase
        self.eosTokenID = eosTokenID
        self.layerMultipliers = layerMultipliers
        self.headVocabSizes = headVocabSizes
        self.headOffsets = headOffsets
        self.paddedVocabSize = paddedVocabSize
    }

    private static func readMetadata(_ view: TensorView,
                                     name: String,
                                     count: Int) throws -> [Int64] {
        guard count > 0,
              count <= Int.max / MemoryLayout<Int64>.stride,
              count <= Int(UInt32.max) else {
            throw ModelError.indexCorrupt(detail: "\(name) has an invalid element count")
        }
        let expectedBytes = UInt64(count * MemoryLayout<Int64>.stride)
        guard view.dtype == GTurboFormatV1.DType.i64.rawValue,
              view.length == expectedBytes,
              view.scaleLength == 0, view.biasLength == 0,
              view.shape.0 == UInt32(count), view.shape.1 == 0,
              view.shape.2 == 0, view.shape.3 == 0,
              view.offset <= UInt64(Int.max),
              expectedBytes <= UInt64(view.buffer.length) - view.offset else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 PLE metadata tensor \(name) has an invalid layout")
        }

        let source = view.buffer.contents().advanced(by: Int(view.offset))
        var values: [Int64] = []
        values.reserveCapacity(count)
        for index in 0..<count {
            var raw: UInt64 = 0
            memcpy(&raw,
                   source.advanced(by: index * MemoryLayout<UInt64>.stride),
                   MemoryLayout<UInt64>.stride)
            values.append(Int64(bitPattern: UInt64(littleEndian: raw)))
        }
        return values
    }

    func addresses(tokens: [Int64], previousContext: [Int64] = []) -> [[Int64]] {
        var context = Array(previousContext.suffix(ngramSize - 1))
        return addresses(tokens: tokens, context: &context)
    }

    func addresses(tokens: [Int64], context: inout [Int64]) -> [[Int64]] {
        context = Array(context.suffix(ngramSize - 1))
        return tokens.map { token in
            var shifted = [token]
            shifted += context.reversed()
            if shifted.count < ngramSize {
                shifted += repeatElement(eosTokenID, count: ngramSize - shifted.count)
            }

            var result: [Int64] = []
            result.reserveCapacity((ngramSize - 1) * headsPerNgram)
            for order in 2...ngramSize {
                var mixed = shifted[0] &* layerMultipliers[0]
                for position in 1..<order {
                    mixed ^= shifted[position] &* layerMultipliers[position]
                }
                let start = (order - 2) * headsPerNgram
                for head in start..<(start + headsPerNgram) {
                    let remainder = Self.positiveRemainder(mixed, headVocabSizes[head])
                    result.append(remainder + headOffsets[head])
                }
            }

            if token == eosTokenID {
                context.removeAll(keepingCapacity: true)
            } else {
                context.append(token)
                if context.count == ngramSize {
                    context.removeFirst()
                }
            }
            return result
        }
    }

    private static let splitMixGamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    private static let splitMixM1: UInt64 = 0xBF58_476D_1CE4_E5B9
    private static let splitMixM2: UInt64 = 0x94D0_49BB_1331_11EB
    private static let layerPrime: UInt64 = 10_007

    private static func splitMix64(_ input: UInt64) -> UInt64 {
        var value = input &+ splitMixGamma
        value = (value ^ (value >> 30)) &* splitMixM1
        value = (value ^ (value >> 27)) &* splitMixM2
        return value ^ (value >> 31)
    }

    private static func buildLayerMultipliers(unigramVocabSize: Int64,
                                              ngramSize: Int,
                                              pleLayerIndex: Int,
                                              seed: UInt64) -> [Int64] {
        let multiplierMax = UInt64(Int64.max / max(unigramVocabSize, 1))
        let halfBound = max(UInt64(1), multiplierMax / 2)
        let baseSeed = seed &+ layerPrime &* UInt64(pleLayerIndex)
        return (0..<ngramSize).map { index in
            let value = baseSeed &+ splitMixGamma &* UInt64(index + 1)
            return Int64(2 &* (splitMix64(value) % halfBound) &+ 1)
        }
    }

    private static func positiveRemainder(_ value: Int64, _ modulus: Int64) -> Int64 {
        let remainder = value % modulus
        return remainder >= 0 ? remainder : remainder + modulus
    }

    private static func findNthPrime(after start: Int64, count: Int) -> Int64 {
        var prime = start
        for _ in 0..<count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    private static func isPrime(_ value: Int64) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor: Int64 = 3
        while divisor <= value / divisor {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }
}

struct Qwen38PLEQuantizedProjection {
    let weights: MTLBuffer
    let weightsOffset: Int
    let scales: MTLBuffer
    let scalesOffset: Int
    let biases: MTLBuffer
    let biasesOffset: Int

    init(weights: MTLBuffer,
         weightsOffset: Int = 0,
         scales: MTLBuffer,
         scalesOffset: Int = 0,
         biases: MTLBuffer,
         biasesOffset: Int = 0) {
        self.weights = weights
        self.weightsOffset = weightsOffset
        self.scales = scales
        self.scalesOffset = scalesOffset
        self.biases = biases
        self.biasesOffset = biasesOffset
    }

    func validateCompanions(rows: Int, columns: Int, field: String) throws {
        let weightBytes = rows * columns / 2
        let companionCount = rows * (columns / 32)
        let packed = weights.contents()
            .advanced(by: weightsOffset)
            .assumingMemoryBound(to: UInt8.self)
        let hasPackedValues = (0..<weightBytes).contains { packed[$0] != 0 }
        guard hasPackedValues else { return }

        let scaleValues = scales.contents()
            .advanced(by: scalesOffset)
            .assumingMemoryBound(to: UInt16.self)
        let biasValues = biases.contents()
            .advanced(by: biasesOffset)
            .assumingMemoryBound(to: UInt16.self)
        let hasScale = (0..<companionCount).contains { scaleValues[$0] != 0 }
        let hasBias = (0..<companionCount).contains { biasValues[$0] != 0 }
        guard hasScale || hasBias else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 PLE \(field) has packed values but zero affine companions")
        }
    }
}

struct Qwen38PLEWeights {
    let keyProjection: Qwen38PLEQuantizedProjection
    let valueProjection: Qwen38PLEQuantizedProjection
    let keyNorm: MTLBuffer
    let keyNormOffset: Int
    let queryNorm: MTLBuffer
    let queryNormOffset: Int
    let convolutionNorm: MTLBuffer
    let convolutionNormOffset: Int
    let convolution: MTLBuffer
    let convolutionOffset: Int

    init(keyProjection: Qwen38PLEQuantizedProjection,
         valueProjection: Qwen38PLEQuantizedProjection,
         keyNorm: MTLBuffer,
         keyNormOffset: Int = 0,
         queryNorm: MTLBuffer,
         queryNormOffset: Int = 0,
         convolutionNorm: MTLBuffer,
         convolutionNormOffset: Int = 0,
         convolution: MTLBuffer,
         convolutionOffset: Int = 0) {
        self.keyProjection = keyProjection
        self.valueProjection = valueProjection
        self.keyNorm = keyNorm
        self.keyNormOffset = keyNormOffset
        self.queryNorm = queryNorm
        self.queryNormOffset = queryNormOffset
        self.convolutionNorm = convolutionNorm
        self.convolutionNormOffset = convolutionNormOffset
        self.convolution = convolution
        self.convolutionOffset = convolutionOffset
    }

    init(model: Model,
         layer: Int,
         geometry: Qwen38HyperConnectionGeometry = .qwen) throws {
        guard Qwen38TensorNames.hasPLE(layer: layer) else {
            throw ModelError.archMismatch(
                field: "pleLayer", expected: "layer with PLE tensors", actual: "\(layer)")
        }
        let embeddingSize = model.config.qwen38Architecture?.pleEmbeddingSize ?? 0
        let keyProjection = try model.qwen38PLE(layer: layer, tensor: .keyProjection)
        let valueProjection = try model.qwen38PLE(layer: layer, tensor: .valueProjection)
        let keyNorm = try model.qwen38PLE(layer: layer, tensor: .keyNorm)
        let queryNorm = try model.qwen38PLE(layer: layer, tensor: .queryNorm)
        let convolutionNorm = try model.qwen38PLE(layer: layer, tensor: .convolutionNorm)
        let convolution = try model.qwen38PLE(layer: layer, tensor: .convolution)

        let validatedKeyNorm = try Self.unquantized(
            keyNorm, elements: geometry.hyperWidth, field: "keyNorm")
        let validatedQueryNorm = try Self.unquantized(
            queryNorm, elements: geometry.hyperWidth, field: "queryNorm")
        let validatedConvolutionNorm = try Self.unquantized(
            convolutionNorm, elements: geometry.hyperWidth, field: "convolutionNorm")
        let validatedConvolution = try Self.unquantized(
            convolution,
            elements: geometry.hyperWidth * 4,
            field: "convolution",
            expectedShape: (geometry.hyperWidth, 4, 1, 0))

        let validatedKeyProjection = try Self.projection(
            keyProjection, rows: geometry.hyperWidth, columns: UInt32(embeddingSize))
        try validatedKeyProjection.validateCompanions(
            rows: Int(geometry.hyperWidth), columns: embeddingSize, field: "keyProjection")
        let validatedValueProjection = try Self.projection(
            valueProjection, rows: geometry.hiddenSize, columns: UInt32(embeddingSize))
        try validatedValueProjection.validateCompanions(
            rows: Int(geometry.hiddenSize), columns: embeddingSize, field: "valueProjection")

        self.init(
            keyProjection: validatedKeyProjection,
            valueProjection: validatedValueProjection,
            keyNorm: validatedKeyNorm.buffer,
            keyNormOffset: Int(validatedKeyNorm.offset),
            queryNorm: validatedQueryNorm.buffer,
            queryNormOffset: Int(validatedQueryNorm.offset),
            convolutionNorm: validatedConvolutionNorm.buffer,
            convolutionNormOffset: Int(validatedConvolutionNorm.offset),
            convolution: validatedConvolution.buffer,
            convolutionOffset: Int(validatedConvolution.offset))
    }

    private static func projection(
        _ view: TensorView,
        rows: UInt32,
        columns: UInt32
    ) throws -> Qwen38PLEQuantizedProjection {
        let weightBytes = UInt64(rows) * UInt64(columns) / 2
        let auxiliaryBytes = UInt64(rows) * UInt64(columns / 32)
            * UInt64(MemoryLayout<UInt16>.stride)
        guard columns.isMultiple(of: 32),
              view.dtype == GTurboFormatV1.DType.u32.rawValue,
              view.shape.0 == rows, view.shape.1 == columns,
              view.shape.2 == 0, view.shape.3 == 0,
              view.length == weightBytes,
              view.scaleLength == auxiliaryBytes,
              view.biasLength == auxiliaryBytes,
              view.offset.isMultiple(of: 2),
              view.scaleOffset.isMultiple(of: 2),
              view.biasOffset.isMultiple(of: 2) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 PLE affine-Q4 metadata mismatch")
        }
        return Qwen38PLEQuantizedProjection(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset))
    }

    private static func unquantized(
        _ view: TensorView,
        elements: UInt32,
        field: String,
        expectedShape: (UInt32, UInt32, UInt32, UInt32)? = nil
    ) throws -> TensorView {
        let expectedBytes = UInt64(elements) * UInt64(MemoryLayout<UInt16>.stride)
        let shape = expectedShape ?? (elements, 0, 0, 0)
        guard view.dtype == GTurboFormatV1.DType.bf16.rawValue,
              view.shape.0 == shape.0,
              view.shape.1 == shape.1,
              view.shape.2 == shape.2,
              view.shape.3 == shape.3,
              view.length == expectedBytes,
              view.scaleLength == 0, view.biasLength == 0,
              view.offset.isMultiple(of: UInt64(MemoryLayout<UInt16>.alignment)) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 PLE \(field) metadata mismatch")
        }
        return view
    }
}

struct Qwen38PLEScratch {
    let projectedKey: MTLBuffer
    let value: MTLBuffer
    let normalizedKey: MTLBuffer
    let normalizedQuery: MTLBuffer
    let gatedValue: MTLBuffer
    let normalizedGatedValue: MTLBuffer
    let convolution: MTLBuffer
}

final class Qwen38PLEPipeline {
    private let projection: Qwen38PLEProjection
    private let norm: Qwen38GatedResidual
    private let gate: Qwen38PLEGate
    private let convolution: Qwen38PLEConvolution

    init(context: MetalContext) throws {
        self.projection = try Qwen38PLEProjection(context: context)
        self.norm = try Qwen38GatedResidual(context: context)
        self.gate = try Qwen38PLEGate(context: context)
        self.convolution = try Qwen38PLEConvolution(context: context)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                embedding: MTLBuffer,
                hiddenStates: MTLBuffer,
                weights: Qwen38PLEWeights,
                scratch: Qwen38PLEScratch,
                state: Qwen38PLEConvolutionState,
                output: MTLBuffer,
                tokenCount: UInt32,
                streamCount: UInt32,
                hiddenSize: UInt32,
                embeddingSize: UInt32,
                epsilon: Float) {
        precondition(tokenCount > 0 && streamCount > 0 && hiddenSize > 0)
        precondition(embeddingSize > 0 && embeddingSize.isMultiple(of: 32))
        let tokenElements = Int(tokenCount)
        let hiddenElements = Int(hiddenSize)
        let channelElements = Int(streamCount) * hiddenElements
        let channelBytes = tokenElements * channelElements * MemoryLayout<UInt16>.stride
        let valueBytes = tokenElements * hiddenElements * MemoryLayout<UInt16>.stride
        precondition(state.channels == channelElements)
        precondition(embedding.length >= tokenElements * Int(embeddingSize)
            * MemoryLayout<UInt16>.stride)
        precondition(hiddenStates.length >= channelBytes && output.length >= channelBytes)
        precondition(scratch.projectedKey.length >= channelBytes)
        precondition(scratch.normalizedKey.length >= channelBytes)
        precondition(scratch.normalizedQuery.length >= channelBytes)
        precondition(scratch.gatedValue.length >= channelBytes)
        precondition(scratch.normalizedGatedValue.length >= channelBytes)
        precondition(scratch.convolution.length >= channelBytes)
        precondition(scratch.value.length >= valueBytes)

        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.keyProjection.weights,
            weightsOffset: weights.keyProjection.weightsOffset,
            scales: weights.keyProjection.scales,
            scalesOffset: weights.keyProjection.scalesOffset,
            biases: weights.keyProjection.biases,
            biasesOffset: weights.keyProjection.biasesOffset,
            input: embedding,
            output: scratch.projectedKey,
            tokenCount: tokenCount,
            outputWidth: streamCount * hiddenSize,
            inputWidth: embeddingSize)
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.valueProjection.weights,
            weightsOffset: weights.valueProjection.weightsOffset,
            scales: weights.valueProjection.scales,
            scalesOffset: weights.valueProjection.scalesOffset,
            biases: weights.valueProjection.biases,
            biasesOffset: weights.valueProjection.biasesOffset,
            input: embedding,
            output: scratch.value,
            tokenCount: tokenCount,
            outputWidth: hiddenSize,
            inputWidth: embeddingSize)
        norm.encodeGroupedNorm(
            commandBuffer: commandBuffer,
            input: scratch.projectedKey,
            weight: weights.keyNorm,
            weightOffset: weights.keyNormOffset,
            output: scratch.normalizedKey,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize,
            epsilon: epsilon)
        norm.encodeGroupedNorm(
            commandBuffer: commandBuffer,
            input: hiddenStates,
            weight: weights.queryNorm,
            weightOffset: weights.queryNormOffset,
            output: scratch.normalizedQuery,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize,
            epsilon: epsilon)
        gate.encodeGate(
            commandBuffer: commandBuffer,
            normalizedKey: scratch.normalizedKey,
            normalizedQuery: scratch.normalizedQuery,
            value: scratch.value,
            output: scratch.gatedValue,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize)
        norm.encodeGroupedNorm(
            commandBuffer: commandBuffer,
            input: scratch.gatedValue,
            weight: weights.convolutionNorm,
            weightOffset: weights.convolutionNormOffset,
            output: scratch.normalizedGatedValue,
            tokenCount: tokenCount,
            streamCount: streamCount,
            hiddenSize: hiddenSize,
            epsilon: epsilon)
        convolution.encode(
            commandBuffer: commandBuffer,
            input: scratch.normalizedGatedValue,
            weights: weights.convolution,
            weightsOffset: weights.convolutionOffset,
            output: scratch.convolution,
            state: state,
            tokenCount: tokenCount)
        gate.encodeResidualMerge(
            commandBuffer: commandBuffer,
            gatedValue: scratch.gatedValue,
            convolution: scratch.convolution,
            output: output,
            count: tokenCount * streamCount * hiddenSize)
    }
}

final class Qwen38PLEProjection {
    private static let groupSize: UInt32 = 32
    private static let rowsPerThreadgroup = 8

    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "qwen38_ple_affine_q4_group32_projection",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                input: MTLBuffer,
                output: MTLBuffer,
                tokenCount: UInt32,
                outputWidth: UInt32,
                inputWidth: UInt32,
                transposeWeights: Bool = false) {
        precondition(tokenCount > 0 && outputWidth > 0)
        precondition(inputWidth > 0 && inputWidth.isMultiple(of: Self.groupSize))
        precondition(weightsOffset >= 0 && scalesOffset >= 0 && biasesOffset >= 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        var outputs = outputWidth
        var inputs = inputWidth
        var tokens = tokenCount
        var transpose = transposeWeights ? UInt32(1) : UInt32(0)
        encoder.setBytes(&outputs, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&inputs, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&transpose, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(outputWidth) + Self.rowsPerThreadgroup - 1)
                    / Self.rowsPerThreadgroup,
                height: Int(tokenCount),
                depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

final class Qwen38PLEGate {
    private let gatePipeline: MTLComputePipelineState
    private let mergePipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.gatePipeline = try context.pipeline("qwen38_ple_gate")
        self.mergePipeline = try context.pipeline("qwen38_ple_residual_merge")
    }

    func encodeGate(commandBuffer: MTLCommandBuffer,
                    normalizedKey: MTLBuffer,
                    normalizedQuery: MTLBuffer,
                    value: MTLBuffer,
                    output: MTLBuffer,
                    tokenCount: UInt32,
                    streamCount: UInt32,
                    hiddenSize: UInt32) {
        precondition(tokenCount > 0 && streamCount > 0 && hiddenSize > 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gatePipeline)
        encoder.setBuffer(normalizedKey, offset: 0, index: 0)
        encoder.setBuffer(normalizedQuery, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var tokens = tokenCount
        var streams = streamCount
        var hidden = hiddenSize
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&streams, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&hidden, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(streamCount), height: Int(tokenCount), depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(streamCount), gatePipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    func encodeResidualMerge(commandBuffer: MTLCommandBuffer,
                             gatedValue: MTLBuffer,
                             convolution: MTLBuffer,
                             output: MTLBuffer,
                             count: UInt32) {
        precondition(count > 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mergePipeline)
        encoder.setBuffer(gatedValue, offset: 0, index: 0)
        encoder.setBuffer(convolution, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var elementCount = count
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(count), mergePipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }
}

struct Qwen38PLEConvolutionSnapshot {
    let history: [UInt8]
}

final class Qwen38PLEConvolutionState {
    let buffer: MTLBuffer
    let channels: Int
    let kernelSize: Int
    let dilation: Int
    var historyLength: Int { (kernelSize - 1) * dilation }

    init(device: MTLDevice,
         channels: Int,
         kernelSize: Int = 4,
         dilation: Int = 3) throws {
        precondition(channels > 0 && kernelSize > 1 && dilation > 0)
        self.channels = channels
        self.kernelSize = kernelSize
        self.dilation = dilation
        let bytes = channels * (kernelSize - 1) * dilation * MemoryLayout<UInt16>.stride
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.buffer = buffer
        reset()
    }

    func snapshot() -> Qwen38PLEConvolutionSnapshot {
        let pointer = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return Qwen38PLEConvolutionSnapshot(
            history: Array(UnsafeBufferPointer(start: pointer, count: buffer.length)))
    }

    func restore(_ snapshot: Qwen38PLEConvolutionSnapshot) {
        precondition(snapshot.history.count == buffer.length,
                     "PLE convolution snapshot size does not match state")
        snapshot.history.withUnsafeBytes { source in
            memcpy(buffer.contents(), source.baseAddress!, snapshot.history.count)
        }
    }

    func reset() {
        memset(buffer.contents(), 0, buffer.length)
    }
}

final class Qwen38PLEConvolution {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline("qwen38_ple_dilated_causal_conv")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                input: MTLBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                output: MTLBuffer,
                state: Qwen38PLEConvolutionState,
                tokenCount: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(state.buffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var channels = UInt32(state.channels)
        var kernelSize = UInt32(state.kernelSize)
        var dilation = UInt32(state.dilation)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kernelSize, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&dilation, length: MemoryLayout<UInt32>.stride, index: 6)
        var tokens = tokenCount
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 7)
        let width = min(state.channels, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: state.channels, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
