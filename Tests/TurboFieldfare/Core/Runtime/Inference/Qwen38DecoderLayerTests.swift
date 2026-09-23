import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareFormat
import TurboFieldfareValidationSupport

@Suite struct Qwen38DecoderLayerTests {
    @Test func selectsLinearAndSparseStateDomains() throws {
        let fixture = try makeRuntimeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let executor = try Qwen38DecoderLayerExecutor(context: fixture.context)

        switch try executor.attentionState(layer: 0, runtimeState: fixture.state) {
        case .linear:
            break
        case .sparse:
            Issue.record("layer 0 must use DeltaNet state")
        }

        switch try executor.attentionState(layer: 3, runtimeState: fixture.state) {
        case .linear:
            Issue.record("layer 3 must use paired sparse-attention state")
        case .sparse(let qsa, let cache):
            #expect(qsa.layer == 3)
            #expect(qsa.rawKeyCache.count == 0)
            #expect(cache.count == 0)
        }
        #expect(throws: ModelError.self) {
            _ = try executor.attentionState(layer: -1, runtimeState: fixture.state)
        }
        #expect(throws: ModelError.self) {
            _ = try executor.attentionState(layer: 48, runtimeState: fixture.state)
        }
    }

    @Test func restoresContinuationStateAcrossOwners() throws {
        let fixture = try makeRuntimeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let deltaState = try #require(fixture.state.deltaState(layer: 0))
        fixture.state.pleConvolution.buffer.contents()
            .assumingMemoryBound(to: UInt8.self).pointee = 0x34
        deltaState.recurrentBuffer.contents()
            .assumingMemoryBound(to: UInt8.self).pointee = 0x56
        let snapshot = fixture.state.snapshot()

        memset(fixture.state.pleConvolution.buffer.contents(), 0,
               fixture.state.pleConvolution.buffer.length)
        memset(deltaState.recurrentBuffer.contents(), 0,
               deltaState.recurrentBuffer.length)
        fixture.state.restore(snapshot)

        #expect(fixture.state.pleConvolution.buffer.contents()
            .assumingMemoryBound(to: UInt8.self).pointee == 0x34)
        #expect(deltaState.recurrentBuffer.contents()
            .assumingMemoryBound(to: UInt8.self).pointee == 0x56)
        #expect(snapshot.qsa.layers.count == ArchConfig.qwen38FlashNextText.numLayers)
        #expect(snapshot.deltaStates.count == ArchConfig.qwen38FlashNextText.numLayers)
        #expect(snapshot.fullCaches.count == ArchConfig.qwen38FlashNextText.numLayers)
        #expect(fixture.state.linearLayerCount == 36)
        #expect(fixture.state.sparseLayerCount == 12)
    }

    @Test func rejectsInjectionFreeLayerWeights() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = Qwen38HyperConnectionGeometry(
            streamCount: 2,
            hiddenSize: 32,
            lowRankSize: 32)
        let valid = try makeWeights(device: device, geometry: geometry)
        let finalMixer = Qwen38HyperConnectionWeights(
            norm: valid.norm,
            normOffset: valid.normOffset,
            inputMixDown: valid.inputMixDown,
            inputMixUp: valid.inputMixUp,
            blockInject: nil)

        #expect(throws: ModelError.self) {
            _ = try Qwen38DecoderLayerWeights(
                layer: 0,
                attention: valid,
                mlp: finalMixer)
        }
    }

    @Test func deltaNetWeightsRejectMalformedProjectionMetadata() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = Qwen38DeltaNetGeometry(
            hiddenSize: 32,
            keyHeads: 1,
            valueHeads: 1,
            keyHeadDimension: 32,
            valueHeadDimension: 32,
            convolutionKernel: 4)
        let views = try makeDeltaWeightViews(device: device, geometry: geometry)
        _ = try views.weights(geometry: geometry)
        let malformedQKV = TensorView(
            buffer: views.qkv.buffer,
            offset: views.qkv.offset,
            length: views.qkv.length,
            scaleOffset: views.qkv.scaleOffset,
            scaleLength: views.qkv.scaleLength,
            biasOffset: views.qkv.biasOffset,
            biasLength: views.qkv.biasLength,
            shape: (geometry.hiddenSize, geometry.qkvWidth, 0, 0),
            dtype: views.qkv.dtype)

        #expect(throws: ModelError.self) {
            _ = try Qwen38DeltaNetWeights(
                qkv: malformedQKV,
                gate: views.gate,
                beta: views.beta,
                decay: views.decay,
                convolution: views.convolution,
                decayLog: views.decayLog,
                timeBias: views.timeBias,
                norm: views.norm,
                output: views.output,
                geometry: geometry)
        }
    }

    @Test func deltaNetDecoderAdvancesOwnedState() throws {
        let context = try MetalContext()
        let geometry = Qwen38DeltaNetGeometry(
            hiddenSize: 32,
            keyHeads: 1,
            valueHeads: 1,
            keyHeadDimension: 32,
            valueHeadDimension: 32,
            convolutionKernel: 4)
        let views = try makeDeltaWeightViews(
            device: context.device,
            geometry: geometry)
        let weights = try views.weights(geometry: geometry)
        let stateGeometry = QwenGatedDeltaNetGeometry(
            keyHeads: 1,
            valueHeads: 1,
            keyHeadDim: 32,
            valueHeadDim: 32,
            convolutionKernel: 4)
        let state = try QwenGatedDeltaNetState(
            device: context.device,
            geometry: stateGeometry,
            convolutionChannels: Int(geometry.qkvWidth))
        let decoder = try Qwen38DeltaNetDecoder(
            context: context,
            geometry: geometry)
        let scratch = try makeDeltaScratch(
            device: context.device,
            geometry: geometry)
        let input = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: 1, count: Int(geometry.hiddenSize))))
        let output = try #require(Fp16Buffer.make(
            context.device,
            count: Int(geometry.hiddenSize)))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        try decoder.encode(
            commandBuffer: commandBuffer,
            state: .linear(state),
            weights: weights,
            input: input,
            scratch: scratch,
            output: output,
            epsilon: 1e-6)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let values = Fp16Buffer.read(output, count: Int(geometry.hiddenSize))
        #expect(values.allSatisfy { $0.isFinite })
        #expect(values.contains { abs($0) > 0 })
        #expect(Data(
            bytes: state.convolutionBuffer.contents(),
            count: state.convolutionBuffer.length).contains { $0 != 0 })
        #expect(Data(
            bytes: state.recurrentBuffer.contents(),
            count: state.recurrentBuffer.length).contains { $0 != 0 })
    }

    @Test func deltaNetDecoderBatchMatchesScalarSequence() throws {
        let context = try MetalContext()
        let geometry = Qwen38DeltaNetGeometry(
            hiddenSize: 32,
            keyHeads: 1,
            valueHeads: 1,
            keyHeadDimension: 32,
            valueHeadDimension: 32,
            convolutionKernel: 4)
        let views = try makeDeltaWeightViews(
            device: context.device,
            geometry: geometry)
        let weights = try views.weights(geometry: geometry)
        let decoder = try Qwen38DeltaNetDecoder(
            context: context,
            geometry: geometry)
        let tokenCount = 2
        let inputs = (0..<(tokenCount * Int(geometry.hiddenSize))).map {
            Float(($0 * 5) % 17 - 8) / 9
        }

        let scalarState = try QwenGatedDeltaNetState(
            device: context.device,
            geometry: QwenGatedDeltaNetGeometry(
                keyHeads: 1,
                valueHeads: 1,
                keyHeadDim: 32,
                valueHeadDim: 32,
                convolutionKernel: 4),
            convolutionChannels: Int(geometry.qkvWidth))
        var scalarOutput: [Float] = []
        for token in 0..<tokenCount {
            let inputStart = token * Int(geometry.hiddenSize)
            let inputEnd = (token + 1) * Int(geometry.hiddenSize)
            let input = try #require(Fp16Buffer.make(
                context.device,
                values: Array(inputs[inputStart..<inputEnd])))
            let output = try #require(Fp16Buffer.make(
                context.device,
                count: Int(geometry.hiddenSize)))
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            try decoder.encode(
                commandBuffer: commandBuffer,
                state: .linear(scalarState),
                weights: weights,
                input: input,
                scratch: try makeDeltaScratch(
                    device: context.device,
                    geometry: geometry),
                output: output,
                epsilon: 1e-6)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            scalarOutput += Fp16Buffer.read(
                output,
                count: Int(geometry.hiddenSize))
        }

        let batchState = try QwenGatedDeltaNetState(
            device: context.device,
            geometry: scalarState.geometry,
            convolutionChannels: Int(geometry.qkvWidth))
        let batchInput = try #require(Fp16Buffer.make(
            context.device,
            values: inputs))
        let batchOutput = try #require(Fp16Buffer.make(
            context.device,
            count: tokenCount * Int(geometry.hiddenSize)))
        let batchCommandBuffer = try #require(context.queue.makeCommandBuffer())
        try decoder.encodeBatch(
            commandBuffer: batchCommandBuffer,
            state: .linear(batchState),
            weights: weights,
            input: batchInput,
            scratch: try makeDeltaScratch(
                device: context.device,
                geometry: geometry,
                tokenCount: tokenCount),
            output: batchOutput,
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6)
        batchCommandBuffer.commit()
        batchCommandBuffer.waitUntilCompleted()
        #expect(batchCommandBuffer.error == nil)

        let batchOutputValues = Fp16Buffer.read(
            batchOutput,
            count: tokenCount * Int(geometry.hiddenSize))
        let maxDifference = zip(scalarOutput, batchOutputValues)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxDifference < 0.02, "maxDifference=\(maxDifference)")
    }

    @Test func composesAttentionThenMoEHyperConnections() throws {
        let fixture = try makeRuntimeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let geometry = Qwen38HyperConnectionGeometry(
            streamCount: 2,
            hiddenSize: 32,
            lowRankSize: 32)
        let executor = try Qwen38DecoderLayerExecutor(
            context: fixture.context,
            geometry: geometry)
        let hyperConnection = try Qwen38HyperConnection(
            context: fixture.context,
            geometry: geometry)
        let tokenCount = 2
        let inputValues = (0..<(tokenCount * Int(geometry.hyperWidth))).map {
            Float(($0 * 7) % 23 - 11) / 8
        }
        let input = try #require(Fp16Buffer.make(
            fixture.context.device,
            values: inputValues))
        let weights = try Qwen38DecoderLayerWeights(
            layer: 0,
            attention: try makeWeights(device: fixture.context.device, geometry: geometry),
            mlp: try makeWeights(device: fixture.context.device, geometry: geometry))
        let scratch = try makeLayerScratch(
            device: fixture.context.device,
            tokenCount: tokenCount,
            geometry: geometry)
        let output = try #require(Fp16Buffer.make(
            fixture.context.device,
            count: tokenCount * Int(geometry.hyperWidth)))
        var branchOrder: [String] = []
        let commandBuffer = try #require(fixture.context.queue.makeCommandBuffer())

        try executor.encode(
            commandBuffer: commandBuffer,
            weights: weights,
            runtimeState: fixture.state,
            hyperInput: input,
            scratch: scratch,
            output: output,
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6,
            attentionEncoder: { commandBuffer, state, input, output, _ in
                if case .linear = state {
                    branchOrder.append("attention")
                } else {
                    Issue.record("layer 0 must provide linear state")
                }
                try encodeCopy(
                    commandBuffer: commandBuffer,
                    input: input,
                    output: output,
                    byteCount: tokenCount * Int(geometry.hiddenSize)
                        * MemoryLayout<Float16>.stride)
            },
            moeEncoder: { commandBuffer, input, output, _ in
                branchOrder.append("moe")
                try encodeCopy(
                    commandBuffer: commandBuffer,
                    input: input,
                    output: output,
                    byteCount: tokenCount * Int(geometry.hiddenSize)
                        * MemoryLayout<Float16>.stride)
            })
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(branchOrder == ["attention", "moe"])

        let expectedScratch = try makeLayerScratch(
            device: fixture.context.device,
            tokenCount: tokenCount,
            geometry: geometry)
        let expected = try #require(Fp16Buffer.make(
            fixture.context.device,
            count: tokenCount * Int(geometry.hyperWidth)))
        let expectedCommandBuffer = try #require(
            fixture.context.queue.makeCommandBuffer())
        hyperConnection.encodePrepare(
            commandBuffer: expectedCommandBuffer,
            hyperInput: input,
            weights: weights.attention,
            scratch: expectedScratch.attentionHyperConnection,
            mixedInput: expectedScratch.attentionInput,
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6)
        try encodeCopy(
            commandBuffer: expectedCommandBuffer,
            input: expectedScratch.attentionInput,
            output: expectedScratch.attentionOutput,
            byteCount: tokenCount * Int(geometry.hiddenSize)
                * MemoryLayout<Float16>.stride)
        hyperConnection.encodeInject(
            commandBuffer: expectedCommandBuffer,
            hyperInput: input,
            branchOutput: expectedScratch.attentionOutput,
            injectionWeights: expectedScratch.attentionHyperConnection.injectionWeights,
            output: expectedScratch.afterAttention,
            tokenCount: UInt32(tokenCount))
        hyperConnection.encodePrepare(
            commandBuffer: expectedCommandBuffer,
            hyperInput: expectedScratch.afterAttention,
            weights: weights.mlp,
            scratch: expectedScratch.mlpHyperConnection,
            mixedInput: expectedScratch.mlpInput,
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6)
        try encodeCopy(
            commandBuffer: expectedCommandBuffer,
            input: expectedScratch.mlpInput,
            output: expectedScratch.mlpOutput,
            byteCount: tokenCount * Int(geometry.hiddenSize)
                * MemoryLayout<Float16>.stride)
        hyperConnection.encodeInject(
            commandBuffer: expectedCommandBuffer,
            hyperInput: expectedScratch.afterAttention,
            branchOutput: expectedScratch.mlpOutput,
            injectionWeights: expectedScratch.mlpHyperConnection.injectionWeights,
            output: expected,
            tokenCount: UInt32(tokenCount))
        expectedCommandBuffer.commit()
        expectedCommandBuffer.waitUntilCompleted()
        #expect(expectedCommandBuffer.error == nil)

        #expect(Fp16Buffer.read(
            output,
            count: tokenCount * Int(geometry.hyperWidth)) == Fp16Buffer.read(
                expected,
                count: tokenCount * Int(geometry.hyperWidth)))
    }
}

private struct Qwen38RuntimeFixture {
    let directory: URL
    let context: MetalContext
    let state: Qwen38RuntimeState
}

private func makeRuntimeFixture() throws -> Qwen38RuntimeFixture {
    let directory = try ModelLoaderTests.writeToySynthetic()
    let context = try MetalContext()
    let base = try Model.load(
        directoryURL: directory,
        device: context.device,
        expecting: .gemma4Toy())
    let qsaNames = (0..<ArchConfig.qwen38FlashNextText.numLayers)
        .filter(Qwen38TensorNames.hasQSA(layer:))
        .flatMap { layer in
            [
                Qwen38TensorNames.qsa(layer: layer, tensor: .queryKeyProjection),
                Qwen38TensorNames.qsa(layer: layer, tensor: .queryNorm),
                Qwen38TensorNames.qsa(layer: layer, tensor: .keyNorm),
            ]
        }
    let source = try #require(
        base.residentIndex.entries["language_model.model.embed_tokens.weight"])
    var entries = base.residentIndex.entries
    for name in qsaNames {
        entries[name] = ResidentIndexEntry(
            name: name,
            dtype: source.dtype,
            fileOffset: source.fileOffset,
            sizeBytes: source.sizeBytes,
            shape: source.shape,
            scaleOffset: source.scaleOffset,
            scaleSize: source.scaleSize,
            biasOffset: source.biasOffset,
            biasSize: source.biasSize)
    }
    let model = Model(
        device: base.device,
        config: .qwen38FlashNextText,
        streamingMode: base.streamingMode,
        expertCachePolicy: base.expertCachePolicy,
        integrityPolicy: base.integrityPolicy,
        residentBuffer: base.residentBuffer,
        residentIndex: ResidentIndex(header: base.residentIndex.header, entries: entries),
        packedExpertsLayout: base.packedExpertsLayout,
        manifest: base.manifest,
        directoryURL: base.directoryURL,
        modelDirectory: base.modelDirectory,
        trustedInstallReceipt: base.trustedInstallReceipt)
    return Qwen38RuntimeFixture(
        directory: directory,
        context: context,
        state: try Qwen38RuntimeState(model: model, maxContext: 4))
}

private func makeWeights(
    device: MTLDevice,
    geometry: Qwen38HyperConnectionGeometry
) throws -> Qwen38HyperConnectionWeights {
    let norm = try #require(device.makeBuffer(
        length: Int(geometry.hyperWidth) * MemoryLayout<UInt16>.stride,
        options: .storageModeShared))
    memset(norm.contents(), 0, norm.length)
    return Qwen38HyperConnectionWeights(
        norm: norm,
        inputMixDown: try makeProjection(
            device: device,
            outputWidth: geometry.lowRankSize,
            inputWidth: geometry.hyperWidth),
        inputMixUp: try makeProjection(
            device: device,
            outputWidth: geometry.hyperWidth,
            inputWidth: geometry.lowRankSize),
        blockInject: try makeProjection(
            device: device,
            outputWidth: geometry.streamCount,
            inputWidth: geometry.hyperWidth))
}

private func makeProjection(
    device: MTLDevice,
    outputWidth: UInt32,
    inputWidth: UInt32
) throws -> Qwen38PLEQuantizedProjection {
    let weightBytes = Int(outputWidth * inputWidth / 2)
    let auxiliaryBytes = Int(outputWidth * (inputWidth / 32))
        * MemoryLayout<UInt16>.stride
    let buffer = try #require(device.makeBuffer(
        length: weightBytes + auxiliaryBytes * 2,
        options: .storageModeShared))
    memset(buffer.contents(), 0, buffer.length)
    return Qwen38PLEQuantizedProjection(
        weights: buffer,
        weightsOffset: 0,
        scales: buffer,
        scalesOffset: weightBytes,
        biases: buffer,
        biasesOffset: weightBytes + auxiliaryBytes)
}

private func makeLayerScratch(
    device: MTLDevice,
    tokenCount: Int,
    geometry: Qwen38HyperConnectionGeometry
) throws -> Qwen38DecoderLayerScratch {
    let hiddenCount = tokenCount * Int(geometry.hiddenSize)
    let hyperCount = tokenCount * Int(geometry.hyperWidth)
    return Qwen38DecoderLayerScratch(
        attentionHyperConnection: try makeHyperConnectionScratch(
            device: device,
            tokenCount: tokenCount,
            geometry: geometry),
        attentionInput: try #require(device.makeBuffer(
            length: hiddenCount * MemoryLayout<Float>.stride,
            options: .storageModeShared)),
        attentionOutput: try #require(Fp16Buffer.make(device, count: hiddenCount)),
        attentionOutputFloat: try #require(device.makeBuffer(
            length: hiddenCount * MemoryLayout<Float>.stride,
            options: .storageModeShared)),
        afterAttention: try #require(Fp16Buffer.make(device, count: hyperCount)),
        mlpHyperConnection: try makeHyperConnectionScratch(
            device: device,
            tokenCount: tokenCount,
            geometry: geometry),
        mlpInput: try #require(Fp16Buffer.make(device, count: hiddenCount)),
        mlpOutput: try #require(Fp16Buffer.make(device, count: hiddenCount)))
}

private func makeHyperConnectionScratch(
    device: MTLDevice,
    tokenCount: Int,
    geometry: Qwen38HyperConnectionGeometry
) throws -> Qwen38HyperConnectionScratch {
    let hyperCount = tokenCount * Int(geometry.hyperWidth)
    let lowRankCount = tokenCount * Int(geometry.lowRankSize)
    let streamCount = tokenCount * Int(geometry.streamCount)
    return Qwen38HyperConnectionScratch(
        normalized: try #require(device.makeBuffer(
            length: hyperCount * MemoryLayout<Float>.stride,
            options: .storageModeShared)),
        lowRank: try #require(Fp16Buffer.make(device, count: lowRankCount)),
        activatedLowRank: try #require(Fp16Buffer.make(device, count: lowRankCount)),
        mixLogits: try #require(Fp16Buffer.make(device, count: hyperCount)),
        injectionLogits: try #require(Fp16Buffer.make(device, count: streamCount)),
        injectionWeights: try #require(Fp16Buffer.make(device, count: streamCount)))
}

private func encodeCopy(
    commandBuffer: MTLCommandBuffer,
    input: MTLBuffer,
    output: MTLBuffer,
    byteCount: Int
) throws {
    let encoder = try #require(commandBuffer.makeBlitCommandEncoder())
    encoder.copy(
        from: input,
        sourceOffset: 0,
        to: output,
        destinationOffset: 0,
        size: byteCount)
    encoder.endEncoding()
}

private struct Qwen38DeltaWeightViews {
    let qkv: TensorView
    let gate: TensorView
    let beta: TensorView
    let decay: TensorView
    let convolution: TensorView
    let decayLog: TensorView
    let timeBias: TensorView
    let norm: TensorView
    let output: TensorView

    func weights(geometry: Qwen38DeltaNetGeometry) throws -> Qwen38DeltaNetWeights {
        try Qwen38DeltaNetWeights(
            qkv: qkv,
            gate: gate,
            beta: beta,
            decay: decay,
            convolution: convolution,
            decayLog: decayLog,
            timeBias: timeBias,
            norm: norm,
            output: output,
            geometry: geometry)
    }
}

private func makeDeltaWeightViews(
    device: MTLDevice,
    geometry: Qwen38DeltaNetGeometry
) throws -> Qwen38DeltaWeightViews {
    var convolutionBits = [UInt16](
        repeating: 0,
        count: Int(geometry.qkvWidth * geometry.convolutionKernel))
    for channel in 0..<Int(geometry.qkvWidth) {
        convolutionBits[channel * Int(geometry.convolutionKernel)
            + Int(geometry.convolutionKernel - 1)] = Quantization.bf16Bits(1)
    }
    return Qwen38DeltaWeightViews(
        qkv: try makeDeltaProjectionView(
            device: device,
            rows: geometry.qkvWidth,
            columns: geometry.hiddenSize),
        gate: try makeDeltaProjectionView(
            device: device,
            rows: geometry.valueWidth,
            columns: geometry.hiddenSize),
        beta: try makeDeltaProjectionView(
            device: device,
            rows: geometry.valueHeads,
            columns: geometry.hiddenSize),
        decay: try makeDeltaProjectionView(
            device: device,
            rows: geometry.valueHeads,
            columns: geometry.hiddenSize),
        convolution: try makeBF16View(
            device: device,
            values: convolutionBits,
            shape: (
                geometry.qkvWidth,
                geometry.convolutionKernel,
                1,
                0)),
        decayLog: try makeBF16View(
            device: device,
            values: [UInt16](
                repeating: Quantization.bf16Bits(0),
                count: Int(geometry.valueHeads)),
            shape: (geometry.valueHeads, 0, 0, 0)),
        timeBias: try makeBF16View(
            device: device,
            values: [UInt16](
                repeating: Quantization.bf16Bits(0),
                count: Int(geometry.valueHeads)),
            shape: (geometry.valueHeads, 0, 0, 0)),
        norm: try makeBF16View(
            device: device,
            values: [UInt16](
                repeating: Quantization.bf16Bits(1),
                count: Int(geometry.valueHeadDimension)),
            shape: (geometry.valueHeadDimension, 0, 0, 0)),
        output: try makeDeltaProjectionView(
            device: device,
            rows: geometry.hiddenSize,
            columns: geometry.valueWidth))
}

private func makeDeltaProjectionView(
    device: MTLDevice,
    rows: UInt32,
    columns: UInt32
) throws -> TensorView {
    let weightBytes = Int(rows * columns / 2)
    let auxiliaryCount = Int(rows * (columns / 32))
    let auxiliaryBytes = auxiliaryCount * MemoryLayout<UInt16>.stride
    let buffer = try #require(device.makeBuffer(
        length: weightBytes + auxiliaryBytes * 2,
        options: .storageModeShared))
    memset(buffer.contents(), 0, buffer.length)
    let biases = buffer.contents()
        .advanced(by: weightBytes + auxiliaryBytes)
        .assumingMemoryBound(to: UInt16.self)
    let bias = Quantization.bf16Bits(0.01)
    for index in 0..<auxiliaryCount {
        biases[index] = bias
    }
    return TensorView(
        buffer: buffer,
        offset: 0,
        length: UInt64(weightBytes),
        scaleOffset: UInt64(weightBytes),
        scaleLength: UInt64(auxiliaryBytes),
        biasOffset: UInt64(weightBytes + auxiliaryBytes),
        biasLength: UInt64(auxiliaryBytes),
        shape: (rows, columns, 0, 0),
        dtype: GTurboFormatV1.DType.u32.rawValue)
}

private func makeBF16View(
    device: MTLDevice,
    values: [UInt16],
    shape: (UInt32, UInt32, UInt32, UInt32)
) throws -> TensorView {
    let buffer = try #require(device.makeBuffer(
        bytes: values,
        length: values.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared))
    return TensorView(
        buffer: buffer,
        offset: 0,
        length: UInt64(buffer.length),
        scaleOffset: 0,
        scaleLength: 0,
        biasOffset: 0,
        biasLength: 0,
        shape: shape,
        dtype: GTurboFormatV1.DType.bf16.rawValue)
}

private func makeFP32View(
    device: MTLDevice,
    values: [Float]
) throws -> TensorView {
    let buffer = try #require(device.makeBuffer(
        bytes: values,
        length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared))
    return TensorView(
        buffer: buffer,
        offset: 0,
        length: UInt64(buffer.length),
        scaleOffset: 0,
        scaleLength: 0,
        biasOffset: 0,
        biasLength: 0,
        shape: (UInt32(values.count), 0, 0, 0),
        dtype: GTurboFormatV1.DType.fp32.rawValue)
}

private func makeDeltaScratch(
    device: MTLDevice,
    geometry: Qwen38DeltaNetGeometry,
    tokenCount: Int = 1
) throws -> Qwen38DeltaNetScratch {
    let qkvWidth = Int(geometry.qkvWidth)
    let keyWidth = Int(geometry.keyWidth)
    let valueWidth = Int(geometry.valueWidth)
    let heads = Int(geometry.valueHeads)
    return Qwen38DeltaNetScratch(
        qkv: try #require(Fp16Buffer.make(device, count: tokenCount * qkvWidth)),
        gate: try #require(Fp16Buffer.make(device, count: tokenCount * valueWidth)),
        betaInput: try #require(Fp16Buffer.make(device, count: tokenCount * heads)),
        decayInput: try #require(Fp16Buffer.make(device, count: tokenCount * heads)),
        convolution: try #require(Fp16Buffer.make(device, count: tokenCount * qkvWidth)),
        query: try #require(Fp16Buffer.make(device, count: tokenCount * keyWidth)),
        key: try #require(Fp16Buffer.make(device, count: tokenCount * keyWidth)),
        value: try #require(Fp16Buffer.make(device, count: tokenCount * valueWidth)),
        decay: try #require(device.makeBuffer(
            length: tokenCount * heads * MemoryLayout<Float>.stride,
            options: .storageModeShared)),
        beta: try #require(device.makeBuffer(
            length: tokenCount * heads * MemoryLayout<Float>.stride,
            options: .storageModeShared)),
        recurrent: try #require(Fp16Buffer.make(device, count: tokenCount * valueWidth)),
        normalized: try #require(Fp16Buffer.make(device, count: tokenCount * valueWidth)),
        projectionFloat: try #require(device.makeBuffer(
            length: tokenCount * valueWidth * MemoryLayout<Float>.stride,
            options: .storageModeShared)))
}
