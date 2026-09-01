import Metal
import TurboFieldfareFormat

struct Qwen38DeltaNetGeometry: Equatable {
    let hiddenSize: UInt32
    let keyHeads: UInt32
    let valueHeads: UInt32
    let keyHeadDimension: UInt32
    let valueHeadDimension: UInt32
    let convolutionKernel: UInt32

    static let qwen = Qwen38DeltaNetGeometry(
        hiddenSize: 2_560,
        keyHeads: 16,
        valueHeads: 48,
        keyHeadDimension: 128,
        valueHeadDimension: 128,
        convolutionKernel: 4)

    var keyWidth: UInt32 { keyHeads * keyHeadDimension }
    var valueWidth: UInt32 { valueHeads * valueHeadDimension }
    var qkvWidth: UInt32 { keyWidth * 2 + valueWidth }
}

struct Qwen38DeltaNetWeights {
    let qkv: Qwen38PLEQuantizedProjection
    let gate: Qwen38PLEQuantizedProjection
    let beta: Qwen38PLEQuantizedProjection
    let decay: Qwen38PLEQuantizedProjection
    let convolution: TensorView
    let decayLog: TensorView
    let timeBias: TensorView
    let norm: TensorView
    let output: Qwen38PLEQuantizedProjection

    init(model: Model,
         layer: Int,
         geometry: Qwen38DeltaNetGeometry = .qwen) throws {
        try self.init(
            qkv: model.qwen38LinearAttention(
                layer: layer, tensor: .queryKeyValueProjection),
            gate: model.qwen38LinearAttention(
                layer: layer, tensor: .gateProjection),
            beta: model.qwen38LinearAttention(
                layer: layer, tensor: .bProjection),
            decay: model.qwen38LinearAttention(
                layer: layer, tensor: .aProjection),
            convolution: model.qwen38LinearAttention(
                layer: layer, tensor: .convolution),
            decayLog: model.qwen38LinearAttention(
                layer: layer, tensor: .decayLog),
            timeBias: model.qwen38LinearAttention(
                layer: layer, tensor: .timeBias),
            norm: model.qwen38LinearAttention(
                layer: layer, tensor: .norm),
            output: model.qwen38LinearAttention(
                layer: layer, tensor: .outputProjection),
            geometry: geometry)
    }

    init(qkv: TensorView,
         gate: TensorView,
         beta: TensorView,
         decay: TensorView,
         convolution: TensorView,
         decayLog: TensorView,
         timeBias: TensorView,
         norm: TensorView,
         output: TensorView,
         geometry: Qwen38DeltaNetGeometry = .qwen) throws {
        self.qkv = try Self.projection(
            qkv, rows: geometry.qkvWidth, columns: geometry.hiddenSize)
        self.gate = try Self.projection(
            gate, rows: geometry.valueWidth, columns: geometry.hiddenSize)
        self.beta = try Self.projection(
            beta, rows: geometry.valueHeads, columns: geometry.hiddenSize)
        self.decay = try Self.projection(
            decay, rows: geometry.valueHeads, columns: geometry.hiddenSize)
        self.output = try Self.projection(
            output, rows: geometry.hiddenSize, columns: geometry.valueWidth)
        try Self.validateUnquantized(
            convolution,
            dtype: .bf16,
            shape: (geometry.qkvWidth, geometry.convolutionKernel, 1, 0))
        try Self.validateUnquantized(
            decayLog,
            dtype: .bf16,
            shape: (geometry.valueHeads, 0, 0, 0))
        try Self.validateUnquantized(
            timeBias,
            dtype: .bf16,
            shape: (geometry.valueHeads, 0, 0, 0))
        try Self.validateUnquantized(
            norm,
            dtype: .bf16,
            shape: (geometry.valueHeadDimension, 0, 0, 0))
        self.convolution = convolution
        self.decayLog = decayLog
        self.timeBias = timeBias
        self.norm = norm
    }

    private static func projection(
        _ view: TensorView,
        rows: UInt32,
        columns: UInt32
    ) throws -> Qwen38PLEQuantizedProjection {
        let groupSize: UInt32 = 32
        let weightBytes = UInt64(rows) * UInt64(columns) / 2
        let auxiliaryBytes = UInt64(rows) * UInt64(columns / groupSize)
            * UInt64(MemoryLayout<UInt16>.stride)
        guard columns.isMultiple(of: groupSize),
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
                detail: "Qwen3.8 DeltaNet affine-Q4 metadata mismatch")
        }
        return Qwen38PLEQuantizedProjection(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset))
    }

    private static func validateUnquantized(
        _ view: TensorView,
        dtype: GTurboFormatV1.DType,
        shape: (UInt32, UInt32, UInt32, UInt32)
    ) throws {
        let elementBytes: UInt64 = dtype == .fp32 ? 4 : 2
        let dimensions = [shape.0, shape.1, shape.2, shape.3].filter { $0 > 0 }
        let expectedBytes = dimensions.reduce(UInt64(1)) {
            $0 * UInt64($1)
        } * elementBytes
        guard view.dtype == dtype.rawValue,
              view.shape.0 == shape.0, view.shape.1 == shape.1,
              view.shape.2 == shape.2, view.shape.3 == shape.3,
              view.length == expectedBytes,
              view.scaleLength == 0, view.biasLength == 0,
              view.offset.isMultiple(of: elementBytes) else {
            throw ModelError.indexCorrupt(
                detail: "Qwen3.8 DeltaNet unquantized metadata mismatch")
        }
    }
}

struct Qwen38DeltaNetScratch {
    let qkv: MTLBuffer
    let gate: MTLBuffer
    let betaInput: MTLBuffer
    let decayInput: MTLBuffer
    let convolution: MTLBuffer
    let query: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let decay: MTLBuffer
    let beta: MTLBuffer
    let recurrent: MTLBuffer
    let normalized: MTLBuffer
}

enum Qwen38DeltaNetProjectionGroup {
    case qkvGate
    case betaDecay
}

final class Qwen38DeltaNetDecoder {
    private let projection: Qwen38PLEProjection
    private let deltaNet: QwenGatedDeltaNet
    private let elementwise: QwenElementwise
    let geometry: Qwen38DeltaNetGeometry

    init(context: MetalContext,
         geometry: Qwen38DeltaNetGeometry = .qwen) throws {
        self.geometry = geometry
        self.projection = try Qwen38PLEProjection(context: context)
        self.deltaNet = try QwenGatedDeltaNet(context: context)
        self.elementwise = try QwenElementwise(context: context)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                state: Qwen38DecoderAttentionState,
                weights: Qwen38DeltaNetWeights,
                input: MTLBuffer,
                scratch: Qwen38DeltaNetScratch,
                output: MTLBuffer,
                epsilon: Float) throws {
        try encodeBatch(
            commandBuffer: commandBuffer,
            state: state,
            weights: weights,
            input: input,
            scratch: scratch,
            output: output,
            tokenCount: 1,
            epsilon: epsilon)
    }

    func encodeProjectionGroup(commandBuffer: MTLCommandBuffer,
                                group: Qwen38DeltaNetProjectionGroup,
                                weights: Qwen38DeltaNetWeights,
                                input: MTLBuffer,
                                scratch: Qwen38DeltaNetScratch,
                                tokenCount: UInt32 = 1) {
        switch group {
        case .qkvGate:
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.qkv,
                input: input,
                output: scratch.qkv,
                tokenCount: tokenCount,
                outputWidth: geometry.qkvWidth,
                inputWidth: geometry.hiddenSize)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.gate,
                input: input,
                output: scratch.gate,
                tokenCount: tokenCount,
                outputWidth: geometry.valueWidth,
                inputWidth: geometry.hiddenSize)
        case .betaDecay:
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.beta,
                input: input,
                output: scratch.betaInput,
                tokenCount: tokenCount,
                outputWidth: geometry.valueHeads,
                inputWidth: geometry.hiddenSize)
            encodeProjection(
                commandBuffer: commandBuffer,
                weights: weights.decay,
                input: input,
                output: scratch.decayInput,
                tokenCount: tokenCount,
                outputWidth: geometry.valueHeads,
                inputWidth: geometry.hiddenSize)
        }
    }

    func encodeBatch(commandBuffer: MTLCommandBuffer,
                     state: Qwen38DecoderAttentionState,
                     weights: Qwen38DeltaNetWeights,
                     input: MTLBuffer,
                     scratch: Qwen38DeltaNetScratch,
                     output: MTLBuffer,
                     tokenCount: UInt32,
                     epsilon: Float) throws {
        guard case .linear = state else {
            throw ModelError.archMismatch(
                field: "qwen38DeltaNetState",
                expected: "linear",
                actual: "sparse")
        }
        let tokenElements = Int(tokenCount)
        let hiddenElements = Int(geometry.hiddenSize)
        let keyElements = Int(geometry.keyWidth)
        let valueElements = Int(geometry.valueWidth)
        let qkvElements = Int(geometry.qkvWidth)
        let headElements = Int(geometry.valueHeads)
        let fp16Bytes = MemoryLayout<Float16>.stride
        let fp32Bytes = MemoryLayout<Float>.stride
        precondition(tokenCount > 0)
        precondition(input.length >= tokenElements * hiddenElements * fp16Bytes)
        precondition(output.length >= tokenElements * hiddenElements * fp16Bytes)
        precondition(scratch.qkv.length >= tokenElements * qkvElements * fp16Bytes)
        precondition(scratch.gate.length >= tokenElements * valueElements * fp16Bytes)
        precondition(scratch.betaInput.length >= tokenElements * headElements * fp16Bytes)
        precondition(scratch.decayInput.length >= tokenElements * headElements * fp16Bytes)
        precondition(scratch.convolution.length >= tokenElements * qkvElements * fp16Bytes)
        precondition(scratch.query.length >= tokenElements * keyElements * fp16Bytes)
        precondition(scratch.key.length >= tokenElements * keyElements * fp16Bytes)
        precondition(scratch.value.length >= tokenElements * valueElements * fp16Bytes)
        precondition(scratch.decay.length >= tokenElements * headElements * fp32Bytes)
        precondition(scratch.beta.length >= tokenElements * headElements * fp32Bytes)
        precondition(scratch.recurrent.length >= tokenElements * valueElements * fp16Bytes)
        precondition(scratch.normalized.length >= tokenElements * valueElements * fp16Bytes)

        encodeProjectionGroup(
            commandBuffer: commandBuffer,
            group: .qkvGate,
            weights: weights,
            input: input,
            scratch: scratch,
            tokenCount: tokenCount)
        encodeProjectionGroup(
            commandBuffer: commandBuffer,
            group: .betaDecay,
            weights: weights,
            input: input,
            scratch: scratch,
            tokenCount: tokenCount)
        try encodeAfterProjections(
            commandBuffer: commandBuffer,
            state: state,
            weights: weights,
            scratch: scratch,
            output: output,
            tokenCount: tokenCount,
            epsilon: epsilon)
    }

    func encodeAfterProjections(commandBuffer: MTLCommandBuffer,
                                state: Qwen38DecoderAttentionState,
                                weights: Qwen38DeltaNetWeights,
                                scratch: Qwen38DeltaNetScratch,
                                output: MTLBuffer,
                                tokenCount: UInt32,
                                epsilon: Float) throws {
        guard case .linear(let deltaState) = state else {
            throw ModelError.archMismatch(
                field: "qwen38DeltaNetState",
                expected: "linear",
                actual: "sparse")
        }
        deltaNet.encodePrefillCausalConvolution(
            commandBuffer: commandBuffer,
            input: scratch.qkv,
            weights: weights.convolution.buffer,
            weightsOffset: Int(weights.convolution.offset),
            output: scratch.convolution,
            state: deltaState,
            tokenCount: tokenCount)
        deltaNet.encodePrefillSplitQKV(
            commandBuffer: commandBuffer,
            input: scratch.convolution,
            query: scratch.query,
            key: scratch.key,
            value: scratch.value,
            tokenCount: tokenCount,
            keyWidth: geometry.keyWidth,
            valueWidth: geometry.valueWidth)
        elementwise.encodeDeltaParametersBatch(
            commandBuffer: commandBuffer,
            a: scratch.decayInput,
            betaInput: scratch.betaInput,
            aLog: weights.decayLog.buffer,
            aLogOffset: Int(weights.decayLog.offset),
            dtBias: weights.timeBias.buffer,
            dtBiasOffset: Int(weights.timeBias.offset),
            decay: scratch.decay,
            beta: scratch.beta,
            tokenCount: tokenCount,
            headCount: geometry.valueHeads)
        deltaNet.encodePrefillRecurrent(
            commandBuffer: commandBuffer,
            query: scratch.query,
            key: scratch.key,
            value: scratch.value,
            decay: scratch.decay,
            beta: scratch.beta,
            output: scratch.recurrent,
            state: deltaState,
            tokenCount: tokenCount)
        elementwise.encodeGatedNormBatch(
            commandBuffer: commandBuffer,
            input: scratch.recurrent,
            gate: scratch.gate,
            weight: weights.norm.buffer,
            weightOffset: Int(weights.norm.offset),
            output: scratch.normalized,
            tokenCount: tokenCount,
            headCount: geometry.valueHeads,
            headDimension: geometry.valueHeadDimension,
            epsilon: epsilon)
        encodeProjection(
            commandBuffer: commandBuffer,
            weights: weights.output,
            input: scratch.normalized,
            output: output,
            tokenCount: tokenCount,
            outputWidth: geometry.hiddenSize,
            inputWidth: geometry.valueWidth)
    }

    private func encodeProjection(commandBuffer: MTLCommandBuffer,
                                  weights: Qwen38PLEQuantizedProjection,
                                  input: MTLBuffer,
                                  output: MTLBuffer,
                                  tokenCount: UInt32 = 1,
                                  outputWidth: UInt32,
                                  inputWidth: UInt32) {
        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights.weights,
            weightsOffset: weights.weightsOffset,
            scales: weights.scales,
            scalesOffset: weights.scalesOffset,
            biases: weights.biases,
            biasesOffset: weights.biasesOffset,
            input: input,
            output: output,
            tokenCount: tokenCount,
            outputWidth: outputWidth,
            inputWidth: inputWidth)
    }

    private func encodeCopy(commandBuffer: MTLCommandBuffer,
                            source: MTLBuffer,
                            sourceOffset: Int,
                            destination: MTLBuffer,
                            count: UInt32) {
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else { return }
        encoder.copy(
            from: source,
            sourceOffset: sourceOffset,
            to: destination,
            destinationOffset: 0,
            size: Int(count) * MemoryLayout<Float16>.stride)
        encoder.endEncoding()
    }
}

enum Qwen38DecoderAttentionState {
    case linear(QwenGatedDeltaNetState)
    case sparse(qsa: Qwen38QSALayerState, cache: QwenFullAttentionKVCache)
}

struct Qwen38DecoderLayerWeights {
    let layer: Int
    let attention: Qwen38HyperConnectionWeights
    let mlp: Qwen38HyperConnectionWeights

    init(model: Model, layer: Int) throws {
        guard model.config.modelFamily == .qwen38FlashNextText else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: "qwen38FlashNextText",
                actual: "\(model.config.modelFamily)")
        }
        guard layer >= 0 && layer < model.config.numLayers else {
            throw ModelError.archMismatch(
                field: "layer",
                expected: "0..<\(model.config.numLayers)",
                actual: "\(layer)")
        }
        try self.init(
            layer: layer,
            attention: model.qwen38HyperConnectionWeights(
                layer: layer,
                branch: .attention),
            mlp: model.qwen38HyperConnectionWeights(
                layer: layer,
                branch: .mlp))
    }

    init(layer: Int,
         attention: Qwen38HyperConnectionWeights,
         mlp: Qwen38HyperConnectionWeights) throws {
        guard attention.blockInject != nil, mlp.blockInject != nil else {
            throw ModelError.archMismatch(
                field: "qwen38HyperConnection[\(layer)]",
                expected: "attention and MLP block injection projections",
                actual: "injection-free final mixer weights")
        }
        self.layer = layer
        self.attention = attention
        self.mlp = mlp
    }
}

struct Qwen38DecoderLayerScratch {
    let attentionHyperConnection: Qwen38HyperConnectionScratch
    let attentionInput: MTLBuffer
    let attentionOutput: MTLBuffer
    let afterAttention: MTLBuffer
    let mlpHyperConnection: Qwen38HyperConnectionScratch
    let mlpInput: MTLBuffer
    let mlpOutput: MTLBuffer
}

final class Qwen38DecoderLayerExecutor {
    typealias AttentionEncoder = (
        _ commandBuffer: MTLCommandBuffer,
        _ state: Qwen38DecoderAttentionState,
        _ input: MTLBuffer,
        _ output: MTLBuffer,
        _ tokenCount: UInt32
    ) throws -> Void

    typealias MoEEncoder = (
        _ commandBuffer: MTLCommandBuffer,
        _ input: MTLBuffer,
        _ output: MTLBuffer,
        _ tokenCount: UInt32
    ) throws -> Void

    private let hyperConnection: Qwen38HyperConnection
    let geometry: Qwen38HyperConnectionGeometry

    init(context: MetalContext,
         geometry: Qwen38HyperConnectionGeometry = .qwen) throws {
        self.geometry = geometry
        self.hyperConnection = try Qwen38HyperConnection(
            context: context,
            geometry: geometry)
    }

    func attentionState(
        layer: Int,
        runtimeState: Qwen38RuntimeState
    ) throws -> Qwen38DecoderAttentionState {
        let layerCount = ArchConfig.qwen38FlashNextText.numLayers
        guard layer >= 0 && layer < layerCount else {
            throw ModelError.archMismatch(
                field: "layer",
                expected: "0..<\(layerCount)",
                actual: "\(layer)")
        }
        if Qwen38TensorNames.hasQSA(layer: layer) {
            guard let qsa = runtimeState.qsa.state(layer: layer),
                  let cache = runtimeState.fullCache(layer: layer),
                  runtimeState.deltaState(layer: layer) == nil else {
                throw ModelError.archMismatch(
                    field: "qwen38LayerState[\(layer)]",
                    expected: "paired QSA and full-attention cache",
                    actual: "inconsistent sparse state")
            }
            return .sparse(qsa: qsa, cache: cache)
        }
        guard let delta = runtimeState.deltaState(layer: layer),
              runtimeState.fullCache(layer: layer) == nil,
              runtimeState.qsa.state(layer: layer) == nil else {
            throw ModelError.archMismatch(
                field: "qwen38LayerState[\(layer)]",
                expected: "DeltaNet state",
                actual: "inconsistent linear state")
        }
        return .linear(delta)
    }

    func encodeAttentionPrepare(
        commandBuffer: MTLCommandBuffer,
        weights: Qwen38DecoderLayerWeights,
        hyperInput: MTLBuffer,
        scratch: Qwen38DecoderLayerScratch,
        tokenCount: UInt32,
        epsilon: Float
    ) {
        hyperConnection.encodePrepare(
            commandBuffer: commandBuffer,
            hyperInput: hyperInput,
            weights: weights.attention,
            scratch: scratch.attentionHyperConnection,
            mixedInput: scratch.attentionInput,
            tokenCount: tokenCount,
            epsilon: epsilon)
    }

    func encodeAttentionInject(
        commandBuffer: MTLCommandBuffer,
        hyperInput: MTLBuffer,
        scratch: Qwen38DecoderLayerScratch,
        tokenCount: UInt32
    ) {
        hyperConnection.encodeInject(
            commandBuffer: commandBuffer,
            hyperInput: hyperInput,
            branchOutput: scratch.attentionOutput,
            injectionWeights: scratch.attentionHyperConnection.injectionWeights,
            output: scratch.afterAttention,
            tokenCount: tokenCount)
    }

    func encodeMLPPrepare(
        commandBuffer: MTLCommandBuffer,
        weights: Qwen38DecoderLayerWeights,
        scratch: Qwen38DecoderLayerScratch,
        tokenCount: UInt32,
        epsilon: Float
    ) {
        hyperConnection.encodePrepare(
            commandBuffer: commandBuffer,
            hyperInput: scratch.afterAttention,
            weights: weights.mlp,
            scratch: scratch.mlpHyperConnection,
            mixedInput: scratch.mlpInput,
            tokenCount: tokenCount,
            epsilon: epsilon)
    }

    func encodeMLPInject(
        commandBuffer: MTLCommandBuffer,
        scratch: Qwen38DecoderLayerScratch,
        output: MTLBuffer,
        tokenCount: UInt32
    ) {
        hyperConnection.encodeInject(
            commandBuffer: commandBuffer,
            hyperInput: scratch.afterAttention,
            branchOutput: scratch.mlpOutput,
            injectionWeights: scratch.mlpHyperConnection.injectionWeights,
            output: output,
            tokenCount: tokenCount)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: Qwen38DecoderLayerWeights,
        runtimeState: Qwen38RuntimeState,
        hyperInput: MTLBuffer,
        scratch: Qwen38DecoderLayerScratch,
        output: MTLBuffer,
        tokenCount: UInt32,
        epsilon: Float,
        attentionEncoder: AttentionEncoder,
        moeEncoder: MoEEncoder
    ) throws {
        precondition(tokenCount > 0)
        let state = try attentionState(
            layer: weights.layer,
            runtimeState: runtimeState)
        encodeAttentionPrepare(
            commandBuffer: commandBuffer,
            weights: weights,
            hyperInput: hyperInput,
            scratch: scratch,
            tokenCount: tokenCount,
            epsilon: epsilon)
        try attentionEncoder(
            commandBuffer,
            state,
            scratch.attentionInput,
            scratch.attentionOutput,
            tokenCount)
        encodeAttentionInject(
            commandBuffer: commandBuffer,
            hyperInput: hyperInput,
            scratch: scratch,
            tokenCount: tokenCount)
        encodeMLPPrepare(
            commandBuffer: commandBuffer,
            weights: weights,
            scratch: scratch,
            tokenCount: tokenCount,
            epsilon: epsilon)
        try moeEncoder(
            commandBuffer,
            scratch.mlpInput,
            scratch.mlpOutput,
            tokenCount)
        encodeMLPInject(
            commandBuffer: commandBuffer,
            scratch: scratch,
            output: output,
            tokenCount: tokenCount)
    }
}
