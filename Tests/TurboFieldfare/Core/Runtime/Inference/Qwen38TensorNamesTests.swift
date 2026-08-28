import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite struct Qwen38TensorNamesTests {
    @Test func hyperConnectionNamesMatchPinnedCheckpoint() {
        #expect(Qwen38TensorNames.hyperConnection(
            layer: 0,
            branch: .attention,
            tensor: .blockInjectWeight
        ) == "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight")
        #expect(Qwen38TensorNames.hyperConnection(
            layer: 47,
            branch: .mlp,
            tensor: .inputMixWeightUp
        ) == "language_model.model.layers.47.mlp_hyper_connection.input_mix_weight_up.weight")
        #expect(Qwen38TensorNames.hyperConnection(
            layer: 8,
            branch: .mlp,
            tensor: .norm
        ) == "language_model.model.layers.8.mlp_hyper_connection.hc_norm.weight")
        #expect(Qwen38TensorNames.hyperConnectionMixer(tensor: .inputMixWeightDown)
            == "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight")
    }

    @Test func pleNamesAndLayerPredicateMatchPinnedCheckpoint() {
        #expect(Qwen38TensorNames.hasPLE(layer: 1))
        #expect(!Qwen38TensorNames.hasPLE(layer: 0))
        #expect(!Qwen38TensorNames.hasPLE(layer: 2))
        #expect(Qwen38TensorNames.ple(layer: 1, tensor: .keyProjection)
            == "language_model.model.layers.1.ple.key_proj.weight")
        #expect(Qwen38TensorNames.ple(layer: 1, tensor: .convolution)
            == "language_model.model.layers.1.ple.conv1d.weight")
        #expect(Qwen38TensorNames.ple(layer: 1, tensor: .layerMultipliers)
            == "language_model.model.layers.1.ple.ple_embedding.layer_multipliers")
        #expect(Qwen38TensorNames.ple(layer: 1, tensor: .ngramHeadOffsets)
            == "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets")
    }

    @Test func qsaNamesAndLayerPredicateMatchPinnedCheckpoint() {
        #expect(!Qwen38TensorNames.hasQSA(layer: 2))
        #expect(Qwen38TensorNames.hasQSA(layer: 3))
        #expect(!Qwen38TensorNames.hasQSA(layer: 4))
        #expect(Qwen38TensorNames.hasQSA(layer: 47))
        #expect(Qwen38TensorNames.qsa(layer: 3, tensor: .queryKeyProjection)
            == "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight")
        #expect(Qwen38TensorNames.qsa(layer: 3, tensor: .queryNorm)
            == "language_model.model.layers.3.self_attn.indexer.q_layernorm.weight")
        #expect(Qwen38TensorNames.qsa(layer: 3, tensor: .keyNorm)
            == "language_model.model.layers.3.self_attn.indexer.k_layernorm.weight")
    }

    @Test func attentionAndLinearAttentionNamesMatchPinnedCheckpoint() {
        #expect(Qwen38TensorNames.attention(layer: 11, tensor: .queryProjection)
            == "language_model.model.layers.11.self_attn.q_proj.weight")
        #expect(Qwen38TensorNames.attention(layer: 11, tensor: .keyNorm)
            == "language_model.model.layers.11.self_attn.k_norm.weight")
        #expect(Qwen38TensorNames.linearAttention(layer: 10, tensor: .decayLog)
            == "language_model.model.layers.10.linear_attn.A_log")
        #expect(Qwen38TensorNames.linearAttention(layer: 10, tensor: .queryKeyValueProjection)
            == "language_model.model.layers.10.linear_attn.in_proj_qkv.weight")
        #expect(Qwen38TensorNames.linearAttention(layer: 10, tensor: .timeBias)
            == "language_model.model.layers.10.linear_attn.dt_bias")
    }

    @Test func moeNamesMatchSourcePreservedV3Entries() {
        #expect(Qwen38TensorNames.moe(layer: 6, tensor: .router)
            == "language_model.model.layers.6.mlp.gate.weight")
        #expect(Qwen38TensorNames.moe(layer: 6, tensor: .sharedExpertGate)
            == "language_model.model.layers.6.mlp.shared_expert.gate_proj.weight")
        #expect(Qwen38TensorNames.moe(layer: 6, tensor: .sharedExpertMultiplier)
            == "language_model.model.layers.6.mlp.shared_expert_gate.weight")
    }

    @Test func modelAccessorsResolveSourcePreservedNames() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let base = try Model.load(
            directoryURL: directory, device: device, expecting: .gemma4Toy())
        let names = [
            Qwen38TensorNames.hyperConnection(
                layer: 0, branch: .attention, tensor: .blockInjectWeight),
            Qwen38TensorNames.hyperConnectionMixer(tensor: .inputMixWeightDown),
            Qwen38TensorNames.ple(layer: 1, tensor: .keyProjection),
            Qwen38TensorNames.qsa(layer: 3, tensor: .queryKeyProjection),
            Qwen38TensorNames.attention(layer: 3, tensor: .queryProjection),
            Qwen38TensorNames.linearAttention(layer: 2, tensor: .decayLog),
            Qwen38TensorNames.moe(layer: 6, tensor: .router),
        ]
        let model = try modelAliasingEmbedding(base, as: names)
        let expected = model.embedding
        let views = try [
            model.qwen38HyperConnection(
                layer: 0, branch: .attention, tensor: .blockInjectWeight),
            model.qwen38HyperConnectionMixer(tensor: .inputMixWeightDown),
            model.qwen38PLE(layer: 1, tensor: .keyProjection),
            model.qwen38QSA(layer: 3, tensor: .queryKeyProjection),
            model.qwen38Attention(layer: 3, tensor: .queryProjection),
            model.qwen38LinearAttention(layer: 2, tensor: .decayLog),
            model.qwen38MoE(layer: 6, tensor: .router),
        ]

        for view in views {
            #expect(view.buffer === expected.buffer)
            #expect(view.offset == expected.offset)
            #expect(view.length == expected.length)
            #expect(view.shape.0 == expected.shape.0)
            #expect(view.shape.1 == expected.shape.1)
        }
    }

    @Test func qsaStateManagerOwnsCanonicalLayerCachesAndWeights() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let base = try Model.load(
            directoryURL: directory, device: device, expecting: .gemma4Toy())
        let qsaLayers = (0..<ArchConfig.qwen38FlashNextText.numLayers)
            .filter(Qwen38TensorNames.hasQSA(layer:))
        let names = qsaLayers.flatMap { layer in
            [
                Qwen38TensorNames.qsa(layer: layer, tensor: .queryKeyProjection),
                Qwen38TensorNames.qsa(layer: layer, tensor: .queryNorm),
                Qwen38TensorNames.qsa(layer: layer, tensor: .keyNorm),
            ]
        }
        let model = try modelAliasingEmbedding(
            base,
            as: names,
            config: .qwen38FlashNextText)
        let manager = try Qwen38QSAStateManager(model: model, capacity: 3)
        let expectedView = model.embedding

        #expect(manager.geometry == .qwen)
        #expect(manager.capacity == 3)
        #expect(manager.layerCount == 12)
        for layer in 0..<model.config.numLayers {
            let state = manager.state(layer: layer)
            if Qwen38TensorNames.hasQSA(layer: layer) {
                let state = try #require(state)
                #expect(state.layer == layer)
                #expect(state.rawKeyCache.capacity == 3)
                #expect(state.rawKeyCache.count == 0)
                #expect(state.weights.projection.buffer === expectedView.buffer)
                #expect(state.weights.queryNorm.buffer === expectedView.buffer)
                #expect(state.weights.keyNorm.buffer === expectedView.buffer)
            } else {
                #expect(state == nil)
            }
        }

        let firstState = try #require(manager.state(layer: 3))
        let rawKey = try #require(device.makeBuffer(
            length: firstState.rawKeyCache.tokenBytes,
            options: .storageModeShared))
        var position: UInt32 = 9
        let positionBuffer = try #require(device.makeBuffer(
            bytes: &position,
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        firstState.rawKeyCache.append(
            commandBuffer: commandBuffer,
            rawKey: rawKey,
            position: positionBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(firstState.rawKeyCache.count == 1)

        manager.reset()
        #expect(firstState.rawKeyCache.count == 0)
    }

    @Test func runtimeStateOwnsAndResetsEveryQwen38StateDomain() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let base = try Model.load(
            directoryURL: directory, device: device, expecting: .gemma4Toy())
        let qsaLayers = (0..<ArchConfig.qwen38FlashNextText.numLayers)
            .filter(Qwen38TensorNames.hasQSA(layer:))
        let names = qsaLayers.flatMap { layer in
            [
                Qwen38TensorNames.qsa(layer: layer, tensor: .queryKeyProjection),
                Qwen38TensorNames.qsa(layer: layer, tensor: .queryNorm),
                Qwen38TensorNames.qsa(layer: layer, tensor: .keyNorm),
            ]
        }
        let model = try modelAliasingEmbedding(
            base,
            as: names,
            config: .qwen38FlashNextText)
        let runtimeState = try Qwen38RuntimeState(model: model, maxContext: 3)

        #expect(runtimeState.maxContext == 3)
        #expect(runtimeState.linearLayerCount == 36)
        #expect(runtimeState.sparseLayerCount == 12)
        #expect(runtimeState.qsa.layerCount == 12)
        #expect(runtimeState.deltaGeometry == QwenGatedDeltaNetGeometry(
            keyHeads: 16,
            valueHeads: 48,
            keyHeadDim: 128,
            valueHeadDim: 128,
            convolutionKernel: 4))
        #expect(runtimeState.fullAttentionGeometry == QwenFullAttentionGeometry(
            queryHeads: 24,
            keyValueHeads: 2,
            headDimension: 256,
            rotaryDimension: 64,
            ropeTheta: 10_000_000))
        #expect(runtimeState.pleConvolution.channels == 2_560)
        #expect(runtimeState.pleConvolution.kernelSize == 4)

        for layer in 0..<model.config.numLayers {
            if Qwen38TensorNames.hasQSA(layer: layer) {
                #expect(runtimeState.deltaState(layer: layer) == nil)
                #expect(runtimeState.fullCache(layer: layer) != nil)
                #expect(runtimeState.qsa.state(layer: layer) != nil)
            } else {
                #expect(runtimeState.deltaState(layer: layer) != nil)
                #expect(runtimeState.fullCache(layer: layer) == nil)
                #expect(runtimeState.qsa.state(layer: layer) == nil)
            }
        }

        let firstDelta = try #require(runtimeState.deltaState(layer: 0))
        let secondDelta = try #require(runtimeState.deltaState(layer: 1))
        #expect(firstDelta.recurrentBuffer !== secondDelta.recurrentBuffer)
        firstDelta.recurrentBuffer.contents().assumingMemoryBound(to: UInt8.self)[0] = 0x7f
        firstDelta.convolutionBuffer.contents().assumingMemoryBound(to: UInt8.self)[0] = 0x7f
        runtimeState.pleConvolution.buffer.contents()
            .assumingMemoryBound(to: UInt8.self)[0] = 0x7f

        let fullCache = try #require(runtimeState.fullCache(layer: 3))
        let keyValueBytes = runtimeState.fullAttentionGeometry.keyValueWidth
            * MemoryLayout<Float16>.stride
        let key = try #require(device.makeBuffer(
            length: keyValueBytes,
            options: .storageModeShared))
        let value = try #require(device.makeBuffer(
            length: keyValueBytes,
            options: .storageModeShared))
        let qsaState = try #require(runtimeState.qsa.state(layer: 3))
        let rawKey = try #require(device.makeBuffer(
            length: Int(runtimeState.qsa.geometry.rawKeyWidth)
                * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        var position: UInt32 = 9
        let positionBuffer = try #require(device.makeBuffer(
            bytes: &position,
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        fullCache.append(commandBuffer: commandBuffer, key: key, value: value)
        qsaState.rawKeyCache.append(
            commandBuffer: commandBuffer,
            rawKey: rawKey,
            position: positionBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(fullCache.count == 1)
        #expect(qsaState.rawKeyCache.count == 1)

        runtimeState.reset()

        #expect(firstDelta.recurrentBuffer.contents()
            .assumingMemoryBound(to: UInt8.self)[0] == 0)
        #expect(firstDelta.convolutionBuffer.contents()
            .assumingMemoryBound(to: UInt8.self)[0] == 0)
        #expect(runtimeState.pleConvolution.buffer.contents()
            .assumingMemoryBound(to: UInt8.self)[0] == 0)
        #expect(fullCache.count == 0)
        #expect(qsaState.rawKeyCache.count == 0)
    }

    @Test func modelAccessorPreservesMissingTensorName() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: directory, device: device, expecting: .gemma4Toy())
        let expected = Qwen38TensorNames.qsa(layer: 3, tensor: .queryNorm)

        #expect {
            _ = try model.qwen38QSA(layer: 3, tensor: .queryNorm)
        } throws: { error in
            if case ModelError.tensorNotFound(let name) = error {
                return name == expected
            }
            return false
        }
    }

    private func modelAliasingEmbedding(
        _ model: Model,
        as names: [String],
        config: ArchConfig? = nil
    ) throws -> Model {
        let source = try #require(
            model.residentIndex.entries["language_model.model.embed_tokens.weight"])
        var entries = model.residentIndex.entries
        for name in names {
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
        return Model(
            device: model.device,
            config: config ?? model.config,
            streamingMode: model.streamingMode,
            expertCachePolicy: model.expertCachePolicy,
            integrityPolicy: model.integrityPolicy,
            residentBuffer: model.residentBuffer,
            residentIndex: ResidentIndex(header: model.residentIndex.header, entries: entries),
            packedExpertsLayout: model.packedExpertsLayout,
            manifest: model.manifest,
            directoryURL: model.directoryURL,
            modelDirectory: model.modelDirectory,
            trustedInstallReceipt: model.trustedInstallReceipt)
    }
}
