import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct RepackPlannerTests {
    @Test
    func canonicalizesMtpSidecarNames() {
        let name = RepackPlanner.canonicalSourceTensorName(
            "mtp.layers.0.self_attn.q_proj.weight")

        #expect(name == "language_model.mtp.layers.0.self_attn.q_proj.weight")
        #expect(RepackPlanner.classify(
            name,
            numLayers: 48,
            modelFamily: "qwen4_exp_text") == .lmResident)
    }

    @Test
    func assignsMtpSourceQuantizationContracts() {
        let meta = metadata()

        #expect(RepackPlanner.sourceQuantSpec(
            for: "language_model.mtp.layers.0.self_attn.q_proj.weight",
            meta: meta) == QuantSpec(bits: 4, groupSize: 32))
        #expect(RepackPlanner.sourceQuantSpec(
            for: "language_model.mtp.layers.0.self_attn.k_proj.weight",
            meta: meta) == QuantSpec(bits: 4, groupSize: 32))
        #expect(RepackPlanner.sourceQuantSpec(
            for: "language_model.mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            meta: meta) == QuantSpec(bits: 8, groupSize: 64))
        #expect(RepackPlanner.sourceQuantSpec(
            for: "language_model.mtp.layers.0.mlp.gate.weight",
            meta: meta) == QuantSpec(bits: 8, groupSize: 64))
        #expect(RepackPlanner.sourceQuantSpec(
            for: "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
            meta: meta) == QuantSpec(bits: 4, groupSize: 32))
    }

    @Test
    func recognizesMtpProjectionsForBF16Q4Conversion() {
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.mtp.fc_embedding.weight"))
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.mtp.fc_hidden.weight"))
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.model.layers.0.ple.key_proj.weight"))
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.model.layers.0.ple.value_proj.weight"))
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"))
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight"))
        #expect(RepackPlanner.isBF16Qwen38Projection(
            "language_model.model.layers.0.hyper_connection_mixer.block_inject_weight.weight"))
    }

    @Test
    func keepsMtpTensorsResident() {
        let mtp = RepackPlanner.classify(
            "language_model.mtp.layers.0.self_attn.q_proj.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text")
        let vision = RepackPlanner.classify(
            "vision_tower.encoder.layers.0.attn.q_proj.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text")

        #expect(mtp == .lmResident)
        #expect(vision == .excludedMultimodal)
    }

    @Test
    func keepsQwen38RouterResident() {
        let router = RepackPlanner.classify(
            "language_model.model.layers.0.mlp.gate.weight",
            numLayers: 48,
            modelFamily: "qwen4_exp_text")

        #expect(router == .lmResident)
    }

    @Test
    func qwen36SharedRouterGateUsesAffine8() {
        let qwen36Gate = RepackPlanner.residentOutputQuantSpec(
            for: "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            modelFamily: "qwen3_5_moe_text")
        let qwen36ExpertGate = RepackPlanner.residentOutputQuantSpec(
            for: "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            modelFamily: "qwen3_5_moe_text")
        let qwen36Router = RepackPlanner.residentOutputQuantSpec(
            for: "language_model.model.layers.0.mlp.gate.weight",
            modelFamily: "qwen3_5_moe_text")
        let qwen38Gate = RepackPlanner.residentOutputQuantSpec(
            for: "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            modelFamily: "qwen4_exp_text")

        #expect(qwen36Gate == QuantSpec(bits: 8, groupSize: 64))
        #expect(qwen36ExpertGate == QuantSpec(bits: 4, groupSize: 32))
        #expect(qwen36Router == QuantSpec(bits: 4, groupSize: 32))
        #expect(qwen38Gate == QuantSpec(bits: 4, groupSize: 32))
    }

    @Test
    func qwen38CompatibilityAcceptsQ4Group32() {
        let issues = RepackPlanner.qwen38RuntimeCompatibilityIssues(
            meta: metadata(),
            tensors: [sourceTensor(name: "language_model.model.layers.0.self_attn.q_proj.weight")])

        #expect(issues.isEmpty)
    }

    @Test
    func qwen38CompatibilityIgnoresMtpQuantizationOverrides() {
        let meta = metadata(overrides: [
            "language_model.mtp.layers.0.self_attn.q_proj": QuantSpec(bits: 8, groupSize: 128)
        ])
        let issues = RepackPlanner.qwen38RuntimeCompatibilityIssues(
            meta: meta,
            tensors: [sourceTensor(name: "language_model.mtp.layers.0.self_attn.q_proj.weight")])

        #expect(issues.isEmpty)
    }

    @Test
    func qwen38CompatibilityAcceptsConvertibleQuantizationLayouts() {
        let meta = metadata(overrides: [
            "language_model.model.layers.0.linear_attn.out_proj": QuantSpec(bits: 8, groupSize: 64)
        ])
        let tensors = [
            sourceTensor(
                name: "language_model.model.layers.0.linear_attn.out_proj.weight",
                shape: [8, 16])
        ]

        let issues = RepackPlanner.qwen38RuntimeCompatibilityIssues(
            meta: meta, tensors: tensors)

        #expect(issues.isEmpty)
    }

    @Test
    func qwen38CompatibilityRejectsUnsupportedQuantizationLayouts() {
        let meta = metadata(overrides: [
            "language_model.model.layers.0.linear_attn.out_proj": QuantSpec(bits: 5, groupSize: 128),
            "language_model.model.layers.1.shared_expert_gate": QuantSpec(bits: 8, groupSize: 64),
            "language_model.model.layers.2.shared_expert_gate": QuantSpec(bits: 8, groupSize: 128)
        ])
        let tensors = [
            sourceTensor(name: "language_model.model.layers.0.linear_attn.out_proj.weight"),
            sourceTensor(name: "language_model.model.layers.1.shared_expert_gate.weight"),
            sourceTensor(name: "language_model.model.layers.2.shared_expert_gate.weight")
        ]

        let issues = RepackPlanner.qwen38RuntimeCompatibilityIssues(
            meta: meta,
            tensors: tensors)

        #expect(issues == [
            "language_model.model.layers.0.linear_attn.out_proj.weight: source 5-bit/group-128; runtime supports 4-bit/group-32",
            "language_model.model.layers.1.shared_expert_gate.weight: source 8-bit/group-64; runtime supports 4-bit/group-32",
            "language_model.model.layers.2.shared_expert_gate.weight: source 8-bit/group-128; runtime supports 4-bit/group-32"
        ])
    }

    private func sourceTensor(
        name: String, shape: [UInt64] = [8, 8]) -> SourceTensor {
        SourceTensor(
            name: name,
            shardPath: "model.safetensors",
            dtype: .u32,
            shape: shape,
            absoluteOffset: 0,
            sizeBytes: UInt64(shape.reduce(1, *)) * 4)
    }

    private func metadata(overrides: [String: QuantSpec] = [:]) -> IndexLoader.SourceMetadata {
        IndexLoader.SourceMetadata(
            indexPath: "model.safetensors.index.json",
            configPath: "config.json",
            indexSha256Hex: "test",
            weightMap: [:],
            baseBits: 4,
            baseGroupSize: 32,
            baseMode: "affine",
            bitsOverrides: overrides,
            shardFilenames: [])
    }
}
