import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct RepackPlannerTests {
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
