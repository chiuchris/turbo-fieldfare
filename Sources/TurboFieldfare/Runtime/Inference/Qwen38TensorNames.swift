enum Qwen38TensorNames {
    enum HyperConnectionBranch: String {
        case attention = "attn_hyper_connection"
        case mlp = "mlp_hyper_connection"
    }

    enum HyperConnectionTensor: String {
        case norm = "hc_norm.weight"
        case inputMixWeightDown = "input_mix_weight_down.weight"
        case inputMixWeightUp = "input_mix_weight_up.weight"
        case blockInjectWeight = "block_inject_weight.weight"
    }

    enum PLETensor: String {
        case keyProjection = "key_proj.weight"
        case valueProjection = "value_proj.weight"
        case convolution = "conv1d.weight"
        case keyNorm = "norm_key.weight"
        case queryNorm = "norm_query.weight"
        case convolutionNorm = "norm_conv.weight"
        case layerMultipliers = "ple_embedding.layer_multipliers"
        case ngramHeadOffsets = "ple_embedding.ngram_heads_offsets"
        case ngramHeadVocabSizes = "ple_embedding.ngram_heads_vocab_sizes"
    }

    enum QSATensor: String {
        case queryKeyProjection = "index_qk_proj.weight"
        case queryNorm = "q_layernorm.weight"
        case keyNorm = "k_layernorm.weight"
    }

    enum AttentionTensor: String {
        case queryProjection = "q_proj.weight"
        case keyProjection = "k_proj.weight"
        case valueProjection = "v_proj.weight"
        case outputProjection = "o_proj.weight"
        case queryNorm = "q_norm.weight"
        case keyNorm = "k_norm.weight"
    }

    enum LinearAttentionTensor: String {
        case decayLog = "A_log"
        case convolution = "conv1d.weight"
        case timeBias = "dt_bias"
        case aProjection = "in_proj_a.weight"
        case bProjection = "in_proj_b.weight"
        case queryKeyValueProjection = "in_proj_qkv.weight"
        case gateProjection = "in_proj_z.weight"
        case norm = "norm.weight"
        case outputProjection = "out_proj.weight"
    }

    enum MoETensor: String {
        case router = "gate.weight"
        case sharedExpertGate = "shared_expert.gate_proj.weight"
        case sharedExpertUp = "shared_expert.up_proj.weight"
        case sharedExpertDown = "shared_expert.down_proj.weight"
        case sharedExpertMultiplier = "shared_expert_gate.weight"
    }

    static func hyperConnection(
        layer: Int,
        branch: HyperConnectionBranch,
        tensor: HyperConnectionTensor
    ) -> String {
        "\(layerPrefix(layer)).\(branch.rawValue).\(tensor.rawValue)"
    }

    static func hyperConnectionMixer(tensor: HyperConnectionTensor) -> String {
        "language_model.model.hyper_connection_mixer.\(tensor.rawValue)"
    }

    static func hasPLE(layer: Int) -> Bool {
        layer == 1
    }

    static func ple(layer: Int, tensor: PLETensor) -> String {
        "\(layerPrefix(layer)).ple.\(tensor.rawValue)"
    }

    static func hasQSA(layer: Int) -> Bool {
        layer >= 3 && layer % 4 == 3
    }

    static func qsa(layer: Int, tensor: QSATensor) -> String {
        "\(layerPrefix(layer)).self_attn.indexer.\(tensor.rawValue)"
    }

    static func attention(layer: Int, tensor: AttentionTensor) -> String {
        "\(layerPrefix(layer)).self_attn.\(tensor.rawValue)"
    }

    static func linearAttention(layer: Int, tensor: LinearAttentionTensor) -> String {
        "\(layerPrefix(layer)).linear_attn.\(tensor.rawValue)"
    }

    static func moe(layer: Int, tensor: MoETensor) -> String {
        "\(layerPrefix(layer)).mlp.\(tensor.rawValue)"
    }

    private static func layerPrefix(_ layer: Int) -> String {
        "language_model.model.layers.\(layer)"
    }
}
