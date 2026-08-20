import Foundation

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    let modelFamily: String
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String
    let linearNumKeyHeads: Int
    let linearNumValueHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelDim: Int

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        let rawModelFamily = (tc["model_type"] as? String)
            ?? (root["model_type"] as? String)
            ?? (root["architectures"] as? [String])?.first
            ?? "unknown"
        let modelFamily = rawModelFamily == "qwen3_5_moe"
            ? "qwen3_5_moe_text" : rawModelFamily
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func optionalI(_ k: String) -> Int? {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue
        }
        func requiredOrDefault(_ keys: [String], _ fallback: Int) throws -> Int {
            for key in keys {
                if let value = optionalI(key) { return value }
            }
            return fallback
        }
        let layerTypes: [String]
        if let configured = tc["layer_types"] as? [String], !configured.isEmpty {
            layerTypes = configured
        } else if let interval = optionalI("full_attention_interval"), interval > 0 {
            let layerCount = try requiredOrDefault(["num_hidden_layers"], 0)
            layerTypes = (0..<layerCount).map {
                ($0 + 1) % interval == 0 ? "full_attention" : "linear_attention"
            }
        } else {
            layerTypes = []
        }
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue
            ?? (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue
            ?? (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue
            ?? fullTheta
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "silu"
        return ArchInfo(
            modelFamily: modelFamily,
            hiddenSize: try i("hidden_size"),
            intermediateSize: try requiredOrDefault(
                ["shared_expert_intermediate_size", "intermediate_size"], 1),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try requiredOrDefault(
                ["num_global_key_value_heads", "num_key_value_heads"], 1),
            headDim: try i("head_dim"),
            fullHeadDim: try requiredOrDefault(["global_head_dim", "head_dim"], 1),
            vocabSize: try i("vocab_size"),
            slidingWindow: try requiredOrDefault(["sliding_window"], 1),
            finalLogitSoftcap: (try? d("final_logit_softcapping")) ?? 0,
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            linearNumKeyHeads: try requiredOrDefault(["linear_num_key_heads"], 16),
            linearNumValueHeads: try requiredOrDefault(["linear_num_value_heads"], 32),
            linearKeyHeadDim: try requiredOrDefault(["linear_key_head_dim"], 128),
            linearValueHeadDim: try requiredOrDefault(["linear_value_head_dim"], 128),
            linearConvKernelDim: try requiredOrDefault(["linear_conv_kernel_dim"], 4))
    }
}
