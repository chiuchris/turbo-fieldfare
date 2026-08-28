#include <metal_stdlib>
using namespace metal;

inline float qwen38_qsa_query_value(
    device const half* projected,
    device const bfloat* weight,
    uint query,
    uint head,
    uint feature,
    uint query_stride,
    uint head_dimension,
    float inverse_rms) {
    const uint index = query * query_stride + head * head_dimension + feature;
    return float(projected[index]) * inverse_rms * (1.0f + float(weight[feature]));
}

inline float qwen38_qsa_pooled_key_value(
    device const half* raw_keys,
    device const bfloat* weight,
    uint block,
    uint feature,
    uint head_dimension,
    uint compress_ratio,
    float inverse_rms) {
    float value = 0.0f;
    const uint first_token = block * compress_ratio;
    for (uint token = 0; token < compress_ratio; ++token) {
        value += float(raw_keys[(first_token + token) * head_dimension + feature]);
    }
    value /= float(compress_ratio);
    return value * inverse_rms * (1.0f + float(weight[feature]));
}

kernel void qwen38_qsa_cache_append(
    device const half* projected_rows [[buffer(0)]],
    device const uint* source_positions [[buffer(1)]],
    device half* cached_raw_keys [[buffer(2)]],
    device uint* cached_positions [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& projection_width [[buffer(5)]],
    constant uint& query_width [[buffer(6)]],
    constant uint& raw_key_width [[buffer(7)]],
    constant uint& destination_token_offset [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= raw_key_width || gid.y >= token_count) return;
    const uint destination_token = destination_token_offset + gid.y;
    cached_raw_keys[destination_token * raw_key_width + gid.x] =
        projected_rows[gid.y * projection_width + query_width + gid.x];
    if (gid.x == 0u) {
        cached_positions[destination_token] = source_positions[gid.y];
    }
}

kernel void qwen38_qsa_block_scores(
    device const half* projected_queries [[buffer(0)]],
    device const half* raw_keys [[buffer(1)]],
    device const bfloat* query_norm [[buffer(2)]],
    device const bfloat* key_norm [[buffer(3)]],
    device const uint* query_positions [[buffer(4)]],
    device const uint* key_positions [[buffer(5)]],
    device float* output_scores [[buffer(6)]],
    constant uint& query_count [[buffer(7)]],
    constant uint& key_count [[buffer(8)]],
    constant uint& query_heads [[buffer(9)]],
    constant uint& key_value_heads [[buffer(10)]],
    constant uint& head_dimension [[buffer(11)]],
    constant uint& compress_ratio [[buffer(12)]],
    constant uint& rotary_dimension [[buffer(13)]],
    constant float& rope_theta [[buffer(14)]],
    constant float& epsilon [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint block_count = key_count / compress_ratio;
    if (gid.x >= block_count || gid.y >= query_count || key_value_heads != 1u) return;

    const uint block = gid.x;
    const uint query = gid.y;
    const uint query_stride = (query_heads + key_value_heads) * head_dimension;
    float pooled_square_sum = 0.0f;
    for (uint feature = 0; feature < head_dimension; ++feature) {
        float pooled = 0.0f;
        for (uint token = 0; token < compress_ratio; ++token) {
            pooled += float(raw_keys[(block * compress_ratio + token) * head_dimension + feature]);
        }
        pooled /= float(compress_ratio);
        pooled_square_sum = fma(pooled, pooled, pooled_square_sum);
    }
    const float key_inverse_rms = rsqrt(pooled_square_sum / float(head_dimension) + epsilon);
    const uint rotary_half = rotary_dimension / 2u;
    const float query_position = float(query_positions[query]);
    const float key_position = float(key_positions[block * compress_ratio]);
    float score = 0.0f;

    for (uint head = 0; head < query_heads; ++head) {
        float query_square_sum = 0.0f;
        const uint query_base = query * query_stride + head * head_dimension;
        for (uint feature = 0; feature < head_dimension; ++feature) {
            const float value = float(projected_queries[query_base + feature]);
            query_square_sum = fma(value, value, query_square_sum);
        }
        const float query_inverse_rms = rsqrt(
            query_square_sum / float(head_dimension) + epsilon);
        float dot = 0.0f;
        for (uint feature = 0; feature < head_dimension; ++feature) {
            float query_value = qwen38_qsa_query_value(
                projected_queries, query_norm, query, head, feature,
                query_stride, head_dimension, query_inverse_rms);
            float key_value = qwen38_qsa_pooled_key_value(
                raw_keys, key_norm, block, feature, head_dimension,
                compress_ratio, key_inverse_rms);
            if (feature < rotary_dimension) {
                const uint pair = feature < rotary_half
                    ? feature + rotary_half
                    : feature - rotary_half;
                const uint frequency = feature % rotary_half;
                const float inverse_frequency = pow(
                    rope_theta,
                    -2.0f * float(frequency) / float(rotary_dimension));
                const float query_angle = query_position * inverse_frequency;
                const float key_angle = key_position * inverse_frequency;
                const float query_pair = qwen38_qsa_query_value(
                    projected_queries, query_norm, query, head, pair,
                    query_stride, head_dimension, query_inverse_rms);
                const float key_pair = qwen38_qsa_pooled_key_value(
                    raw_keys, key_norm, block, pair, head_dimension,
                    compress_ratio, key_inverse_rms);
                if (feature < rotary_half) {
                    query_value = fma(-query_pair, sin(query_angle),
                                      query_value * cos(query_angle));
                    key_value = fma(-key_pair, sin(key_angle),
                                    key_value * cos(key_angle));
                } else {
                    query_value = fma(query_pair, sin(query_angle),
                                      query_value * cos(query_angle));
                    key_value = fma(key_pair, sin(key_angle),
                                    key_value * cos(key_angle));
                }
            }
            dot = fma(query_value, key_value, dot);
        }
        score += max(dot, 0.0f);
    }
    output_scores[query * block_count + block] = score / sqrt(float(head_dimension));
}

kernel void qwen38_qsa_selection_initialize(
    device const uint* visible_token_counts [[buffer(0)]],
    device uint* state [[buffer(1)]],
    constant uint& query_count [[buffer(2)]],
    constant uint& key_count [[buffer(3)]],
    constant uint& compress_ratio [[buffer(4)]],
    constant uint& block_top_k [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= query_count) return;
    const uint visible_tokens = min(visible_token_counts[gid], key_count);
    const uint visible_blocks = visible_tokens / compress_ratio;
    const uint base = gid * 5u;
    state[base] = 0u;
    state[base + 1u] = min(block_top_k, visible_blocks);
    state[base + 2u] = visible_blocks;
    state[base + 3u] = block_top_k;
    state[base + 4u] = visible_tokens;
}

kernel void qwen38_qsa_selection_radix(
    device const float* scores [[buffer(0)]],
    device uint* state [[buffer(1)]],
    constant uint& block_count [[buffer(2)]],
    constant uint& byte_shift [[buffer(3)]],
    uint query [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint histogram[256];
    atomic_store_explicit(&histogram[lane], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = query * 5u;
    const uint prefix = state[base];
    const uint remaining = state[base + 1u];
    const uint visible_blocks = state[base + 2u];
    const uint block_top_k = state[base + 3u];
    if (visible_blocks <= block_top_k) return;

    const uint prefix_mask = byte_shift == 24u ? 0u : ~((1u << (byte_shift + 8u)) - 1u);
    for (uint block = lane; block < visible_blocks; block += 256u) {
        const uint score_bits = as_type<uint>(scores[query * block_count + block]);
        if ((score_bits & prefix_mask) == prefix) {
            const uint digit = (score_bits >> byte_shift) & 0xffu;
            atomic_fetch_add_explicit(&histogram[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        uint greater = 0u;
        uint selected_digit = 0u;
        for (int digit = 255; digit >= 0; --digit) {
            const uint count = atomic_load_explicit(
                &histogram[uint(digit)], memory_order_relaxed);
            if (greater + count >= remaining) {
                selected_digit = uint(digit);
                break;
            }
            greater += count;
        }
        state[base] = prefix | (selected_digit << byte_shift);
        state[base + 1u] = remaining - greater;
    }
}

kernel void qwen38_qsa_selection_mask(
    device const float* scores [[buffer(0)]],
    device const uint* state [[buffer(1)]],
    device uchar* token_mask [[buffer(2)]],
    constant uint& query_count [[buffer(3)]],
    constant uint& key_count [[buffer(4)]],
    constant uint& block_count [[buffer(5)]],
    constant uint& compress_ratio [[buffer(6)]],
    uint query [[thread_position_in_grid]]) {
    if (query >= query_count) return;
    const uint base = query * 5u;
    const uint threshold = state[base];
    const uint remaining = state[base + 1u];
    const uint visible_blocks = state[base + 2u];
    const uint block_top_k = state[base + 3u];
    const uint visible_tokens = state[base + 4u];
    const uint row = query * key_count;
    for (uint token = 0; token < key_count; ++token) {
        token_mask[row + token] = 0u;
    }

    if (visible_blocks <= block_top_k) {
        for (uint token = 0; token < visible_tokens; ++token) {
            token_mask[row + token] = 1u;
        }
        return;
    }

    uint threshold_ties = 0u;
    for (uint block = 0; block < visible_blocks; ++block) {
        const uint score_bits = as_type<uint>(scores[query * block_count + block]);
        const bool above_threshold = score_bits > threshold;
        const bool selected_tie = score_bits == threshold &&
            threshold_ties++ < remaining;
        if (above_threshold || selected_tie) {
            const uint first_token = block * compress_ratio;
            for (uint token = 0; token < compress_ratio; ++token) {
                token_mask[row + first_token + token] = 1u;
            }
        }
    }
    for (uint token = visible_blocks * compress_ratio;
         token < visible_tokens;
         ++token) {
        token_mask[row + token] = 1u;
    }
}
