#include <metal_stdlib>
using namespace metal;

kernel void qwen38_grouped_rmsnorm(
    device const half* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& stream_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& one_centered [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= stream_count || gid.y >= token_count) return;
    const uint stream_base = (gid.y * stream_count + gid.x) * hidden_size;
    const uint weight_base = gid.x * hidden_size;
    float partial[64] = {};
    const uint partial_size = (hidden_size + 63u) / 64u;
    for (uint feature = 0; feature < hidden_size; ++feature) {
        const float value = float(input[stream_base + feature]);
        const uint partial_index = min(feature / partial_size, 63u);
        partial[partial_index] = fma(
            value, value, partial[partial_index]);
    }
    for (uint stride = 32u; stride > 0u; stride /= 2u) {
        for (uint index = 0; index < stride; ++index) {
            partial[index] += partial[index + stride];
        }
    }
    const float sum = partial[0];
    const float inverse = rsqrt(sum / float(hidden_size) + epsilon);
    for (uint feature = 0; feature < hidden_size; ++feature) {
        const float stored = float(weight[weight_base + feature]);
        const float scale = one_centered != 0u ? 1.0f + stored : stored;
        output[stream_base + feature] = half(
            float(input[stream_base + feature]) * inverse * scale);
    }
}

kernel void qwen38_grouped_rmsnorm_float(
    device const half* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& stream_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& one_centered [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= stream_count || gid.y >= token_count) return;
    const uint stream_base = (gid.y * stream_count + gid.x) * hidden_size;
    const uint weight_base = gid.x * hidden_size;
    float partial[16] = {};
    const uint partial_size = (hidden_size + 15u) / 16u;
    for (uint feature = 0; feature < hidden_size; ++feature) {
        const float value = float(input[stream_base + feature]);
        const uint partial_index = min(feature / partial_size, 15u);
        partial[partial_index] = fma(
            value, value, partial[partial_index]);
    }
    float sum = (partial[0] + partial[1])
        + (partial[2] + partial[3]);
    sum += (partial[4] + partial[5])
        + (partial[6] + partial[7]);
    sum += (partial[8] + partial[9])
        + (partial[10] + partial[11]);
    sum += (partial[12] + partial[13])
        + (partial[14] + partial[15]);
    const float inverse = 1.0f / sqrt(sum / float(hidden_size) + epsilon);
    for (uint feature = 0; feature < hidden_size; ++feature) {
        const half normalized = half(
            float(input[stream_base + feature]) * inverse);
        const bfloat stored = weight[weight_base + feature];
        const bfloat effective = one_centered != 0u
            ? stored + bfloat(1.0f)
            : stored;
        const float scale = float(effective);
        output[stream_base + feature] = float(normalized) * scale;
    }
}

kernel void qwen38_rmsnorm(
    device const half* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= token_count) return;
    const uint base = index * width;
    float sum = 0.0f;
    for (uint feature = 0; feature < width; ++feature) {
        const float value = float(input[base + feature]);
        sum = fma(value, value, sum);
    }
    const float inverse = rsqrt(sum / float(width) + epsilon);
    for (uint feature = 0; feature < width; ++feature) {
        const float checkpointScale = float(weight[feature]);
        output[base + feature] = half(
            float(input[base + feature]) * inverse * checkpointScale);
    }
}

kernel void qwen38_zero_centered_rmsnorm(
    device const half* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= token_count) return;
    const uint base = index * width;
    float sum = 0.0f;
    for (uint feature = 0; feature < width; ++feature) {
        const float value = float(input[base + feature]);
        sum = fma(value, value, sum);
    }
    const float inverse = rsqrt(sum / float(width) + epsilon);
    for (uint feature = 0; feature < width; ++feature) {
        const float checkpointScale = 1.0f + float(weight[feature]);
        output[base + feature] = half(
            float(input[base + feature]) * inverse * checkpointScale);
    }
}

kernel void qwen38_collapse_streams(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& token_count [[buffer(2)]],
    constant uint& stream_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= hidden_size || gid.y >= token_count) return;
    const uint token_base = gid.y * stream_count * hidden_size;
    float collapsed = 0.0f;
    for (uint stream = 0; stream < stream_count; ++stream) {
        collapsed += float(input[token_base + stream * hidden_size + gid.x]);
    }
    output[gid.y * hidden_size + gid.x] = half(collapsed / float(stream_count));
}

kernel void qwen38_low_rank_silu(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& divisor [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float value = float(input[index]) / divisor;
    output[index] = half(value / (1.0f + exp(-value)));
}

kernel void qwen38_mix_streams(
    device const half* normalized [[buffer(0)]],
    device const half* mix_logits [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& stream_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= hidden_size || gid.y >= token_count) return;
    float mixed = 0.0f;
    const uint token_base = gid.y * stream_count * hidden_size;
    for (uint stream = 0; stream < stream_count; ++stream) {
        const uint index = token_base + stream * hidden_size + gid.x;
        const float gate = 1.0f / (1.0f + exp(-float(mix_logits[index])));
        mixed = fma(gate, float(normalized[index]), mixed);
    }
    output[gid.y * hidden_size + gid.x] = half(mixed / float(stream_count));
}

kernel void qwen38_mix_streams_float(
    device const float* normalized [[buffer(0)]],
    device const half* mix_logits [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& stream_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= hidden_size || gid.y >= token_count) return;
    float mixed = 0.0f;
    const uint token_base = gid.y * stream_count * hidden_size;
    for (uint stream = 0; stream < stream_count; ++stream) {
        const uint index = token_base + stream * hidden_size + gid.x;
        const float gate = 1.0f / (1.0f + exp(-float(mix_logits[index])));
        mixed = fma(gate, normalized[index], mixed);
    }
    output[gid.y * hidden_size + gid.x] = mixed / float(stream_count);
}

kernel void qwen38_mix_streams_float_half(
    device const float* normalized [[buffer(0)]],
    device const half* mix_logits [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& stream_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= hidden_size || gid.y >= token_count) return;
    float mixed = 0.0f;
    const uint token_base = gid.y * stream_count * hidden_size;
    for (uint stream = 0; stream < stream_count; ++stream) {
        const uint index = token_base + stream * hidden_size + gid.x;
        const float gate = 1.0f / (1.0f + exp(-float(mix_logits[index])));
        mixed = fma(gate, normalized[index], mixed);
    }
    output[gid.y * hidden_size + gid.x] = half(mixed / float(stream_count));
}

kernel void qwen38_injection_weights(
    device const half* logits [[buffer(0)]],
    device half* weights [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& divisor [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float value = float(logits[index]) / divisor;
    weights[index] = half(2.0f / (1.0f + exp(-value)));
}

kernel void qwen38_inject_streams(
    device const half* hyper_input [[buffer(0)]],
    device const half* block_output [[buffer(1)]],
    device const half* injection_weights [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& stream_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= hidden_size || gid.y >= stream_count || gid.z >= token_count) return;
    const uint hyper_index = (gid.z * stream_count + gid.y) * hidden_size + gid.x;
    const uint block_index = gid.z * hidden_size + gid.x;
    const uint weight_index = gid.z * stream_count + gid.y;
    output[hyper_index] = half(
        float(hyper_input[hyper_index])
        + float(block_output[block_index]) * float(injection_weights[weight_index]));
}

kernel void qwen38_repeat_streams(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& token_count [[buffer(2)]],
    constant uint& stream_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= hidden_size || gid.y >= stream_count || gid.z >= token_count) return;
    output[(gid.z * stream_count + gid.y) * hidden_size + gid.x]
        = input[gid.z * hidden_size + gid.x];
}
