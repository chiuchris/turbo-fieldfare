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
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= stream_count || gid.y >= token_count) return;
    const uint stream_base = (gid.y * stream_count + gid.x) * hidden_size;
    const uint weight_base = gid.x * hidden_size;
    float sum = 0.0f;
    for (uint feature = 0; feature < hidden_size; ++feature) {
        const float value = float(input[stream_base + feature]);
        sum = fma(value, value, sum);
    }
    const float inverse = rsqrt(sum / float(hidden_size) + epsilon);
    for (uint feature = 0; feature < hidden_size; ++feature) {
        const float scale = 1.0f + float(weight[weight_base + feature]);
        output[stream_base + feature] = half(
            float(input[stream_base + feature]) * inverse * scale);
    }
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
