#include <metal_stdlib>
using namespace metal;

kernel void qwen38_ple_affine_q4_group32_projection(
    device const uchar* weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* input [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& output_width [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint group_size = 32u;
    constexpr uint rows_per_threadgroup = 8u;
    const uint row = threadgroup_position.x * rows_per_threadgroup + simd_group;
    const uint token = threadgroup_position.y;
    if (row >= output_width || token >= token_count) return;

    const uint group_count = input_width / group_size;
    const uint row_byte_count = input_width / 2u;
    device const uchar* row_weights = weights + row * row_byte_count;
    device const bfloat* row_scales = scales + row * group_count;
    device const bfloat* row_biases = biases + row * group_count;
    device const half* token_input = input + token * input_width;

    float accumulator = 0.0f;
    for (uint group = 0; group < group_count; ++group) {
        if (lane < group_size / 2u) {
            const uchar packed = row_weights[group * (group_size / 2u) + lane];
            const float x0 = float(token_input[group * group_size + lane * 2u]);
            const float x1 = float(token_input[group * group_size + lane * 2u + 1u]);
            float dot = float(packed & 0x0Fu) * x0;
            dot = fma(float(packed >> 4), x1, dot);
            accumulator = fma(float(row_scales[group]), dot, accumulator);
            accumulator = fma(float(row_biases[group]), x0 + x1, accumulator);
        }
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0) {
        output[token * output_width + row] = half(accumulator);
    }
}

kernel void qwen38_ple_gate(
    device const half* normalized_key [[buffer(0)]],
    device const half* normalized_query [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& stream_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= stream_count || gid.y >= token_count) return;

    const uint stream_base = (gid.y * stream_count + gid.x) * hidden_size;
    const uint value_base = gid.y * hidden_size;
    float gate = 0.0f;
    for (uint feature = 0; feature < hidden_size; ++feature) {
        gate = fma(
            float(normalized_key[stream_base + feature]),
            float(normalized_query[stream_base + feature]),
            gate);
    }
    gate /= sqrt(float(hidden_size));
    const float transformed = gate == 0.0f
        ? 0.0f
        : copysign(sqrt(max(abs(gate), 1.0e-6f)), gate);
    const float weight = 1.0f / (1.0f + exp(-transformed));
    for (uint feature = 0; feature < hidden_size; ++feature) {
        output[stream_base + feature] = half(weight * float(value[value_base + feature]));
    }
}

kernel void qwen38_ple_residual_merge(
    device const half* gated_value [[buffer(0)]],
    device const half* convolution [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = gated_value[index] + convolution[index];
}

kernel void qwen38_ple_dilated_causal_conv(
    device const half* input [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device half* state [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    constant uint& dilation [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= channels || kernel_size < 2u || dilation == 0u) return;

    const uint history_length = (kernel_size - 1u) * dilation;
    const uint state_base = channel * history_length;
    const uint weight_base = channel * kernel_size;
    for (uint token = 0; token < token_count; ++token) {
        float value = 0.0f;
        for (uint tap = 0; tap + 1u < kernel_size; ++tap) {
            value = fma(
                float(weights[weight_base + tap]),
                float(state[state_base + tap * dilation]),
                value);
        }
        value = fma(
            float(weights[weight_base + kernel_size - 1u]),
            float(input[token * channels + channel]),
            value);

        for (uint index = 0; index + 1u < history_length; ++index) {
            state[state_base + index] = state[state_base + index + 1u];
        }
        state[state_base + history_length - 1u] = input[token * channels + channel];
        output[token * channels + channel] = half(value / (1.0f + exp(-value)));
    }
}
