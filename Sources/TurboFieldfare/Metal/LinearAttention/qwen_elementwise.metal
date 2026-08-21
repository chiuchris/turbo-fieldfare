#include <metal_stdlib>
using namespace metal;

kernel void qwen_delta_parameters(
    device const half* a [[buffer(0)]],
    device const half* beta_input [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device float* decay [[buffer(4)]],
    device float* beta [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float timestep = a_log[index];
    decay[index] = -exp(timestep) * log(1.0f + exp(float(a[index]) + dt_bias[index]));
    beta[index] = 1.0f / (1.0f + exp(-float(beta_input[index])));
}

kernel void qwen_gated_rmsnorm(
    device const half* input [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& head_dimension [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= head_count) return;
    const uint base = head * head_dimension;
    float sum = 0.0f;
    for (uint i = 0; i < head_dimension; ++i) {
        const float value = float(input[base + i]);
        sum = fma(value, value, sum);
    }
    const float inverse = rsqrt(sum / float(head_dimension) + epsilon);
    for (uint i = 0; i < head_dimension; ++i) {
        const float normalized = float(input[base + i]) * inverse;
        const float silu = float(gate[base + i]) /
            (1.0f + exp(-float(gate[base + i])));
        output[base + i] = half(normalized * float(weight[i]) * silu);
    }
}

kernel void qwen_residual_add(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = half(float(lhs[index]) + float(rhs[index]));
}