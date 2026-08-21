#include <metal_stdlib>
using namespace metal;

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same Gemma activation without compiling a private shader module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(gelu_pytorch_tanh(g) * u);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half((g / (1.0f + exp(-g))) * u);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_mul_fp16_block(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     token_count [[buffer(3)]],
    constant uint&     feature_count [[buffer(4)]],
    uint2              gid [[thread_position_in_grid]]
) {
    if (gid.y >= token_count || gid.x >= feature_count) return;
    const uint index = gid.y * feature_count + gid.x;
    const float g = float(gate[index]);
    const float u = float(up[index]);
    out[index] = half((g / (1.0f + exp(-g))) * u);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_combine_shared_silu(
    device const half* gateLogit [[buffer(0)]],
    device const half* shared    [[buffer(1)]],
    device const half* routed    [[buffer(2)]],
    device half*       out       [[buffer(3)]],
    constant uint&     count      [[buffer(4)]],
    uint               tid        [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float logit = float(gateLogit[0]);
    const float gate = logit >= 0.0f
        ? 1.0f / (1.0f + exp(-logit))
        : exp(logit) / (1.0f + exp(logit));
    out[tid] = half(float(routed[tid]) + gate * float(shared[tid]));
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_combine_shared_silu_block(
    device const half* gateLogit [[buffer(0)]],
    device const half* shared    [[buffer(1)]],
    device const half* routed    [[buffer(2)]],
    device half* out             [[buffer(3)]],
    constant uint& rows          [[buffer(4)]],
    constant uint& dimension     [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    if (gid.y >= rows || gid.x >= dimension) return;
    const float logit = float(gateLogit[gid.y]);
    const float gate = logit >= 0.0f
        ? 1.0f / (1.0f + exp(-logit))
        : exp(logit) / (1.0f + exp(logit));
    const uint index = gid.y * dimension + gid.x;
    out[index] = half(float(routed[index]) + gate * float(shared[index]));
}
