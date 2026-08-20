#include <metal_stdlib>
using namespace metal;

kernel void qwen_attention_output_gate(
    device const half* attention [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    const float gateValue = float(gate[gid]);
    const float sigmoid = 1.0f / (1.0f + exp(-gateValue));
    output[gid] = half(float(attention[gid]) * sigmoid);
}