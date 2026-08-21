#include <metal_stdlib>
using namespace metal;

kernel void qwen_split_query_gate(
    device const half* projection [[buffer(0)]],
    device half* query [[buffer(1)]],
    device half* gate [[buffer(2)]],
    constant uint& head_dimension [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint width = head_count * head_dimension;
    if (gid >= width) return;
    const uint head = gid / head_dimension;
    const uint offset = gid % head_dimension;
    const uint source = head * head_dimension * 2u + offset;
    query[gid] = projection[source];
    gate[gid] = projection[source + head_dimension];
}

kernel void qwen_prefill_split_query_gate(
    device const half* projection [[buffer(0)]],
    device half* query [[buffer(1)]],
    device half* gate [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& head_dimension [[buffer(4)]],
    constant uint& head_count [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint width = head_count * head_dimension;
    if (gid.x >= width || gid.y >= token_count) return;
    const uint row = gid.y * width;
    const uint head = gid.x / head_dimension;
    const uint offset = gid.x % head_dimension;
    const uint source = gid.y * width * 2u + head * head_dimension * 2u + offset;
    query[row + gid.x] = projection[source];
    gate[row + gid.x] = projection[source + head_dimension];
}

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

kernel void qwen_prefill_attention_output_gate(
    device const half* attention [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= count || gid.y >= token_count) return;
    const uint index = gid.y * count + gid.x;
    const float gateValue = float(gate[index]);
    const float sigmoid = 1.0f / (1.0f + exp(-gateValue));
    output[index] = half(float(attention[index]) * sigmoid);
}