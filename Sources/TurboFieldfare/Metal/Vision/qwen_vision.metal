#include <metal_stdlib>
using namespace metal;

constant constexpr uint kQwenVisionHeadDim = 72;
constant constexpr uint kQwenVisionQueryTile = 8;
constant constexpr uint kQwenVisionKeyTile = 32;
constant constexpr uint kQwenVisionThreads = 128;

kernel void qwen_bf16_to_fp16_copy(
    device const bfloat* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant uint& sourceOffsetElements [[buffer(2)]],
    constant uint& destinationOffsetElements [[buffer(3)]],
    constant uint& elementCount [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elementCount) return;
    destination[destinationOffsetElements + index] =
        half(float(source[sourceOffsetElements + index]));
}

inline float qwenVisionGELU(float value) {
    constexpr float sqrtTwoOverPi = 0.7978845608028654f;
    constexpr float cubic = 0.044715f;
    const float tanhArgument = sqrtTwoOverPi * value
        * (1.0f + cubic * value * value);
    if (tanhArgument >= 10.0f) return value;
    if (tanhArgument <= -10.0f) return 0.0f;
    return 0.5f * value * (1.0f + tanh(tanhArgument));
}

kernel void qwen_vision_patch_bias_position(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* positionTable [[buffer(1)]],
    device const bfloat* patchBias [[buffer(2)]],
    device const int2* positions [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& gridWidth [[buffer(5)]],
    constant uint& gridHeight [[buffer(6)]],
    constant uint& hiddenSize [[buffer(7)]],
    constant uint& positionGridSide [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * hiddenSize) return;
    const uint row = index / hiddenSize;
    const uint dimension = index % hiddenSize;
    const int2 position = positions[row];
    const float sourceX = gridWidth > 1u
        ? float(position.x) * float(positionGridSide - 1u) / float(gridWidth - 1u) : 0.0f;
    const float sourceY = gridHeight > 1u
        ? float(position.y) * float(positionGridSide - 1u) / float(gridHeight - 1u) : 0.0f;
    const uint x0 = min(uint(floor(sourceX)), positionGridSide - 1u);
    const uint y0 = min(uint(floor(sourceY)), positionGridSide - 1u);
    const uint x1 = min(x0 + 1u, positionGridSide - 1u);
    const uint y1 = min(y0 + 1u, positionGridSide - 1u);
    const float xWeight = sourceX - float(x0);
    const float yWeight = sourceY - float(y0);
    const uint rowStride = positionGridSide * hiddenSize;
    const float top = float(positionTable[y0 * rowStride + x0 * hiddenSize + dimension])
        * (1.0f - xWeight)
        + float(positionTable[y0 * rowStride + x1 * hiddenSize + dimension]) * xWeight;
    const float bottom = float(positionTable[y1 * rowStride + x0 * hiddenSize + dimension])
        * (1.0f - xWeight)
        + float(positionTable[y1 * rowStride + x1 * hiddenSize + dimension]) * xWeight;
    hidden[index] = bfloat(float(hidden[index]) + float(patchBias[dimension])
                           + top * (1.0f - yWeight) + bottom * yWeight);
}

kernel void qwen_vision_layer_norm(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* scale [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float sums[256];
    threadgroup float squares[256];
    float sum = 0.0f;
    float squareSum = 0.0f;
    for (uint column = lane; column < width; column += 256u) {
        const float value = float(input[row * width + column]);
        sum += value;
        squareSum = fma(value, value, squareSum);
    }
    sums[lane] = sum;
    squares[lane] = squareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = sums[0] / float(width);
    const float variance = max(squares[0] / float(width) - mean * mean, 0.0f);
    const float inverse = rsqrt(variance + epsilon);
    for (uint column = lane; column < width; column += 256u) {
        const uint index = row * width + column;
        output[index] = bfloat((float(input[index]) - mean) * inverse
                               * float(scale[column]) + float(bias[column]));
    }
}

kernel void qwen_vision_qkv_rope(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qBias [[buffer(3)]],
    device const bfloat* kBias [[buffer(4)]],
    device const bfloat* vBias [[buffer(5)]],
    device const int2* positions [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    constant float& ropeTheta [[buffer(9)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint row = vectorIndex / heads;
    const uint head = vectorIndex % heads;
    const uint base = row * heads * kQwenVisionHeadDim + head * kQwenVisionHeadDim;
    const int2 position = positions[row];
    for (uint dimension = 0; dimension < 36u; ++dimension) {
        const uint pair = dimension % 18u;
        const int coordinate = dimension < 18u ? position.y : position.x;
        const float inverseFrequency = pow(ropeTheta, -float(pair) / 18.0f);
        const float angle = float(coordinate) * inverseFrequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
        const uint first = base + dimension;
        const uint second = first + 36u;
        const uint weightFirst = head * kQwenVisionHeadDim + dimension;
        const uint weightSecond = weightFirst + 36u;
        const float qFirst = float(q[first]) + float(qBias[weightFirst]);
        const float qSecond = float(q[second]) + float(qBias[weightSecond]);
        const float kFirst = float(k[first]) + float(kBias[weightFirst]);
        const float kSecond = float(k[second]) + float(kBias[weightSecond]);
        q[first] = bfloat(qFirst * cosine - qSecond * sine);
        q[second] = bfloat(qSecond * cosine + qFirst * sine);
        k[first] = bfloat(kFirst * cosine - kSecond * sine);
        k[second] = bfloat(kSecond * cosine + kFirst * sine);
    }
    for (uint dimension = 0; dimension < kQwenVisionHeadDim; ++dimension) {
        const uint index = base + dimension;
        const uint biasIndex = head * kQwenVisionHeadDim + dimension;
        v[index] = bfloat(float(v[index]) + float(vBias[biasIndex]));
    }
}

template <uint QueryTile>
inline void qwen_vision_attention_impl(
    device const bfloat* q,
    device const bfloat* k,
    device const bfloat* v,
    device bfloat* output,
    constant uint& sequenceLength,
    constant uint& numHeads,
    uint2 group,
    uint lane,
    threadgroup bfloat* qTile,
    threadgroup bfloat* kTile,
    threadgroup bfloat* vTile,
    threadgroup float* scores,
    threadgroup float* accumulator,
    threadgroup float* oldScale,
    threadgroup float* rowSum,
    threadgroup float* rowMax) {
    const uint head = group.y;
    const uint queryBase = group.x * QueryTile;
    if (head >= numHeads) return;
    for (uint index = lane; index < QueryTile * kQwenVisionHeadDim;
         index += kQwenVisionThreads) {
        const uint localQuery = index / kQwenVisionHeadDim;
        const uint dimension = index % kQwenVisionHeadDim;
        const uint query = queryBase + localQuery;
        qTile[index] = query < sequenceLength
            ? q[(query * numHeads + head) * kQwenVisionHeadDim + dimension]
            : bfloat(0.0f);
    }
    for (uint index = lane; index < QueryTile * kQwenVisionHeadDim;
         index += kQwenVisionThreads) {
        accumulator[index] = 0.0f;
    }
    if (lane < QueryTile) {
        rowMax[lane] = -INFINITY;
        rowSum[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint keyBase = 0; keyBase < sequenceLength; keyBase += kQwenVisionKeyTile) {
        for (uint index = lane; index < kQwenVisionKeyTile * kQwenVisionHeadDim;
             index += kQwenVisionThreads) {
            const uint localKey = index / kQwenVisionHeadDim;
            const uint dimension = index % kQwenVisionHeadDim;
            const uint key = keyBase + localKey;
            const uint source = (key * numHeads + head) * kQwenVisionHeadDim + dimension;
            kTile[index] = key < sequenceLength ? k[source] : bfloat(0.0f);
            vTile[index] = key < sequenceLength ? v[source] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pair = lane; pair < QueryTile * kQwenVisionKeyTile;
             pair += kQwenVisionThreads) {
            const uint localQuery = pair / kQwenVisionKeyTile;
            const uint localKey = pair % kQwenVisionKeyTile;
            const uint query = queryBase + localQuery;
            const uint key = keyBase + localKey;
            float score = -INFINITY;
            if (query < sequenceLength && key < sequenceLength) {
                score = 0.0f;
                for (uint dimension = 0; dimension < kQwenVisionHeadDim; ++dimension) {
                    score = fma(float(qTile[localQuery * kQwenVisionHeadDim + dimension]),
                                float(kTile[localKey * kQwenVisionHeadDim + dimension]), score);
                }
                score *= 0.11785113019775792f;
            }
            scores[pair] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < QueryTile) {
            const uint query = queryBase + lane;
            float tileMax = -INFINITY;
            for (uint localKey = 0; localKey < kQwenVisionKeyTile; ++localKey) {
                tileMax = max(tileMax, scores[lane * kQwenVisionKeyTile + localKey]);
            }
            const float nextMax = max(rowMax[lane], tileMax);
            const float scale = rowMax[lane] == -INFINITY
                ? 0.0f : exp(rowMax[lane] - nextMax);
            float tileSum = 0.0f;
            for (uint localKey = 0; localKey < kQwenVisionKeyTile; ++localKey) {
                const uint key = keyBase + localKey;
                const float probability = query < sequenceLength && key < sequenceLength
                    ? exp(scores[lane * kQwenVisionKeyTile + localKey] - nextMax) : 0.0f;
                scores[lane * kQwenVisionKeyTile + localKey] = probability;
                tileSum += probability;
            }
            oldScale[lane] = scale;
            rowSum[lane] = rowSum[lane] * scale + tileSum;
            rowMax[lane] = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = lane; index < QueryTile * kQwenVisionHeadDim;
             index += kQwenVisionThreads) {
            const uint localQuery = index / kQwenVisionHeadDim;
            const uint dimension = index % kQwenVisionHeadDim;
            float value = accumulator[index] * oldScale[localQuery];
            if (dimension < kQwenVisionHeadDim) {
                for (uint localKey = 0; localKey < kQwenVisionKeyTile; ++localKey) {
                    value = fma(scores[localQuery * kQwenVisionKeyTile + localKey],
                                float(vTile[localKey * kQwenVisionHeadDim + dimension]), value);
                }
            }
            accumulator[index] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = lane; index < QueryTile * kQwenVisionHeadDim;
         index += kQwenVisionThreads) {
        const uint localQuery = index / kQwenVisionHeadDim;
        const uint dimension = index % kQwenVisionHeadDim;
        const uint query = queryBase + localQuery;
        if (query < sequenceLength) {
            output[(query * numHeads + head) * kQwenVisionHeadDim + dimension] =
                bfloat(accumulator[index] / rowSum[localQuery]);
        }
    }
}

kernel void qwen_vision_attention(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup bfloat qTile[kQwenVisionQueryTile * kQwenVisionHeadDim];
    threadgroup bfloat kTile[kQwenVisionKeyTile * kQwenVisionHeadDim];
    threadgroup bfloat vTile[kQwenVisionKeyTile * kQwenVisionHeadDim];
    threadgroup float scores[kQwenVisionQueryTile * kQwenVisionKeyTile];
    threadgroup float accumulator[kQwenVisionQueryTile * kQwenVisionHeadDim];
    threadgroup float oldScale[kQwenVisionQueryTile];
    threadgroup float rowSum[kQwenVisionQueryTile];
    threadgroup float rowMax[kQwenVisionQueryTile];
    qwen_vision_attention_impl<kQwenVisionQueryTile>(
        q, k, v, output, sequenceLength, numHeads, group, lane,
        qTile, kTile, vTile, scores, accumulator, oldScale, rowSum, rowMax);
}

kernel void qwen_vision_bias_gelu(
    device bfloat* values [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float value = float(values[index]) + float(bias[index % width]);
    values[index] = bfloat(qwenVisionGELU(value));
}

kernel void qwen_vision_residual(
    device const bfloat* state [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = bfloat(float(state[index]) + float(branch[index])
                           + float(bias[index % width]));
}

kernel void qwen_vision_bias_add(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = bfloat(float(input[index]) + float(bias[index % width]));
}

kernel void qwen_vision_merge_gather(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& hiddenSize [[buffer(3)]],
    constant uint& mergeUnit [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const uint outputWidth = hiddenSize * mergeUnit;
    const uint token = index / outputWidth;
    const uint column = index % outputWidth;
    const uint patch = column / hiddenSize;
    const uint dimension = column % hiddenSize;
    output[index] = input[(token * mergeUnit + patch) * hiddenSize + dimension];
}

kernel void qwen_text_mrope_rotate(
    device half* data [[buffer(0)]],
    device const int* positions [[buffer(1)]],
    constant uint& tokenCount [[buffer(2)]],
    constant uint& tokenStrideElements [[buffer(3)]],
    constant uint& headDimension [[buffer(4)]],
    constant uint& headCount [[buffer(5)]],
    constant uint& rotaryPairs [[buffer(6)]],
    constant float& ropeTheta [[buffer(7)]],
    constant uint& heightSection [[buffer(8)]],
    constant uint& widthSection [[buffer(9)]],
    uint3 group [[thread_position_in_grid]]) {
    const uint pair = group.x;
    const uint head = group.y;
    const uint token = group.z;
    if (pair >= rotaryPairs || head >= headCount || token >= tokenCount) return;

    uint axis = 0u;
    if (pair < heightSection * 3u && pair % 3u == 1u) {
        axis = 1u;
    } else if (pair < widthSection * 3u && pair % 3u == 2u) {
        axis = 2u;
    }
    const float coordinate = float(positions[token * 3u + axis]);
    const float inverseFrequency = pow(ropeTheta, -float(pair) / float(rotaryPairs));
    const float angle = coordinate * inverseFrequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint base = token * tokenStrideElements + head * headDimension;
    const uint firstIndex = base + pair;
    const uint secondIndex = base + pair + rotaryPairs;
    const float first = float(data[firstIndex]);
    const float second = float(data[secondIndex]);
    data[firstIndex] = half(first * cosine - second * sine);
    data[secondIndex] = half(second * cosine + first * sine);
}
