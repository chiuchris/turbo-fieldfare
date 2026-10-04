import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite struct QwenVisionRuntimeTests {
    @Test func configMatchesPinnedVisionTower() {
        let config = QwenVisionConfig()
        #expect(config.hiddenSize == 1_152)
        #expect(config.intermediateSize == 4_304)
        #expect(config.numLayers == 27)
        #expect(config.numHeads == 16)
        #expect(config.headDimension == 72)
        #expect(config.patchSize == 16)
        #expect(config.temporalPatchSize == 2)
        #expect(config.spatialMergeSize == 2)
        #expect(config.positionEmbeddingCount == 2_304)
        #expect(config.outputHiddenSize == 2_048)
        #expect(config.mergerInputSize == 4_608)
    }

    @Test func qwenVisionShaderModuleCompiles() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let context = try MetalContext()
        _ = try QwenVisionPrimitives(context: context)
    }

    @Test func attentionIsNonCausalAndHandlesTailRows() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let context = try MetalContext()
        let config = QwenVisionConfig()
        let rows = 3
        let elementCount = rows * config.hiddenSize
        let byteCount = elementCount * MemoryLayout<UInt16>.stride

        func sharedBuffer(_ values: [UInt16]) throws -> MTLBuffer {
            let buffer = try #require(device.makeBuffer(
                length: byteCount, options: .storageModeShared))
            values.withUnsafeBufferPointer { source in
                buffer.contents().copyMemory(
                    from: source.baseAddress!, byteCount: byteCount)
            }
            return buffer
        }

        let q = try sharedBuffer([UInt16](repeating: 0, count: elementCount))
        let k = try sharedBuffer([UInt16](repeating: 0, count: elementCount))
        let v = try sharedBuffer((0..<rows * config.hiddenSize).map {
            Quantization.bf16Bits(Float($0 / config.hiddenSize + 1))
        })
        let output = try #require(device.makeBuffer(
            length: byteCount, options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        try QwenVisionPrimitives(context: context).encodeAttention(
            commandBuffer: commandBuffer,
            q: q,
            k: k,
            v: v,
            output: output,
            rows: rows)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let values = output.contents().bindMemory(
            to: UInt16.self, capacity: elementCount)
        #expect((0..<elementCount).allSatisfy {
            Quantization.bf16ToFloat(values[$0]) == 2
        })
    }

    @Test func multimodalInputAcceptsQwenVisionFeatureWidth() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let geometry = try QwenImageGeometry(sourceWidth: 512, sourceHeight: 512)
        let hiddenSize = QwenVisionConfig().outputHiddenSize
        let featureBytes = geometry.tokenCount * hiddenSize * MemoryLayout<UInt16>.stride
        let buffer = try #require(device.makeBuffer(
            length: featureBytes, options: .storageModeShared))
        let features = QwenVisionFeatures(
            buffer: buffer,
            tokenCount: geometry.tokenCount,
            hiddenSize: hiddenSize,
            geometry: geometry,
            gpuNanoseconds: 0,
            scratchBytes: featureBytes,
            preprocessingWallNanoseconds: 0)
        let tokenCount = geometry.tokenCount + 2
        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: [Int32](repeating: 0, count: tokenCount),
            embeddingTokenIDs: [Int32](repeating: 0, count: tokenCount),
            imageTokenRange: 1..<(geometry.tokenCount + 1),
            imageFeatures: features)

        #expect(input.imageFeatures.hiddenSize == hiddenSize)
        #expect(input.imageTokenRange.count == geometry.tokenCount)
        #expect(input.imageFeatures.tokenGrid == MultimodalVisionTokenGrid(
            temporal: 1,
            height: geometry.patchGridHeight / QwenImageGeometry.mergeSize,
            width: geometry.patchGridWidth / QwenImageGeometry.mergeSize))
    }

    @Test func mropePositionsUseMergedGridAndResumeTextAfterLongestAxis() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let geometry = try QwenImageGeometry(
            sourceWidth: 96,
            sourceHeight: 64,
            minimumPixels: 1,
            maximumPixels: 1_000_000)
        let hiddenSize = QwenVisionConfig().outputHiddenSize
        let featureBytes = geometry.tokenCount * hiddenSize * MemoryLayout<UInt16>.stride
        let buffer = try #require(device.makeBuffer(
            length: featureBytes, options: .storageModeShared))
        let features = QwenVisionFeatures(
            buffer: buffer,
            tokenCount: geometry.tokenCount,
            hiddenSize: hiddenSize,
            geometry: geometry,
            gpuNanoseconds: 0,
            scratchBytes: featureBytes,
            preprocessingWallNanoseconds: 0)
        let span = MultimodalImageSpan(tokenRange: 2..<8, features: features)

        let plan = try QwenMropePositionPlan.make(
            tokenCount: 10,
            imageSpans: [span],
            startPosition: 7,
            previousRopeDelta: 3)

        #expect(plan.positions == [
            QwenMropePosition(temporal: 10, height: 10, width: 10),
            QwenMropePosition(temporal: 11, height: 11, width: 11),
            QwenMropePosition(temporal: 12, height: 12, width: 12),
            QwenMropePosition(temporal: 12, height: 12, width: 13),
            QwenMropePosition(temporal: 12, height: 12, width: 14),
            QwenMropePosition(temporal: 12, height: 13, width: 12),
            QwenMropePosition(temporal: 12, height: 13, width: 13),
            QwenMropePosition(temporal: 12, height: 13, width: 14),
            QwenMropePosition(temporal: 15, height: 15, width: 15),
            QwenMropePosition(temporal: 16, height: 16, width: 16),
        ])
        #expect(plan.ropeDelta == 0)
    }

    @Test func qwenFeatureCopyConvertsBFloat16RowsToFloat16() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let context = try MetalContext()
        let sourceValues: [UInt16] = [
            0,
            Quantization.bf16Bits(2),
            Quantization.bf16Bits(-1.5),
            0,
        ]
        let source = try #require(device.makeBuffer(
            length: sourceValues.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        sourceValues.withUnsafeBufferPointer { values in
            source.contents().copyMemory(
                from: values.baseAddress!,
                byteCount: sourceValues.count * MemoryLayout<UInt16>.stride)
        }
        let destination = try #require(device.makeBuffer(
            length: sourceValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        memset(destination.contents(), 0, destination.length)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try QwenBFloat16ToFloat16Copy(context: context).encode(
            commandBuffer: commandBuffer,
            source: source,
            sourceOffsetElements: 1,
            destination: destination,
            destinationOffsetElements: 1,
            elementCount: 2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let values = destination.contents().bindMemory(to: Float16.self, capacity: 4)
        #expect(values[0] == 0)
        #expect(values[1] == 2)
        #expect(values[2] == -1.5)
        #expect(values[3] == 0)
    }

    @Test func mropeKernelUsesPinnedAxisSections() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let context = try MetalContext()
        let inputValues = (0..<64).map { Float16($0 + 1) }
        let data = try #require(device.makeBuffer(
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        inputValues.withUnsafeBufferPointer { source in
            data.contents().copyMemory(
                from: source.baseAddress!,
                byteCount: inputValues.count * MemoryLayout<Float16>.stride)
        }
        let positionValues: [Int32] = [2, 3, 5]
        let positions = try #require(device.makeBuffer(
            length: positionValues.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared))
        positionValues.withUnsafeBufferPointer { source in
            positions.contents().copyMemory(
                from: source.baseAddress!,
                byteCount: positionValues.count * MemoryLayout<Int32>.stride)
        }
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try QwenMropeRotary(context: context).encode(
            commandBuffer: commandBuffer,
            data: data,
            positions: positions,
            positionsOffset: 0,
            tokenCount: 1,
            tokenStrideElements: 64,
            headDimension: 64,
            headCount: 1,
            rotaryPairs: 32,
            ropeTheta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let output = data.contents().bindMemory(to: Float16.self, capacity: 64)
        for pair in [0, 1, 2, 29, 30, 31] {
            let coordinate: Double
            if pair < 33 && pair % 3 == 1 {
                coordinate = 3
            } else if pair < 30 && pair % 3 == 2 {
                coordinate = 5
            } else {
                coordinate = 2
            }
            let inverseFrequency = Foundation.pow(
                10_000_000,
                -Double(pair) / 32)
            let angle = coordinate * inverseFrequency
            let cosine = Foundation.cos(angle)
            let sine = Foundation.sin(angle)
            let first = Double(inputValues[pair])
            let second = Double(inputValues[pair + 32])
            #expect(abs(Double(output[pair]) - (first * cosine - second * sine)) < 0.04)
            #expect(abs(Double(output[pair + 32]) - (second * cosine + first * sine)) < 0.04)
        }
    }

    @Test func biasGELUSaturatesLargePositiveArguments() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let context = try MetalContext()
        let inputBits: [UInt16] = [36.25, 42.25, 40.75, 37.25].map {
            Quantization.bf16Bits($0)
        }
        let biasBits: [UInt16] = [-14.875, -14.5, -14.6875, -14.875].map {
            Quantization.bf16Bits($0)
        }
        let byteCount = inputBits.count * MemoryLayout<UInt16>.stride

        func sharedBuffer(_ values: [UInt16]) throws -> MTLBuffer {
            let buffer = try #require(device.makeBuffer(
                length: byteCount, options: .storageModeShared))
            values.withUnsafeBufferPointer { source in
                buffer.contents().copyMemory(
                    from: source.baseAddress!, byteCount: byteCount)
            }
            return buffer
        }

        let input = try sharedBuffer(inputBits)
        let bias = try sharedBuffer(biasBits)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try QwenVisionPrimitives(context: context).encodeBiasGELU(
            commandBuffer: commandBuffer,
            values: input,
            bias: bias,
            biasOffset: 0,
            rows: 1,
            width: inputBits.count)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let output = input.contents().bindMemory(to: UInt16.self, capacity: inputBits.count)
        for index in inputBits.indices {
            let value = Quantization.bf16ToFloat(inputBits[index])
                + Quantization.bf16ToFloat(biasBits[index])
            let expected = Quantization.bf16Bits(value)
            #expect(Quantization.bf16ToFloat(output[index]).isFinite)
            #expect(output[index] == expected)
        }
    }
}
