import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct Qwen38QSATests {
    @Test func canonicalGeometryMatchesPinnedCheckpoint() {
        let geometry = Qwen38QSAGeometry.qwen
        #expect(geometry.queryHeads == 4)
        #expect(geometry.keyValueHeads == 1)
        #expect(geometry.headDimension == 128)
        #expect(geometry.compressRatio == 4)
        #expect(geometry.tokenBudget == 2_048)
        #expect(geometry.blockTopK == 512)
        #expect(geometry.rotaryDimension == 32)
        #expect(geometry.ropeTheta == 10_000_000)
    }

    @Test func rawKeyCacheOwnsKeysAndPositionsAcrossRollback() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let cache = try Qwen38QSARawKeyCache(
            device: context.device,
            capacity: 3,
            geometry: geometry)
        let width = Int(geometry.headDimension)
        let initialKeys = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: 1, count: width)
                + [Float](repeating: 2, count: width)))
        let initialPositions = try uint32Buffer(context.device, values: [10, 11])
        let appendBuffer = try #require(context.queue.makeCommandBuffer())
        cache.appendBatch(
            commandBuffer: appendBuffer,
            rawKeys: initialKeys,
            positions: initialPositions,
            tokenCount: 2)
        appendBuffer.commit()
        appendBuffer.waitUntilCompleted()
        try checkCommandBufferError(appendBuffer.error)
        let snapshot = cache.snapshot()

        cache.rewind(to: 1)
        let replacementKey = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: 3, count: width)))
        let replacementPosition = try uint32Buffer(context.device, values: [12])
        let replaceBuffer = try #require(context.queue.makeCommandBuffer())
        cache.append(
            commandBuffer: replaceBuffer,
            rawKey: replacementKey,
            position: replacementPosition)
        replaceBuffer.commit()
        replaceBuffer.waitUntilCompleted()
        try checkCommandBufferError(replaceBuffer.error)

        #expect(cache.count == 2)
        #expect(Fp16Buffer.read(cache.rawKeys, count: width * 2)[width] == 3)
        #expect(readUInt32(cache.positions, count: 2) == [10, 12])

        cache.restore(snapshot)
        #expect(cache.count == 2)
        #expect(Fp16Buffer.read(cache.rawKeys, count: width * 2)[width] == 2)
        #expect(readUInt32(cache.positions, count: 2) == [10, 11])

        cache.reset()
        #expect(cache.count == 0)
        #expect(Fp16Buffer.read(cache.rawKeys, count: width * 3).allSatisfy { $0 == 0 })
        #expect(readUInt32(cache.positions, count: 3).allSatisfy { $0 == 0 })
    }

    @Test func projectionPopulatesRawKeyTailAtCacheOffset() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let projection = try Qwen38QSAProjection(context: context, geometry: geometry)
        let cache = try Qwen38QSARawKeyCache(
            device: context.device,
            capacity: 3,
            geometry: geometry)
        let inputWidth = 32
        let tokenCount = 2
        let outputWidth = Int(geometry.projectionWidth)
        let packedWeights = [UInt8](
            repeating: 0, count: outputWidth * inputWidth / 2)
        let scales = [UInt16](repeating: Quantization.bf16Bits(0), count: outputWidth)
        let biases = (1...outputWidth).map {
            Quantization.bf16Bits(Float($0) / Float(inputWidth))
        }
        let input = [Float](repeating: 1, count: inputWidth)
            + [Float](repeating: 2, count: inputWidth)
        let weightsBuffer = try #require(context.device.makeBuffer(
            bytes: packedWeights,
            length: packedWeights.count,
            options: .storageModeShared))
        let scalesBuffer = try #require(context.device.makeBuffer(
            bytes: scales,
            length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let biasesBuffer = try #require(context.device.makeBuffer(
            bytes: biases,
            length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let inputBuffer = try #require(Fp16Buffer.make(context.device, values: input))
        let projectedRows = try #require(Fp16Buffer.make(
            context.device, count: tokenCount * outputWidth))
        let positionBuffer = try uint32Buffer(context.device, values: [999, 7, 8])
        let existingKey = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: -1, count: Int(geometry.rawKeyWidth))))
        let existingPosition = try uint32Buffer(context.device, values: [6])
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        cache.append(
            commandBuffer: commandBuffer,
            rawKey: existingKey,
            position: existingPosition)

        projection.encode(
            commandBuffer: commandBuffer,
            weights: weightsBuffer,
            scales: scalesBuffer,
            biases: biasesBuffer,
            hiddenStates: inputBuffer,
            projectedRows: projectedRows,
            positions: positionBuffer,
            positionsOffset: MemoryLayout<UInt32>.stride,
            rawKeyCache: cache,
            tokenCount: UInt32(tokenCount),
            inputWidth: UInt32(inputWidth))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let expectedProjection = (1...outputWidth).map(Float.init)
            + (1...outputWidth).map { Float($0 * 2) }
        expectClose(
            Fp16Buffer.read(projectedRows, count: tokenCount * outputWidth),
            expectedProjection,
            tolerance: 0.001)
        let expectedCachedKeys = [Float](
            repeating: -1, count: Int(geometry.rawKeyWidth))
            + Array(expectedProjection[Int(geometry.queryWidth)..<outputWidth])
            + Array(expectedProjection[
                (outputWidth + Int(geometry.queryWidth))..<(outputWidth * 2)])
        expectClose(
            Fp16Buffer.read(
                cache.rawKeys,
                count: cache.count * Int(geometry.rawKeyWidth)),
            expectedCachedKeys,
            tolerance: 0.001)
        #expect(cache.count == 3)
        #expect(readUInt32(cache.positions, count: 3) == [6, 7, 8])
    }

    @Test func layerExecutorTransitionsFromShortHistoryToFirstBlock() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let inputWidth = 32
        let outputWidth = Int(geometry.projectionWidth)
        let packedWeightBytes = outputWidth * inputWidth / 2
        let scaleBytes = outputWidth * MemoryLayout<UInt16>.stride
        let biasBytes = outputWidth * MemoryLayout<UInt16>.stride
        var projectionStorage = [UInt8](repeating: 0, count: packedWeightBytes)
        let zeroScale = Quantization.bf16Bits(0)
        let rowBias = Quantization.bf16Bits(1 / Float(inputWidth))
        for value in [UInt16](repeating: zeroScale, count: outputWidth) {
            projectionStorage.append(UInt8(value & 0xff))
            projectionStorage.append(UInt8(value >> 8))
        }
        for value in [UInt16](repeating: rowBias, count: outputWidth) {
            projectionStorage.append(UInt8(value & 0xff))
            projectionStorage.append(UInt8(value >> 8))
        }
        let projectionBuffer = try #require(context.device.makeBuffer(
            bytes: projectionStorage,
            length: projectionStorage.count,
            options: .storageModeShared))
        let projectionView = TensorView(
            buffer: projectionBuffer,
            offset: 0,
            length: UInt64(packedWeightBytes),
            scaleOffset: UInt64(packedWeightBytes),
            scaleLength: UInt64(scaleBytes),
            biasOffset: UInt64(packedWeightBytes + scaleBytes),
            biasLength: UInt64(biasBytes),
            shape: (UInt32(outputWidth), UInt32(inputWidth), 1, 1),
            dtype: 1)
        let normBuffer = try bf16Buffer(
            context.device,
            values: [Float](repeating: 0, count: Int(geometry.headDimension)))
        let normView = TensorView(
            buffer: normBuffer,
            offset: 0,
            length: UInt64(normBuffer.length),
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (geometry.headDimension, 1, 1, 1),
            dtype: 1)
        let cache = try Qwen38QSARawKeyCache(
            device: context.device,
            capacity: 3,
            geometry: geometry)
        let layerState = Qwen38QSALayerState(
            layer: 3,
            weights: Qwen38QSALayerWeights(
                projection: projectionView,
                queryNorm: normView,
                keyNorm: normView),
            rawKeyCache: cache)
        let executor = try Qwen38QSALayerExecutor(
            context: context,
            geometry: geometry)
        let hiddenStates = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: 1, count: inputWidth)))

        func execute(position: UInt32, expectedMask: [UInt8]) throws {
            let keyCount = cache.count + 1
            let blockCount = keyCount / Int(geometry.compressRatio)
            let positions = try uint32Buffer(context.device, values: [position])
            let visibleCounts = try uint32Buffer(
                context.device, values: [UInt32(keyCount)])
            let projectedRows = try #require(Fp16Buffer.make(
                context.device, count: outputWidth))
            let blockScores = try #require(context.device.makeBuffer(
                length: max(blockCount * MemoryLayout<Float>.stride, 1),
                options: .storageModeShared))
            let selectionState = try #require(context.device.makeBuffer(
                length: Qwen38QSASelector.stateBytesPerQuery,
                options: .storageModeShared))
            let tokenMask = try #require(context.device.makeBuffer(
                length: keyCount,
                options: .storageModeShared))
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            executor.encode(
                commandBuffer: commandBuffer,
                state: layerState,
                hiddenStates: hiddenStates,
                queryPositions: positions,
                visibleTokenCounts: visibleCounts,
                scratch: Qwen38QSALayerExecutionScratch(
                    projectedRows: projectedRows,
                    blockScores: blockScores,
                    selection: Qwen38QSASelectionScratch(
                        state: selectionState,
                        tokenMask: tokenMask)),
                tokenCount: 1,
                inputWidth: UInt32(inputWidth),
                epsilon: 1e-6)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            try checkCommandBufferError(commandBuffer.error)
            let actualMask = Array(UnsafeBufferPointer(
                start: tokenMask.contents().bindMemory(
                    to: UInt8.self, capacity: keyCount),
                count: keyCount))
            #expect(actualMask == expectedMask)
        }

        try execute(position: 0, expectedMask: [1])
        try execute(position: 1, expectedMask: [1, 1])
        #expect(cache.count == 2)
        #expect(readUInt32(cache.positions, count: 2) == [0, 1])

        let paddedHiddenStates = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: 1, count: inputWidth * 2)))
        let paddedPositions = try uint32Buffer(context.device, values: [2, 999])
        let paddedVisibleCounts = try uint32Buffer(context.device, values: [3, 0])
        let paddedProjectedRows = try #require(Fp16Buffer.make(
            context.device,
            count: outputWidth * 2))
        let paddedBlockScores = try #require(context.device.makeBuffer(
            length: MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let paddedSelectionState = try #require(context.device.makeBuffer(
            length: Qwen38QSASelector.stateBytesPerQuery,
            options: .storageModeShared))
        let paddedTokenMask = try #require(context.device.makeBuffer(
            bytes: [UInt8](repeating: 255, count: 6),
            length: 6,
            options: .storageModeShared))
        let paddedCommandBuffer = try #require(context.queue.makeCommandBuffer())
        executor.encode(
            commandBuffer: paddedCommandBuffer,
            state: layerState,
            hiddenStates: paddedHiddenStates,
            queryPositions: paddedPositions,
            visibleTokenCounts: paddedVisibleCounts,
            scratch: Qwen38QSALayerExecutionScratch(
                projectedRows: paddedProjectedRows,
                blockScores: paddedBlockScores,
                selection: Qwen38QSASelectionScratch(
                    state: paddedSelectionState,
                    tokenMask: paddedTokenMask)),
            tokenCount: 2,
            inputWidth: UInt32(inputWidth),
            epsilon: 1e-6,
            validTokenCount: 1)
        paddedCommandBuffer.commit()
        paddedCommandBuffer.waitUntilCompleted()
        try checkCommandBufferError(paddedCommandBuffer.error)
        #expect(cache.count == 3)
        #expect(readUInt32(cache.positions, count: 3) == [0, 1, 2])
        #expect(readBytes(paddedTokenMask, count: 6) == [1, 1, 1, 0, 0, 0])

        memset(paddedTokenMask.contents(), 255, paddedTokenMask.length)
        let emptyCommandBuffer = try #require(context.queue.makeCommandBuffer())
        executor.encode(
            commandBuffer: emptyCommandBuffer,
            state: layerState,
            hiddenStates: paddedHiddenStates,
            queryPositions: paddedPositions,
            visibleTokenCounts: paddedVisibleCounts,
            scratch: Qwen38QSALayerExecutionScratch(
                projectedRows: paddedProjectedRows,
                blockScores: paddedBlockScores,
                selection: Qwen38QSASelectionScratch(
                    state: paddedSelectionState,
                    tokenMask: paddedTokenMask)),
            tokenCount: 2,
            inputWidth: UInt32(inputWidth),
            epsilon: 1e-6,
            validTokenCount: 0)
        emptyCommandBuffer.commit()
        emptyCommandBuffer.waitUntilCompleted()
        try checkCommandBufferError(emptyCommandBuffer.error)
        #expect(cache.count == 3)
        #expect(readUInt32(cache.positions, count: 3) == [0, 1, 2])
        #expect(readBytes(paddedTokenMask, count: 6) == [0, 0, 0, 0, 0, 0])
    }

    @Test func fusedBlockScoresMatchCPUReference() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let scorer = try Qwen38QSABlockScorer(context: context, geometry: geometry)
        let queryCount = 2
        let keyCount = 6
        let queryStride = Int(
            (geometry.queryHeads + geometry.keyValueHeads) * geometry.headDimension)
        let projected = (0..<(queryCount * queryStride)).map { index in
            quantizedHalf(Float((index * 7) % 19 - 9) / 7)
        }
        let rawKeys = (0..<(keyCount * Int(geometry.headDimension))).map { index in
            quantizedHalf(Float((index * 5 + 3) % 17 - 8) / 6)
        }
        let queryNorm = (0..<Int(geometry.headDimension)).map { index in
            quantizedBF16(Float(index - 3) / 32)
        }
        let keyNorm = (0..<Int(geometry.headDimension)).map { index in
            quantizedBF16(Float(4 - index) / 40)
        }
        let queryPositions: [UInt32] = [2, 5]
        let keyPositions: [UInt32] = [0, 1, 2, 3, 4, 5]
        let projectedBuffer = try #require(Fp16Buffer.make(
            context.device, values: projected))
        let rawKeyBuffer = try #require(Fp16Buffer.make(
            context.device, values: rawKeys))
        let queryNormBuffer = try bf16Buffer(context.device, values: queryNorm)
        let keyNormBuffer = try bf16Buffer(context.device, values: keyNorm)
        let queryPositionBuffer = try uint32Buffer(
            context.device, values: queryPositions)
        let keyPositionBuffer = try uint32Buffer(context.device, values: keyPositions)
        let blockCount = keyCount / Int(geometry.compressRatio)
        let output = try #require(context.device.makeBuffer(
            length: queryCount * blockCount * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let cache = try Qwen38QSARawKeyCache(
            device: context.device,
            capacity: keyCount,
            geometry: geometry)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        cache.appendBatch(
            commandBuffer: commandBuffer,
            rawKeys: rawKeyBuffer,
            positions: keyPositionBuffer,
            tokenCount: keyCount)

        scorer.encode(
            commandBuffer: commandBuffer,
            projectedQueries: projectedBuffer,
            rawKeyCache: cache,
            queryNorm: queryNormBuffer,
            keyNorm: keyNormBuffer,
            queryPositions: queryPositionBuffer,
            outputScores: output,
            queryCount: UInt32(queryCount),
            epsilon: 1e-6)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let actual = Array(UnsafeBufferPointer(
            start: output.contents().bindMemory(
                to: Float.self, capacity: queryCount * blockCount),
            count: queryCount * blockCount))
        let expected = referenceScores(
            projected: projected,
            rawKeys: rawKeys,
            queryNorm: queryNorm,
            keyNorm: keyNorm,
            queryPositions: queryPositions,
            keyPositions: keyPositions,
            queryCount: queryCount,
            keyCount: keyCount,
            geometry: geometry,
            epsilon: 1e-6)
        expectClose(actual, expected, tolerance: 0.003)
    }

    @Test func selectionMatchesDenseSparseTailAndTieSemantics() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let selector = try Qwen38QSASelector(context: context, geometry: geometry)
        let queryCount = 3
        let keyCount = 12
        let blockCount = keyCount / Int(geometry.compressRatio)
        let scores: [Float] = [
            90, 80, 70, 60, 50, 40,
            1, 9, 5, 8, 7, 100,
            5, 5, 5, 1, 100, 100,
        ]
        let visibleTokenCounts: [UInt32] = [4, 11, 8]
        let scoreBuffer = try #require(context.device.makeBuffer(
            bytes: scores,
            length: scores.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let visibleTokenBuffer = try uint32Buffer(
            context.device, values: visibleTokenCounts)
        let state = try #require(context.device.makeBuffer(
            length: queryCount * Qwen38QSASelector.stateBytesPerQuery,
            options: .storageModeShared))
        let tokenMask = try #require(context.device.makeBuffer(
            length: queryCount * keyCount,
            options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        selector.encode(
            commandBuffer: commandBuffer,
            scores: scoreBuffer,
            visibleTokenCounts: visibleTokenBuffer,
            scratch: Qwen38QSASelectionScratch(state: state, tokenMask: tokenMask),
            queryCount: UInt32(queryCount),
            keyCount: UInt32(keyCount))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let actual = Array(UnsafeBufferPointer(
            start: tokenMask.contents().bindMemory(
                to: UInt8.self, capacity: queryCount * keyCount),
            count: queryCount * keyCount))
        let expected: [UInt8] = [
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0,
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        #expect(actual == expected)
        #expect(scores.count == queryCount * blockCount)
    }

    @Test func selectionBindsBlockTopKWhenVisibleBlocksExceedBudget() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let selector = try Qwen38QSASelector(context: context, geometry: geometry)
        let scores: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let scoreBuffer = try #require(context.device.makeBuffer(
            bytes: scores,
            length: scores.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let visibleTokenBuffer = try uint32Buffer(context.device, values: [16])
        let state = try #require(context.device.makeBuffer(
            length: Qwen38QSASelector.stateBytesPerQuery,
            options: .storageModeShared))
        let tokenMask = try #require(context.device.makeBuffer(
            length: 16,
            options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        selector.encode(
            commandBuffer: commandBuffer,
            scores: scoreBuffer,
            visibleTokenCounts: visibleTokenBuffer,
            scratch: Qwen38QSASelectionScratch(state: state, tokenMask: tokenMask),
            queryCount: 1,
            keyCount: 16)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let stateValues = readUInt32(state, count: 5)
        #expect(stateValues[1] == 1)
        #expect(stateValues[2] == 8)
        #expect(stateValues[3] == 2)
        #expect(readBytes(tokenMask, count: 16) == [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1,
        ])
    }

    @Test func selectionReusePreservesRowAndAddsVisibleTail() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry(
            queryHeads: 2,
            keyValueHeads: 1,
            headDimension: 8,
            compressRatio: 2,
            tokenBudget: 4,
            rotaryDimension: 4,
            ropeTheta: 100)
        let selector = try Qwen38QSASelector(context: context, geometry: geometry)
        let initialScores: [Float] = [1, 9, 8, 2]
        let initialScoreBuffer = try #require(context.device.makeBuffer(
            bytes: initialScores,
            length: initialScores.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let initialVisibleTokens = try uint32Buffer(context.device, values: [8])
        let state = try #require(context.device.makeBuffer(
            length: Qwen38QSASelector.stateBytesPerQuery,
            options: .storageModeShared))
        let initialMask = try #require(context.device.makeBuffer(
            length: 8,
            options: .storageModeShared))
        let retainedMask = try #require(context.device.makeBuffer(
            length: 8,
            options: .storageModeShared))
        let initialCommandBuffer = try #require(context.queue.makeCommandBuffer())
        selector.encode(
            commandBuffer: initialCommandBuffer,
            scores: initialScoreBuffer,
            visibleTokenCounts: initialVisibleTokens,
            scratch: Qwen38QSASelectionScratch(
                state: state,
                tokenMask: initialMask,
                captureTokenMask: retainedMask),
            queryCount: 1,
            keyCount: 8)
        initialCommandBuffer.commit()
        initialCommandBuffer.waitUntilCompleted()
        try checkCommandBufferError(initialCommandBuffer.error)
        #expect(readBytes(retainedMask, count: 8) == [
            0, 0, 1, 1, 1, 1, 0, 0,
        ])

        let laterScores = [Float](repeating: 0, count: 5)
        let laterScoreBuffer = try #require(context.device.makeBuffer(
            bytes: laterScores,
            length: laterScores.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let laterVisibleTokens = try uint32Buffer(context.device, values: [10])
        let laterMask = try #require(context.device.makeBuffer(
            length: 10,
            options: .storageModeShared))
        let laterCommandBuffer = try #require(context.queue.makeCommandBuffer())
        selector.encode(
            commandBuffer: laterCommandBuffer,
            scores: laterScoreBuffer,
            visibleTokenCounts: laterVisibleTokens,
            scratch: Qwen38QSASelectionScratch(
                state: state,
                tokenMask: laterMask,
                reusableTokenMask: retainedMask,
                reusableKeyCount: 8),
            queryCount: 1,
            keyCount: 10)
        laterCommandBuffer.commit()
        laterCommandBuffer.waitUntilCompleted()
        try checkCommandBufferError(laterCommandBuffer.error)
        #expect(readBytes(laterMask, count: 10) == [
            0, 0, 1, 1, 1, 1, 0, 0, 1, 1,
        ])
    }

    @Test func selectionReuseSupportsCanonicalBudgetTail() throws {
        let context = try MetalContext()
        let geometry = Qwen38QSAGeometry.qwen
        let selector = try Qwen38QSASelector(context: context, geometry: geometry)
        let sourceKeyCount = Int(geometry.tokenBudget)
        let destinationKeyCount = sourceKeyCount + 4
        let retainedBytes = (0..<sourceKeyCount).map { index in
            UInt8(index % 3 == 0 ? 1 : 0)
        }
        let retainedMask = try #require(context.device.makeBuffer(
            length: sourceKeyCount,
            options: .storageModeShared))
        retainedBytes.withUnsafeBytes { source in
            retainedMask.contents().copyMemory(
                from: source.baseAddress!,
                byteCount: retainedBytes.count)
        }
        let scores = [Float](
            repeating: 0,
            count: destinationKeyCount / Int(geometry.compressRatio))
        let scoreBuffer = try #require(context.device.makeBuffer(
            bytes: scores,
            length: scores.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let visibleTokenBuffer = try uint32Buffer(
            context.device, values: [UInt32(destinationKeyCount)])
        let state = try #require(context.device.makeBuffer(
            length: Qwen38QSASelector.stateBytesPerQuery,
            options: .storageModeShared))
        let tokenMask = try #require(context.device.makeBuffer(
            length: destinationKeyCount,
            options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        selector.encode(
            commandBuffer: commandBuffer,
            scores: scoreBuffer,
            visibleTokenCounts: visibleTokenBuffer,
            scratch: Qwen38QSASelectionScratch(
                state: state,
                tokenMask: tokenMask,
                reusableTokenMask: retainedMask,
                reusableKeyCount: UInt32(sourceKeyCount)),
            queryCount: 1,
            keyCount: UInt32(destinationKeyCount))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)
        #expect(readBytes(tokenMask, count: destinationKeyCount) ==
                retainedBytes + Array(repeating: UInt8(1), count: 4))
    }

    private func referenceScores(projected: [Float],
                                 rawKeys: [Float],
                                 queryNorm: [Float],
                                 keyNorm: [Float],
                                 queryPositions: [UInt32],
                                 keyPositions: [UInt32],
                                 queryCount: Int,
                                 keyCount: Int,
                                 geometry: Qwen38QSAGeometry,
                                 epsilon: Float) -> [Float] {
        let heads = Int(geometry.queryHeads)
        let headDimension = Int(geometry.headDimension)
        let compressRatio = Int(geometry.compressRatio)
        let queryStride = Int(
            (geometry.queryHeads + geometry.keyValueHeads) * geometry.headDimension)
        let blockCount = keyCount / compressRatio
        var scores: [Float] = []
        for query in 0..<queryCount {
            for block in 0..<blockCount {
                var pooled = [Float](repeating: 0, count: headDimension)
                for token in 0..<compressRatio {
                    let keyBase = (block * compressRatio + token) * headDimension
                    for feature in 0..<headDimension {
                        pooled[feature] += rawKeys[keyBase + feature] / Float(compressRatio)
                    }
                }
                let normalizedKey = rotate(
                    centeredRMSNorm(pooled, weight: keyNorm, epsilon: epsilon),
                    position: keyPositions[block * compressRatio],
                    rotaryDimension: Int(geometry.rotaryDimension),
                    theta: geometry.ropeTheta)
                var score: Float = 0
                for head in 0..<heads {
                    let base = query * queryStride + head * headDimension
                    let values = Array(projected[base..<(base + headDimension)])
                    let normalizedQuery = rotate(
                        centeredRMSNorm(values, weight: queryNorm, epsilon: epsilon),
                        position: queryPositions[query],
                        rotaryDimension: Int(geometry.rotaryDimension),
                        theta: geometry.ropeTheta)
                    let dot = zip(normalizedQuery, normalizedKey).reduce(Float(0)) {
                        $0 + $1.0 * $1.1
                    }
                    score += max(dot, 0)
                }
                scores.append(score / sqrt(Float(headDimension)))
            }
        }
        return scores
    }

    private func centeredRMSNorm(_ values: [Float],
                                 weight: [Float],
                                 epsilon: Float) -> [Float] {
        let meanSquare = values.reduce(Float(0)) { $0 + $1 * $1 } / Float(values.count)
        let inverseRMS = 1 / sqrt(meanSquare + epsilon)
        return zip(values, weight).map { $0 * inverseRMS * (1 + $1) }
    }

    private func rotate(_ values: [Float],
                        position: UInt32,
                        rotaryDimension: Int,
                        theta: Float) -> [Float] {
        var output = values
        let half = rotaryDimension / 2
        for feature in 0..<rotaryDimension {
            let pair = feature < half ? feature + half : feature - half
            let frequency = feature % half
            let inverseFrequency = pow(theta, -2 * Float(frequency) / Float(rotaryDimension))
            let angle = Float(position) * inverseFrequency
            output[feature] = values[feature] * cos(angle)
                + (feature < half ? -values[pair] : values[pair]) * sin(angle)
        }
        return output
    }

    private func expectClose(_ actual: [Float],
                             _ expected: [Float],
                             tolerance: Float) {
        #expect(actual.count == expected.count)
        for (index, pair) in zip(actual, expected).enumerated() {
            #expect(abs(pair.0 - pair.1) <= tolerance,
                    "index \(index): \(pair.0) != \(pair.1)")
        }
    }

    private func quantizedHalf(_ value: Float) -> Float {
        Float(Float16(value))
    }

    private func quantizedBF16(_ value: Float) -> Float {
        Quantization.bf16ToFloat(Quantization.bf16Bits(value))
    }

    private func bf16Buffer(_ device: MTLDevice, values: [Float]) throws -> MTLBuffer {
        let bits = values.map(Quantization.bf16Bits)
        return try #require(device.makeBuffer(
            bytes: bits,
            length: bits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
    }

    private func uint32Buffer(_ device: MTLDevice, values: [UInt32]) throws -> MTLBuffer {
        try #require(device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
    }

    private func readUInt32(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt32.self, capacity: count),
            count: count))
    }

    private func readBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt8.self, capacity: count),
            count: count))
    }
}
