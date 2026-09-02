import Foundation
import Metal

struct Qwen38QSAGeometry: Sendable, Equatable {
    let queryHeads: UInt32
    let keyValueHeads: UInt32
    let headDimension: UInt32
    let compressRatio: UInt32
    let tokenBudget: UInt32
    let rotaryDimension: UInt32
    let ropeTheta: Float

    var queryWidth: UInt32 { queryHeads * headDimension }
    var rawKeyWidth: UInt32 { keyValueHeads * headDimension }
    var projectionWidth: UInt32 { queryWidth + rawKeyWidth }
    var blockTopK: UInt32 { tokenBudget / compressRatio }

    static let qwen = Qwen38QSAGeometry(
        queryHeads: 4,
        keyValueHeads: 1,
        headDimension: 128,
        compressRatio: 4,
        tokenBudget: 2_048,
        rotaryDimension: 32,
        ropeTheta: 10_000_000)
}

struct Qwen38QSARawKeySnapshot {
    let rawKeys: [UInt8]
    let positions: [UInt8]
    let count: Int
}

final class Qwen38QSARawKeyCache {
    let geometry: Qwen38QSAGeometry
    let capacity: Int
    let rawKeys: MTLBuffer
    let positions: MTLBuffer
    private(set) var count = 0

    init(device: MTLDevice,
         capacity: Int,
         geometry: Qwen38QSAGeometry = .qwen) throws {
        precondition(capacity > 0, "QSA raw-key capacity must be positive")
        precondition(geometry.keyValueHeads == 1,
                     "QSA raw-key cache requires one key-value head")
        self.geometry = geometry
        self.capacity = capacity
        guard let rawKeys = device.makeBuffer(
            length: capacity * Self.tokenBytes(for: geometry),
            options: .storageModeShared),
              let positions = device.makeBuffer(
                length: capacity * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.rawKeys = rawKeys
        self.positions = positions
        reset()
    }

    var tokenBytes: Int {
        Self.tokenBytes(for: geometry)
    }

    func snapshot() -> Qwen38QSARawKeySnapshot {
        Qwen38QSARawKeySnapshot(
            rawKeys: bytes(from: rawKeys, count: count * tokenBytes),
            positions: bytes(
                from: positions,
                count: count * MemoryLayout<UInt32>.stride),
            count: count)
    }

    func restore(_ snapshot: Qwen38QSARawKeySnapshot) {
        precondition(snapshot.count >= 0 && snapshot.count <= capacity,
                     "QSA raw-key snapshot exceeds cache capacity")
        precondition(snapshot.rawKeys.count == snapshot.count * tokenBytes,
                     "QSA raw-key snapshot bytes do not match its count")
        precondition(snapshot.positions.count ==
            snapshot.count * MemoryLayout<UInt32>.stride,
            "QSA position snapshot bytes do not match its count")
        copy(snapshot.rawKeys, to: rawKeys)
        copy(snapshot.positions, to: positions)
        count = snapshot.count
    }

    func rewind(to retainedCount: Int) {
        precondition(retainedCount >= 0 && retainedCount <= count,
                     "QSA raw-key rewind must retain a populated prefix")
        count = retainedCount
    }

    func append(commandBuffer: MTLCommandBuffer,
                rawKey sourceRawKey: MTLBuffer,
                rawKeyOffset: Int = 0,
                position sourcePosition: MTLBuffer,
                positionOffset: Int = 0) {
        precondition(count < capacity, "QSA raw-key cache capacity exceeded")
        appendBatch(
            commandBuffer: commandBuffer,
            rawKeys: sourceRawKey,
            rawKeyOffset: rawKeyOffset,
            positions: sourcePosition,
            positionOffset: positionOffset,
            tokenCount: 1)
    }

    func appendBatch(commandBuffer: MTLCommandBuffer,
                     rawKeys sourceRawKeys: MTLBuffer,
                     rawKeyOffset: Int = 0,
                     positions sourcePositions: MTLBuffer,
                     positionOffset: Int = 0,
                     tokenCount: Int) {
        precondition(tokenCount > 0 && count + tokenCount <= capacity,
                     "QSA raw-key batch exceeds cache capacity")
        precondition(rawKeyOffset >= 0 && rawKeyOffset % 2 == 0,
                     "FP16 QSA raw-key offset must be two-byte aligned")
        precondition(positionOffset >= 0 && positionOffset % 4 == 0,
                     "QSA position offset must be four-byte aligned")
        let rawKeyBytes = tokenCount * tokenBytes
        let positionBytes = tokenCount * MemoryLayout<UInt32>.stride
        precondition(rawKeyOffset + rawKeyBytes <= sourceRawKeys.length,
                     "QSA raw-key source is too small")
        precondition(positionOffset + positionBytes <= sourcePositions.length,
                     "QSA position source is too small")
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(
            from: sourceRawKeys,
            sourceOffset: rawKeyOffset,
            to: rawKeys,
            destinationOffset: count * tokenBytes,
            size: rawKeyBytes)
        blit.copy(
            from: sourcePositions,
            sourceOffset: positionOffset,
            to: positions,
            destinationOffset: count * MemoryLayout<UInt32>.stride,
            size: positionBytes)
        blit.endEncoding()
        count += tokenCount
    }

    func reset() {
        memset(rawKeys.contents(), 0, rawKeys.length)
        memset(positions.contents(), 0, positions.length)
        count = 0
    }

    fileprivate func reserveAppend(tokenCount: Int) -> Int {
        precondition(tokenCount > 0 && count + tokenCount <= capacity,
                     "QSA projected raw-key batch exceeds cache capacity")
        let destinationTokenOffset = count
        count += tokenCount
        return destinationTokenOffset
    }

    private static func tokenBytes(for geometry: Qwen38QSAGeometry) -> Int {
        Int(geometry.keyValueHeads * geometry.headDimension)
            * MemoryLayout<Float16>.stride
    }

    private func bytes(from buffer: MTLBuffer, count: Int) -> [UInt8] {
        let pointer = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func copy(_ bytes: [UInt8], to buffer: MTLBuffer) {
        guard !bytes.isEmpty else { return }
        _ = bytes.withUnsafeBytes { source in
            memcpy(buffer.contents(), source.baseAddress!, bytes.count)
        }
    }
}

final class Qwen38QSAProjection {
    private let projection: Qwen38PLEProjection
    private let cacheAppendPipeline: MTLComputePipelineState
    let geometry: Qwen38QSAGeometry

    init(context: MetalContext, geometry: Qwen38QSAGeometry = .qwen) throws {
        precondition(geometry.queryHeads > 0 && geometry.keyValueHeads == 1)
        precondition(geometry.headDimension > 0)
        self.geometry = geometry
        self.projection = try Qwen38PLEProjection(context: context)
        self.cacheAppendPipeline = try context.pipeline("qwen38_qsa_cache_append")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                hiddenStates: MTLBuffer,
                projectedRows: MTLBuffer,
                positions: MTLBuffer,
                positionsOffset: Int = 0,
                rawKeyCache: Qwen38QSARawKeyCache,
                tokenCount: UInt32,
                inputWidth: UInt32) {
        precondition(rawKeyCache.geometry == geometry,
                     "QSA projection and raw-key cache geometry must match")
        precondition(tokenCount > 0 && inputWidth > 0)
        precondition(positionsOffset >= 0 && positionsOffset % 4 == 0)
        let projectedBytes = Int(tokenCount * geometry.projectionWidth)
            * MemoryLayout<Float16>.stride
        let positionBytes = Int(tokenCount) * MemoryLayout<UInt32>.stride
        precondition(projectedRows.length >= projectedBytes,
                     "QSA projected-row output is too small")
        precondition(positionsOffset + positionBytes <= positions.length,
                     "QSA source positions are too small")
        precondition(rawKeyCache.count + Int(tokenCount) <= rawKeyCache.capacity,
                     "QSA projected raw-key batch exceeds cache capacity")

        projection.encode(
            commandBuffer: commandBuffer,
            weights: weights,
            weightsOffset: weightsOffset,
            scales: scales,
            scalesOffset: scalesOffset,
            biases: biases,
            biasesOffset: biasesOffset,
            input: hiddenStates,
            output: projectedRows,
            tokenCount: tokenCount,
            outputWidth: geometry.projectionWidth,
            inputWidth: inputWidth)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let destinationTokenOffset = rawKeyCache.reserveAppend(
            tokenCount: Int(tokenCount))
        encoder.setComputePipelineState(cacheAppendPipeline)
        encoder.setBuffer(projectedRows, offset: 0, index: 0)
        encoder.setBuffer(positions, offset: positionsOffset, index: 1)
        encoder.setBuffer(rawKeyCache.rawKeys, offset: 0, index: 2)
        encoder.setBuffer(rawKeyCache.positions, offset: 0, index: 3)
        var tokens = tokenCount
        var projectionWidth = geometry.projectionWidth
        var queryWidth = geometry.queryWidth
        var rawKeyWidth = geometry.rawKeyWidth
        var destination = UInt32(destinationTokenOffset)
        encoder.setBytes(&tokens, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(
            &projectionWidth, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&queryWidth, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&rawKeyWidth, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&destination, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreads(
            MTLSize(
                width: Int(geometry.rawKeyWidth),
                height: Int(tokenCount),
                depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(
                    Int(geometry.rawKeyWidth),
                    cacheAppendPipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }
}

final class Qwen38QSABlockScorer {
    private let pipeline: MTLComputePipelineState
    let geometry: Qwen38QSAGeometry

    init(context: MetalContext, geometry: Qwen38QSAGeometry = .qwen) throws {
        precondition(geometry.queryHeads > 0 && geometry.keyValueHeads == 1)
        precondition(geometry.headDimension > 0 && geometry.compressRatio > 0)
        precondition(geometry.tokenBudget.isMultiple(of: geometry.compressRatio))
        precondition(geometry.rotaryDimension.isMultiple(of: 2))
        precondition(geometry.rotaryDimension <= geometry.headDimension)
        self.geometry = geometry
        self.pipeline = try context.pipeline("qwen38_qsa_block_scores")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                projectedQueries: MTLBuffer,
                rawKeyCache: Qwen38QSARawKeyCache,
                queryNorm: MTLBuffer,
                queryNormOffset: Int = 0,
                keyNorm: MTLBuffer,
                keyNormOffset: Int = 0,
                queryPositions: MTLBuffer,
                outputScores: MTLBuffer,
                queryCount: UInt32,
                epsilon: Float) {
        precondition(rawKeyCache.geometry == geometry,
                     "QSA scorer and raw-key cache geometry must match")
        encode(
            commandBuffer: commandBuffer,
            projectedQueries: projectedQueries,
            rawKeys: rawKeyCache.rawKeys,
            queryNorm: queryNorm,
            queryNormOffset: queryNormOffset,
            keyNorm: keyNorm,
            keyNormOffset: keyNormOffset,
            queryPositions: queryPositions,
            keyPositions: rawKeyCache.positions,
            outputScores: outputScores,
            queryCount: queryCount,
            keyCount: UInt32(rawKeyCache.count),
            epsilon: epsilon)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                projectedQueries: MTLBuffer,
                rawKeys: MTLBuffer,
                queryNorm: MTLBuffer,
                queryNormOffset: Int = 0,
                keyNorm: MTLBuffer,
                keyNormOffset: Int = 0,
                queryPositions: MTLBuffer,
                keyPositions: MTLBuffer,
                outputScores: MTLBuffer,
                queryCount: UInt32,
                keyCount: UInt32,
                epsilon: Float) {
        precondition(queryCount > 0 && keyCount >= geometry.compressRatio)
        precondition(queryNormOffset >= 0 && keyNormOffset >= 0)
        let blockCount = keyCount / geometry.compressRatio
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(projectedQueries, offset: 0, index: 0)
        encoder.setBuffer(rawKeys, offset: 0, index: 1)
        encoder.setBuffer(queryNorm, offset: queryNormOffset, index: 2)
        encoder.setBuffer(keyNorm, offset: keyNormOffset, index: 3)
        encoder.setBuffer(queryPositions, offset: 0, index: 4)
        encoder.setBuffer(keyPositions, offset: 0, index: 5)
        encoder.setBuffer(outputScores, offset: 0, index: 6)
        var queries = queryCount
        var keys = keyCount
        var queryHeads = geometry.queryHeads
        var keyValueHeads = geometry.keyValueHeads
        var headDimension = geometry.headDimension
        var compressRatio = geometry.compressRatio
        var rotaryDimension = geometry.rotaryDimension
        var theta = geometry.ropeTheta
        var normEpsilon = epsilon
        encoder.setBytes(&queries, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&keys, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&queryHeads, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&keyValueHeads, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&headDimension, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&compressRatio, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&rotaryDimension, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 14)
        encoder.setBytes(&normEpsilon, length: MemoryLayout<Float>.stride, index: 15)
        encoder.dispatchThreads(
            MTLSize(width: Int(blockCount), height: Int(queryCount), depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(blockCount), pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }
}

struct Qwen38QSASelectionScratch {
    let state: MTLBuffer
    let tokenMask: MTLBuffer
    let reusableTokenMask: MTLBuffer?
    let reusableKeyCount: UInt32
    let captureTokenMask: MTLBuffer?

    init(state: MTLBuffer,
         tokenMask: MTLBuffer,
         reusableTokenMask: MTLBuffer? = nil,
         reusableKeyCount: UInt32 = 0,
         captureTokenMask: MTLBuffer? = nil) {
        precondition(reusableKeyCount == 0 || reusableTokenMask != nil,
                     "QSA reuse requires a retained token mask")
        self.state = state
        self.tokenMask = tokenMask
        self.reusableTokenMask = reusableTokenMask
        self.reusableKeyCount = reusableKeyCount
        self.captureTokenMask = captureTokenMask
    }
}

final class Qwen38QSASelector {
    static let stateBytesPerQuery = 5 * MemoryLayout<UInt32>.stride

    private let initializePipeline: MTLComputePipelineState
    private let radixPipeline: MTLComputePipelineState
    private let maskPipeline: MTLComputePipelineState
    let geometry: Qwen38QSAGeometry

    init(context: MetalContext, geometry: Qwen38QSAGeometry = .qwen) throws {
        precondition(geometry.compressRatio > 0)
        precondition(geometry.tokenBudget.isMultiple(of: geometry.compressRatio))
        self.geometry = geometry
        self.initializePipeline = try context.pipeline("qwen38_qsa_selection_initialize")
        self.radixPipeline = try context.pipeline(
            "qwen38_qsa_selection_radix",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.maskPipeline = try context.pipeline("qwen38_qsa_selection_mask")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                scores: MTLBuffer,
                visibleTokenCounts: MTLBuffer,
                scratch: Qwen38QSASelectionScratch,
                queryCount: UInt32,
                keyCount: UInt32) {
        precondition(queryCount > 0 && keyCount > 0)
        precondition(scratch.state.length >= Int(queryCount) * Self.stateBytesPerQuery)
        precondition(scratch.tokenMask.length >= Int(queryCount * keyCount))
        let blockCount = keyCount / geometry.compressRatio
        precondition(scores.length >= Int(queryCount * blockCount) * MemoryLayout<Float>.stride)
        precondition(visibleTokenCounts.length >= Int(queryCount) * MemoryLayout<UInt32>.stride)

        if let reusableTokenMask = scratch.reusableTokenMask {
            encodeReusedMask(
                commandBuffer: commandBuffer,
                source: reusableTokenMask,
                sourceKeyCount: scratch.reusableKeyCount,
                tokenMask: scratch.tokenMask,
                queryCount: queryCount,
                keyCount: keyCount)
        } else {
            encodeInitialize(
                commandBuffer: commandBuffer,
                visibleTokenCounts: visibleTokenCounts,
                state: scratch.state,
                queryCount: queryCount,
                keyCount: keyCount)
            if blockCount > 0 {
                for shift: UInt32 in [24, 16, 8, 0] {
                    encodeRadix(
                        commandBuffer: commandBuffer,
                        scores: scores,
                        state: scratch.state,
                        queryCount: queryCount,
                        blockCount: blockCount,
                        shift: shift)
                }
            }
            encodeMask(
                commandBuffer: commandBuffer,
                scores: scores,
                state: scratch.state,
                tokenMask: scratch.tokenMask,
                queryCount: queryCount,
                keyCount: keyCount,
                blockCount: blockCount)
        }
        if let captureTokenMask = scratch.captureTokenMask {
            encodeCapture(
                commandBuffer: commandBuffer,
                tokenMask: scratch.tokenMask,
                captureTokenMask: captureTokenMask,
                queryCount: queryCount,
                keyCount: keyCount)
        }
    }

    private func encodeReusedMask(commandBuffer: MTLCommandBuffer,
                                  source: MTLBuffer,
                                  sourceKeyCount: UInt32,
                                  tokenMask: MTLBuffer,
                                  queryCount: UInt32,
                                  keyCount: UInt32) {
        precondition(sourceKeyCount > 0 && sourceKeyCount <= keyCount)
        let sourceStride = Int(sourceKeyCount)
        let destinationStride = Int(keyCount)
        precondition(source.length >= Int(queryCount) * sourceStride)
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        for query in 0..<Int(queryCount) {
            let sourceOffset = query * sourceStride
            let destinationOffset = query * destinationStride
            blit.copy(
                from: source,
                sourceOffset: sourceOffset,
                to: tokenMask,
                destinationOffset: destinationOffset,
                size: sourceStride)
            if sourceKeyCount < keyCount {
                blit.fill(
                    buffer: tokenMask,
                    range: (destinationOffset + sourceStride)..<(destinationOffset + destinationStride),
                    value: 1)
            }
        }
        blit.endEncoding()
    }

    private func encodeCapture(commandBuffer: MTLCommandBuffer,
                               tokenMask: MTLBuffer,
                               captureTokenMask: MTLBuffer,
                               queryCount: UInt32,
                               keyCount: UInt32) {
        let bytes = Int(queryCount * keyCount)
        precondition(captureTokenMask.length >= bytes)
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(
            from: tokenMask,
            sourceOffset: 0,
            to: captureTokenMask,
            destinationOffset: 0,
            size: bytes)
        blit.endEncoding()
    }

    private func encodeInitialize(commandBuffer: MTLCommandBuffer,
                                  visibleTokenCounts: MTLBuffer,
                                  state: MTLBuffer,
                                  queryCount: UInt32,
                                  keyCount: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(initializePipeline)
        encoder.setBuffer(visibleTokenCounts, offset: 0, index: 0)
        encoder.setBuffer(state, offset: 0, index: 1)
        var queries = queryCount
        var keys = keyCount
        var ratio = geometry.compressRatio
        var topK = geometry.blockTopK
        encoder.setBytes(&queries, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&keys, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&ratio, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&topK, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: Int(queryCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(queryCount), initializePipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }

    private func encodeRadix(commandBuffer: MTLCommandBuffer,
                             scores: MTLBuffer,
                             state: MTLBuffer,
                             queryCount: UInt32,
                             blockCount: UInt32,
                             shift: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(radixPipeline)
        encoder.setBuffer(scores, offset: 0, index: 0)
        encoder.setBuffer(state, offset: 0, index: 1)
        var blocks = blockCount
        var byteShift = shift
        encoder.setBytes(&blocks, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&byteShift, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(queryCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeMask(commandBuffer: MTLCommandBuffer,
                            scores: MTLBuffer,
                            state: MTLBuffer,
                            tokenMask: MTLBuffer,
                            queryCount: UInt32,
                            keyCount: UInt32,
                            blockCount: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(maskPipeline)
        encoder.setBuffer(scores, offset: 0, index: 0)
        encoder.setBuffer(state, offset: 0, index: 1)
        encoder.setBuffer(tokenMask, offset: 0, index: 2)
        var queries = queryCount
        var keys = keyCount
        var blocks = blockCount
        var ratio = geometry.compressRatio
        encoder.setBytes(&queries, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&keys, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&blocks, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&ratio, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(queryCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(queryCount), maskPipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1))
        encoder.endEncoding()
    }
}
