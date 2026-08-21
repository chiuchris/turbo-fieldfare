import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QwenSharedExpertTests {
    private static let d = 128
    private static let f = 64

    @Test func qwenSharedExpertAndGateMatchReference() throws {
        var rng = SplitMix64(seed: 0x938)
        let x = (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = Self.makeRows(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeRows(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeRows(rows: Self.d, cols: Self.f, rng: &rng)
        let sharedGate = Quantization.quantizeInt8Affine(
            (0..<Self.d).map { _ in rng.uniform(-0.25, 0.25) })
        let expectedShared = QwenMoeRef.runExpert(
            gateRows: gate,
            upRows: up,
            downRows: down,
            x: x,
            d: Self.d,
            f: Self.f)
        let sharedLogit = DequantInt8GemvRef.apply(
            weightRows: [sharedGate], x: x, n: Self.d)[0]
        let sharedWeight = QwenMoeRef.sigmoid(sharedLogit)
        let expected = expectedShared.map { $0 * sharedWeight }

        let context = try MetalContext()
        let kernel = try QwenMoE(context: context)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let sharedOutput = try #require(Fp16Buffer.make(context.device, count: Self.d))
        let routedOutput = try #require(Fp16Buffer.make(context.device, count: Self.d))
        let output = try #require(Fp16Buffer.make(context.device, count: Self.d))
        let scratchGate = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let scratchUp = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let scratchAct = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeSharedExpert(
            commandBuffer: commandBuffer,
            x: xBuffer,
            gate: Self.projection(context, gate, rows: Self.f, cols: Self.d),
            up: Self.projection(context, up, rows: Self.f, cols: Self.d),
            down: Self.projection(context, down, rows: Self.d, cols: Self.f),
            y: sharedOutput,
            scratchGate: scratchGate,
            scratchUp: scratchUp,
            scratchAct: scratchAct)
        kernel.encodeSharedGateAndCombine(
            commandBuffer: commandBuffer,
            x: xBuffer,
            gate: Self.projection(context, sharedGate, rows: 1, cols: Self.d),
            sharedOutput: sharedOutput,
            routedOutput: routedOutput,
            y: output,
            d: UInt32(Self.d))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        let actual = Fp16Buffer.read(output, count: Self.d)
        #expect(RelError.compute(actual: actual, reference: expected) < Tolerance.quantInt4 * 5)
    }

    @Test(arguments: [1, 2, 31, 32, 127, 128, 129])
    func qwenSharedExpertBlockMatchesReference(queryCount: Int) throws {
        var rng = SplitMix64(seed: 0x939 + UInt64(queryCount))
        let inputs = (0..<(queryCount * Self.d)).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = Self.makeRows(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeRows(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeRows(rows: Self.d, cols: Self.f, rng: &rng)
        let expected = (0..<queryCount).flatMap { row in
            let start = row * Self.d
            let x = inputs[start..<(start + Self.d)].map { Float(Float16($0)) }
            return QwenMoeRef.runExpert(
                gateRows: gate,
                upRows: up,
                downRows: down,
                x: x,
                d: Self.d,
                f: Self.f)
        }

        let context = try MetalContext()
        let kernel = try QwenMoE(context: context)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: inputs))
        let output = try #require(Fp16Buffer.make(context.device, count: queryCount * Self.d))
        let scratchGate = try #require(Fp16Buffer.make(context.device, count: queryCount * Self.f))
        let scratchUp = try #require(Fp16Buffer.make(context.device, count: queryCount * Self.f))
        let scratchAct = try #require(Fp16Buffer.make(context.device, count: queryCount * Self.f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeSharedExpertBlock(
            commandBuffer: commandBuffer,
            x: xBuffer,
            y: output,
            gate: Self.projection(context, gate, rows: Self.f, cols: Self.d),
            up: Self.projection(context, up, rows: Self.f, cols: Self.d),
            down: Self.projection(context, down, rows: Self.d, cols: Self.f),
            scratchGate: scratchGate,
            scratchUp: scratchUp,
            scratchAct: scratchAct,
            queryCount: queryCount,
            d: Self.d,
            intermediate: Self.f,
            xStrideElements: Self.d,
            yStrideElements: Self.d)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = Fp16Buffer.read(output, count: queryCount * Self.d)
        let maxAbs = RelError.maxAbsDiff(actual, expected)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(maxAbs < Tolerance.quantInt4 * 5, "maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel < Tolerance.quantInt4 * 5, "rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test(arguments: [1, 2, 31, 129])
    func qwenSharedGateAndCombineBlockMatchesReference(queryCount: Int) throws {
        var rng = SplitMix64(seed: 0x93A + UInt64(queryCount))
        let inputs = (0..<(queryCount * Self.d)).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = Self.makeRows(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeRows(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeRows(rows: Self.d, cols: Self.f, rng: &rng)
        let sharedGate = Quantization.quantizeInt8Affine(
            (0..<Self.d).map { _ in rng.uniform(-0.25, 0.25) })
        let expected = (0..<queryCount).flatMap { row in
            let start = row * Self.d
            let x = inputs[start..<(start + Self.d)].map { Float(Float16($0)) }
            let shared = QwenMoeRef.runExpert(
                gateRows: gate,
                upRows: up,
                downRows: down,
                x: x,
                d: Self.d,
                f: Self.f)
            let logit = DequantInt8GemvRef.apply(
                weightRows: [sharedGate],
                x: x,
                n: Self.d)[0]
            return shared.map { $0 * QwenMoeRef.sigmoid(logit) }
        }

        let context = try MetalContext()
        let kernel = try QwenMoE(context: context)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: inputs))
        let sharedOutput = try #require(Fp16Buffer.make(
            context.device, count: queryCount * Self.d))
        let routedOutput = try #require(Fp16Buffer.make(
            context.device,
            values: [Float](repeating: 0, count: queryCount * Self.d)))
        let output = try #require(Fp16Buffer.make(
            context.device, count: queryCount * Self.d))
        let scratchGate = try #require(Fp16Buffer.make(
            context.device, count: queryCount * Self.f))
        let scratchUp = try #require(Fp16Buffer.make(
            context.device, count: queryCount * Self.f))
        let scratchAct = try #require(Fp16Buffer.make(
            context.device, count: queryCount * Self.f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeSharedExpertBlock(
            commandBuffer: commandBuffer,
            x: xBuffer,
            y: sharedOutput,
            gate: Self.projection(context, gate, rows: Self.f, cols: Self.d),
            up: Self.projection(context, up, rows: Self.f, cols: Self.d),
            down: Self.projection(context, down, rows: Self.d, cols: Self.f),
            scratchGate: scratchGate,
            scratchUp: scratchUp,
            scratchAct: scratchAct,
            queryCount: queryCount,
            d: Self.d,
            intermediate: Self.f,
            xStrideElements: Self.d,
            yStrideElements: Self.d)
        kernel.encodeSharedGateAndCombineBlock(
            commandBuffer: commandBuffer,
            x: xBuffer,
            gate: Self.projection(context, sharedGate, rows: 1, cols: Self.d),
            sharedOutput: sharedOutput,
            routedOutput: routedOutput,
            y: output,
            queryCount: queryCount,
            d: UInt32(Self.d))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = Fp16Buffer.read(output, count: queryCount * Self.d)
        let maxAbs = RelError.maxAbsDiff(actual, expected)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(maxAbs < Tolerance.quantInt4 * 5, "maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel < Tolerance.quantInt4 * 5, "rel=\(rel) maxAbs=\(maxAbs)")
    }

    private static func makeRows(
        rows: Int,
        cols: Int,
        rng: inout SplitMix64
    ) -> [Quantization.Int4AffineRow] {
        (0..<rows).map { _ in
            Quantization.quantizeInt4Affine(
                (0..<cols).map { _ in rng.uniform(-0.4, 0.4) })
        }
    }

    private static func projection(
        _ context: MetalContext,
        _ rows: [Quantization.Int4AffineRow],
        rows rowCount: Int,
        cols: Int
    ) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: context.device.makeBuffer(
                bytes: rows.flatMap(\.packed),
                length: rows.flatMap(\.packed).count,
                options: .storageModeShared)!,
            scales: context.device.makeBuffer(
                bytes: rows.flatMap(\.scales),
                length: rows.flatMap(\.scales).count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)!,
            biases: context.device.makeBuffer(
                bytes: rows.flatMap(\.biases),
                length: rows.flatMap(\.biases).count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)!,
            rows: UInt32(rowCount),
            cols: UInt32(cols))
    }

    private static func projection(
        _ context: MetalContext,
        _ row: Quantization.Int8AffineRow,
        rows rowCount: Int,
        cols: Int
    ) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: context.device.makeBuffer(
                bytes: row.packed, length: row.packed.count, options: .storageModeShared)!,
            scales: context.device.makeBuffer(
                bytes: row.scales,
                length: row.scales.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)!,
            biases: context.device.makeBuffer(
                bytes: row.biases,
                length: row.biases.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)!,
            rows: UInt32(rowCount),
            cols: UInt32(cols))
    }
}