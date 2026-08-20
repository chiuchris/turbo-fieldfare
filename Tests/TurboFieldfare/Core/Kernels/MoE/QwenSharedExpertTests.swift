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