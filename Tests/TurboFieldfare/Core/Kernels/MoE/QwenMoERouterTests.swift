import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QwenMoERouterTests {
    private static let experts = 16
    private static let dimension = 128
    private static let topK = 8

    @Test func qwenRouterMatchesReference() throws {
        var rng = SplitMix64(seed: 0x936)
        let weights = (0..<Self.experts).map { expert in
            (0..<Self.dimension).map { _ in
                rng.uniform(-0.05, 0.05) + Float(expert) * 0.01
            }
        }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(-1.0, 1.0) }
        let rows = weights.map(Quantization.quantizeInt8Affine)
        let expected = QwenMoeRef.route(
            logits: DequantInt8GemvRef.apply(weightRows: rows, x: hidden, n: Self.dimension),
            topK: Self.topK)

        let context = try MetalContext()
        let kernel = try QwenMoE(context: context)
        let packed = rows.flatMap(\.packed)
        let scales = rows.flatMap(\.scales)
        let biases = rows.flatMap(\.biases)
        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: packed, length: packed.count, options: .storageModeShared))
        let scaleBuffer = try #require(context.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let biasBuffer = try #require(context.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let hiddenBuffer = try #require(Fp16Buffer.make(context.device, values: hidden))
        let indexBuffer = try #require(context.device.makeBuffer(
            length: Self.topK * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let weightOutput = try #require(Fp16Buffer.make(context.device, count: Self.topK))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeRouter(commandBuffer: commandBuffer,
                            weights: weightBuffer,
                            scales: scaleBuffer,
                            biases: biasBuffer,
                            hidden: hiddenBuffer,
                            outIndices: indexBuffer,
                            outWeights: weightOutput,
                            numExperts: UInt32(Self.experts),
                            d: UInt32(Self.dimension))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: Self.topK)
        let actualIndices = (0..<Self.topK).map { Int(indexPointer[$0]) }
        let actualWeights = Fp16Buffer.read(weightOutput, count: Self.topK)
        #expect(actualIndices == expected.indices)
        let maxError = zip(actualWeights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3)
    }

    @Test func qwenRoutedExpertsMatchSiluReference() throws {
        let d = 128
        let f = 64
        var rng = SplitMix64(seed: 0x937)
        let expertRows = (0..<Self.topK).map { _ in
            (
                gate: Self.makeRows(rows: f, cols: d, rng: &rng),
                up: Self.makeRows(rows: f, cols: d, rng: &rng),
                down: Self.makeRows(rows: d, cols: f, rng: &rng)
            )
        }
        let x = (0..<d).map { _ in rng.uniform(-0.4, 0.4) }
        let routingWeights = [Float](repeating: 1.0 / Float(Self.topK), count: Self.topK)
        var expected = [Float](repeating: 0, count: d)
        for expert in 0..<Self.topK {
            let output = QwenMoeRef.runExpert(
                gateRows: expertRows[expert].gate,
                upRows: expertRows[expert].up,
                downRows: expertRows[expert].down,
                x: x,
                d: d,
                f: f)
            for index in 0..<d {
                expected[index] += output[index] * routingWeights[expert]
            }
        }

        let context = try MetalContext()
        let kernel = try QwenMoE(context: context)
        let packedExperts = expertRows.map {
            Self.makeExpertBlob(gate: $0.gate, up: $0.up, down: $0.down)
        }
        let expertBuffers = try packedExperts.map { blob in
            try #require(context.device.makeBuffer(
                bytes: blob.bytes, length: blob.bytes.count, options: .storageModeShared))
        }
        let argBuffer = try #require(kernel.makeRoutedArgumentBuffer(
            routedBlobs: expertBuffers))
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let actsBuffer = try #require(Fp16Buffer.make(context.device, count: Self.topK * f))
        let weightsBuffer = try #require(
            Fp16Buffer.make(context.device, values: routingWeights))
        let residualBuffer = try #require(Fp16Buffer.make(context.device, count: d))
        let outputBuffer = try #require(Fp16Buffer.make(context.device, count: d))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeRoutedPhase1(commandBuffer: commandBuffer,
                                  routedArgBuffer: argBuffer,
                                  routedBlobs: expertBuffers,
                                  routedOffsets: packedExperts[0].offsets,
                                  x: xBuffer,
                                  acts: actsBuffer,
                                  d: UInt32(d),
                                  f: UInt32(f))
        kernel.encodeRoutedPhase2(commandBuffer: commandBuffer,
                                  routedArgBuffer: argBuffer,
                                  routedBlobs: expertBuffers,
                                  routedOffsets: packedExperts[0].offsets,
                                  acts: actsBuffer,
                                  routingWeights: weightsBuffer,
                                  residual: residualBuffer,
                                  y: outputBuffer,
                                  d: UInt32(d),
                                  f: UInt32(f))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        let actual = Fp16Buffer.read(outputBuffer, count: d)
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

    private static func makeExpertBlob(
        gate: [Quantization.Int4AffineRow],
        up: [Quantization.Int4AffineRow],
        down: [Quantization.Int4AffineRow]
    ) -> (bytes: [UInt8], offsets: MoEExpertOffsets) {
        var bytes: [UInt8] = []
        func appendBytes(_ values: [UInt8]) -> UInt32 {
            let offset = UInt32(bytes.count)
            bytes.append(contentsOf: values)
            return offset
        }
        func appendWords(_ values: [UInt16]) -> UInt32 {
            let offset = UInt32(bytes.count)
            values.withUnsafeBytes { raw in
                bytes.append(contentsOf: raw)
            }
            return offset
        }
        let gateW = appendBytes(gate.flatMap(\.packed))
        let gateS = appendWords(gate.flatMap(\.scales))
        let gateB = appendWords(gate.flatMap(\.biases))
        let upW = appendBytes(up.flatMap(\.packed))
        let upS = appendWords(up.flatMap(\.scales))
        let upB = appendWords(up.flatMap(\.biases))
        let downW = appendBytes(down.flatMap(\.packed))
        let downS = appendWords(down.flatMap(\.scales))
        let downB = appendWords(down.flatMap(\.biases))
        return (
            bytes,
            MoEExpertOffsets(gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                             upWOff: upW, upSOff: upS, upBOff: upB,
                             downWOff: downW, downSOff: downS, downBOff: downB))
    }
}