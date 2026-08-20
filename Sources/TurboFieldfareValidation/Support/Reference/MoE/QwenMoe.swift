import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the Qwen3.5 MoE block.
///
/// Qwen uses a normalized top-k router, SiLU routed and shared experts, and a
/// sigmoid gate on the shared branch. The production implementation may use
/// different kernels and summation order; this reference deliberately keeps
/// the scalar control flow explicit for independent comparison.
public enum QwenMoeRef {
    public struct Configuration: Sendable, Equatable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numExperts: Int
        public let topK: Int

        public init(hiddenSize: Int = 2048,
                    intermediateSize: Int = 512,
                    numExperts: Int = 256,
                    topK: Int = 8) {
            precondition(hiddenSize > 0)
            precondition(intermediateSize > 0)
            precondition(numExperts > 0)
            precondition(topK > 0 && topK <= numExperts)
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numExperts = numExperts
            self.topK = topK
        }
    }

    public struct Routing: Equatable, Sendable {
        public let indices: [Int]
        public let weights: [Float]

        public init(indices: [Int], weights: [Float]) {
            precondition(indices.count == weights.count)
            self.indices = indices
            self.weights = weights
        }
    }

    public static func silu(_ x: Float) -> Float {
        x / (1.0 + Foundation.exp(-x))
    }

    public static func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            let z = Foundation.exp(-x)
            return 1.0 / (1.0 + z)
        }
        let z = Foundation.exp(x)
        return z / (1.0 + z)
    }

    public static func route(logits: [Float], topK: Int) -> Routing {
        precondition(topK > 0 && topK <= logits.count)
        let ranked = logits.indices.sorted { lhs, rhs in
            logits[lhs] == logits[rhs] ? lhs < rhs : logits[lhs] > logits[rhs]
        }
        let indices = Array(ranked.prefix(topK))
        let maximum = logits[indices[0]]
        let unnormalized = indices.map { Foundation.exp(logits[$0] - maximum) }
        let total = unnormalized.reduce(0, +)
        return Routing(indices: indices, weights: unnormalized.map { $0 / total })
    }

    public static func runExpert(
        gateRows: [Quantization.Int4AffineRow],
        upRows: [Quantization.Int4AffineRow],
        downRows: [Quantization.Int4AffineRow],
        x: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(gateRows.count == f)
        precondition(upRows.count == f)
        precondition(downRows.count == d)
        precondition(x.count == d)

        let gate = DequantInt4GemvRef.apply(weightRows: gateRows, x: x, n: d)
        let up = DequantInt4GemvRef.apply(weightRows: upRows, x: x, n: d)
        let activation = zip(gate, up).map { silu($0) * $1 }
        return DequantInt4GemvRef.apply(weightRows: downRows, x: activation, n: f)
    }

    public static func apply(
        x: [Float],
        residual: [Float],
        routerRows: [Quantization.Int8AffineRow],
        routedGate: [[Quantization.Int4AffineRow]],
        routedUp: [[Quantization.Int4AffineRow]],
        routedDown: [[Quantization.Int4AffineRow]],
        sharedGate: [Quantization.Int4AffineRow],
        sharedUp: [Quantization.Int4AffineRow],
        sharedDown: [Quantization.Int4AffineRow],
        sharedRouterGate: Quantization.Int8AffineRow,
        configuration: Configuration
    ) -> [Float] {
        let d = configuration.hiddenSize
        let f = configuration.intermediateSize
        precondition(x.count == d)
        precondition(residual.count == d)
        precondition(routerRows.count == configuration.numExperts)
        precondition(routedGate.count == configuration.numExperts)
        precondition(routedUp.count == configuration.numExperts)
        precondition(routedDown.count == configuration.numExperts)

        let logits = DequantInt8GemvRef.apply(
            weightRows: routerRows, x: x, n: d)
        let routing = route(logits: logits, topK: configuration.topK)
        var output = residual
        for (slot, expert) in routing.indices.enumerated() {
            let expertOutput = runExpert(
                gateRows: routedGate[expert],
                upRows: routedUp[expert],
                downRows: routedDown[expert],
                x: x,
                d: d,
                f: f)
            var weight = routing.weights[slot]
            var scaled = [Float](repeating: 0, count: d)
            vDSP_vsmul(expertOutput, 1, &weight, &scaled, 1, vDSP_Length(d))
            vDSP_vadd(output, 1, scaled, 1, &output, 1, vDSP_Length(d))
        }

        let sharedOutput = runExpert(
            gateRows: sharedGate,
            upRows: sharedUp,
            downRows: sharedDown,
            x: x,
            d: d,
            f: f)
        let sharedLogit = DequantInt8GemvRef.apply(
            weightRows: [sharedRouterGate], x: x, n: d)[0]
        var sharedWeight = sigmoid(sharedLogit)
        var scaledShared = [Float](repeating: 0, count: d)
        vDSP_vsmul(sharedOutput, 1, &sharedWeight, &scaledShared, 1, vDSP_Length(d))
        vDSP_vadd(output, 1, scaledShared, 1, &output, 1, vDSP_Length(d))
        return output
    }
}