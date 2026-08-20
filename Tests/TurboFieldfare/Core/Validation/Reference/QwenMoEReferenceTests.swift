import Testing
import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QwenMoEReferenceTests {
    @Test("Qwen SiLU and sigmoid match stable scalar definitions")
    func activationsMatchDefinitions() {
        #expect(abs(QwenMoeRef.silu(0) - 0) < 1e-7)
        #expect(abs(QwenMoeRef.silu(1) - 0.7310586) < 1e-6)
        #expect(abs(QwenMoeRef.sigmoid(0) - 0.5) < 1e-7)
        #expect(abs(QwenMoeRef.sigmoid(20) - 0.999999998) < 1e-7)
        #expect(abs(QwenMoeRef.sigmoid(-20) - 0.000000002) < 1e-7)
    }

    @Test("Qwen router selects top-k with deterministic ties and normalized weights")
    func routerSelectsDeterministically() {
        let routing = QwenMoeRef.route(
            logits: [1, 4, 4, 3, 2, 4, -1, 0], topK: 4)
        #expect(routing.indices == [1, 2, 5, 3])
        #expect(abs(routing.weights.reduce(0, +) - 1) < 1e-6)
        #expect(routing.weights.allSatisfy { $0 > 0 })
    }

    @Test("Qwen reference MoE combines routed and gated shared branches")
    func referenceBlockRunsAtToyShape() {
        let configuration = QwenMoeRef.Configuration(
            hiddenSize: 128, intermediateSize: 64, numExperts: 8, topK: 4)
        var rng = SeedTree(0x936).key("qwen-moe-ref")

        func int4Rows(_ rows: Int, _ cols: Int) -> [Quantization.Int4AffineRow] {
            (0..<rows).map { _ in
                Quantization.quantizeInt4Affine(
                    (0..<cols).map { _ in rng.uniform(-0.25, 0.25) })
            }
        }

        func int8Row(_ cols: Int) -> Quantization.Int8AffineRow {
            Quantization.quantizeInt8Affine(
                (0..<cols).map { _ in rng.uniform(-0.25, 0.25) })
        }

        let x = (0..<configuration.hiddenSize).map { _ in rng.uniform(-0.5, 0.5) }
        let residual = (0..<configuration.hiddenSize).map { _ in rng.uniform(-0.5, 0.5) }
        let output = QwenMoeRef.apply(
            x: x,
            residual: residual,
            routerRows: (0..<configuration.numExperts).map { _ in int8Row(configuration.hiddenSize) },
            routedGate: (0..<configuration.numExperts).map { _ in int4Rows(configuration.intermediateSize, configuration.hiddenSize) },
            routedUp: (0..<configuration.numExperts).map { _ in int4Rows(configuration.intermediateSize, configuration.hiddenSize) },
            routedDown: (0..<configuration.numExperts).map { _ in int4Rows(configuration.hiddenSize, configuration.intermediateSize) },
            sharedGate: int4Rows(configuration.intermediateSize, configuration.hiddenSize),
            sharedUp: int4Rows(configuration.intermediateSize, configuration.hiddenSize),
            sharedDown: int4Rows(configuration.hiddenSize, configuration.intermediateSize),
            sharedRouterGate: int8Row(configuration.hiddenSize),
            configuration: configuration)

        #expect(output.count == configuration.hiddenSize)
        #expect(output.allSatisfy { $0.isFinite })
    }
}