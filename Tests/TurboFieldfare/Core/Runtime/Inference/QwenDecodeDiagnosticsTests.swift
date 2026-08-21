import Testing
@testable import TurboFieldfare

struct QwenDecodeDiagnosticsTests {
    @Test func preservesMeasuredDecodeCounters() {
        let diagnostics = QwenDecodeDiagnostics(
            wallNanos: 100,
            embeddingNanos: 10,
            layerNanos: 70,
            logitsNanos: 20,
            expertFetchNanos: 30,
            layerCount: 4,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 3,
            commandBufferCount: 9,
            routerEvaluationCount: 4,
            routedExpertCount: 8,
            routedExpertCacheHitCount: 5,
            routedExpertCacheMissCount: 3,
            routedExpertEstimatedBytes: 1234)

        #expect(diagnostics.wallNanos == 100)
        #expect(diagnostics.embeddingNanos == 10)
        #expect(diagnostics.layerNanos == 70)
        #expect(diagnostics.logitsNanos == 20)
        #expect(diagnostics.expertFetchNanos == 30)
        #expect(diagnostics.layerCount == 4)
        #expect(diagnostics.fullAttentionLayerCount == 1)
        #expect(diagnostics.deltaNetLayerCount == 3)
        #expect(diagnostics.commandBufferCount == 9)
        #expect(diagnostics.routerEvaluationCount == 4)
        #expect(diagnostics.routedExpertCount == 8)
        #expect(diagnostics.routedExpertCacheHitCount == 5)
        #expect(diagnostics.routedExpertCacheMissCount == 3)
        #expect(diagnostics.routedExpertEstimatedBytes == 1234)
        #expect(diagnostics.layers.isEmpty)
    }
}