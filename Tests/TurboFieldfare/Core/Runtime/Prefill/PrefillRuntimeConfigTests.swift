import Testing
@testable import TurboFieldfare

@Suite struct PrefillRuntimeConfigTests {
    @Test(arguments: [32, 64, 128])
    func productionUsesCompleteChunkedPath(_ chunkTokens: Int) throws {
        let config = PrefillRuntimeConfig.production(chunkTokens: chunkTokens)
        #expect(config.mode == .chunked)
        #expect(config.chunkTokens == chunkTokens)
    }

    @Test func offDisablesChunkedPrefill() {
        let config = PrefillRuntimeConfig.off
        #expect(config.mode == .off)
        #expect(!config.enabled)
    }

    @Test func plannerUsesConfiguredChunkSize() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 130,
            startPosition: 7,
            config: .production(chunkTokens: 64))
        #expect(spans.map(\.tokenCount) == [64, 64, 2])
        #expect(spans.map(\.startPosition) == [7, 71, 135])
    }

    @Test func diagnosticsPreserveUnknownValues() {
        let diagnostics = PrefillExecutionDiagnostics(
            config: .production(chunkTokens: 128),
            executedMode: .unsupported,
            kvStorageMode: nil,
            unsupportedReason: "unavailable")
        #expect(diagnostics.kvStorageMode == nil)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.unsupportedReason == "unavailable")
    }

    @Test func scalarCallsAreNotReportedAsChunkedWork() throws {
        var counter = PrefillWorkCounter()
        for _ in 0..<128 {
            counter.recordScalarForward()
        }
        counter.recordCommandBuffers(128 * 4)

        let diagnostics = try #require(counter.diagnostics)
        #expect(diagnostics.executionPath == .scalarFallback)
        #expect(diagnostics.scalarForwardCount == 128)
        #expect(diagnostics.chunkPassCount == 0)
        #expect(diagnostics.commandBufferCount == 512)

        let execution = PrefillExecutionDiagnostics(
            config: .production(chunkTokens: 128),
            executedMode: .chunked,
            kvStorageMode: .fp16,
            work: diagnostics)
        #expect(execution.requestedMode == .chunked)
        #expect(execution.executedMode == .scalarFallback)
        #expect(execution.scalarForwardCount == 128)
        #expect(execution.chunkPassCount == 0)
        #expect(execution.commandBufferCount == 512)
    }
}
