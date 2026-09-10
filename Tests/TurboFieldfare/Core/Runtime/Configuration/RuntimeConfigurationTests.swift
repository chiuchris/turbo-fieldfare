import Testing
@testable import TurboFieldfare

@Suite struct RuntimeConfigurationTests {
    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
        #expect(!runtime.qwenGPUStageTimingEnabled)
        #expect(runtime.qwenGPUExecutionMode == .ordered)
        #expect(runtime.ngramRowProfileMaxRows == 0)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true,
            qwenGPUStageTimingEnabled: true,
            qwenGPUExecutionMode: .parallelDeltaProjections,
            ngramRowProfileMaxRows: 128,
            ngramPinnedRows: [11, 23],
            ngramPinnedRowBytes: 1024)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.ngramRowProfileMaxRows == 128)
        #expect(runtime.ngramPinnedRows == [11, 23])
        #expect(runtime.ngramPinnedRowBytes == 1024)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
        #expect(runtime.qwenGPUStageTimingEnabled)
        #expect(runtime.qwenGPUExecutionMode == .parallelDeltaProjections)
    }

    @Test func qwenGPUExecutionModesKeepStableWireValues() {
        #expect(QwenGPUExecutionMode.ordered.rawValue == "ordered")
        #expect(QwenGPUExecutionMode.parallelDeltaProjections.rawValue
                == "parallel-delta-projections")
    }

    @Test(arguments: [32, 64, 128])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
