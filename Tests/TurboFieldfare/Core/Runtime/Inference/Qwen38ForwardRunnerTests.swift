import Metal
import Testing
@testable import TurboFieldfare

@Suite
struct Qwen38ForwardRunnerTests {
    @Test
    func rejectsNonQwen38ModelBeforeRuntimeAllocation() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: .gemma4Toy())

        #expect(throws: ModelError.self) {
            try Qwen38ForwardRunner(
                model: model,
                context: context,
                maxContext: 4)
        }
    }

    @Test
    func factoryDispatchesQwen38IntoRunnerValidation() throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let context = try MetalContext()
        let base = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: .gemma4Toy())
        let model = Model(
            device: base.device,
            config: .qwen38FlashNextText,
            streamingMode: base.streamingMode,
            expertCachePolicy: base.expertCachePolicy,
            integrityPolicy: base.integrityPolicy,
            residentBuffer: base.residentBuffer,
            residentIndex: base.residentIndex,
            packedExpertsLayout: base.packedExpertsLayout,
            manifest: base.manifest,
            directoryURL: base.directoryURL,
            modelDirectory: base.modelDirectory,
            trustedInstallReceipt: base.trustedInstallReceipt)
        let expected = Qwen38TensorNames.ple(
            layer: 1, tensor: .keyProjection)

        #expect {
            _ = try ForwardRunnerFactory.make(
                model: model,
                context: context,
                maxContext: 4)
        } throws: { error in
            if case ModelError.tensorNotFound(let name) = error {
                return name == expected
            }
            return false
        }
    }

    @Test
    func qwen38RunnerUsesCanonicalGeometry() {
        let config = ArchConfig.qwen38FlashNextText

        #expect(config.hiddenSize == 2_560)
        #expect(config.numLayers == 48)
        #expect(config.numExperts == 512)
        #expect(config.topKExperts == 10)
        #expect(config.qwen38Architecture?.pleLayerIDs == [2])
        #expect(config.qwen38Architecture?.indexerBudget == 2_048)
    }

    @Test
    func decodeTimingSampleHasStableZeroAndCodableFields() throws {
        #expect(Qwen38DecodeTimingSample.zero == Qwen38DecodeTimingSample(
            embeddingNanos: 0,
            pleNanos: 0,
            attentionRouterNanos: 0,
            deltaNetNanos: 0,
            expertFetchNanos: 0,
            moeNanos: 0,
            finalHeadNanos: 0,
            gpuActiveNanos: 0,
            commandBufferCount: 0,
            commandBufferEncodeNanos: 0,
            commandBufferWaitNanos: 0))

        let sample = Qwen38DecodeTimingSample(
            embeddingNanos: 1,
            pleNanos: 2,
            attentionRouterNanos: 3,
            deltaNetNanos: 10,
            expertFetchNanos: 4,
            expertCacheHits: 8,
            expertCacheMisses: 9,
            moeNanos: 5,
            finalHeadNanos: 6,
            gpuActiveNanos: 7,
            commandBufferCount: 7,
            commandBufferEncodeNanos: 11,
            commandBufferWaitNanos: 12)
        let encoded = try JSONEncoder().encode(sample)
        let decoded = try JSONDecoder().decode(
            Qwen38DecodeTimingSample.self, from: encoded)

        #expect(decoded == sample)
        #expect(decoded.commandBufferCount == 7)
        #expect(decoded.expertFetchNanos == 4)
        #expect(decoded.expertCacheHits == 8)
        #expect(decoded.expertCacheMisses == 9)
        #expect(decoded.deltaNetNanos == 10)
        #expect(decoded.gpuActiveNanos == 7)
        #expect(decoded.commandBufferEncodeNanos == 11)
        #expect(decoded.commandBufferWaitNanos == 12)
    }

    @Test
    func prefillDiagnosticsAggregateCommandBufferTimings() throws {
        var counter = PrefillWorkCounter()
        counter.merge(PrefillWorkDiagnostics(
            executionPath: .chunked,
            scalarForwardCount: 0,
            chunkPassCount: 2,
            commandBufferCount: 3,
            commandBufferEncodeNanos: 11,
            commandBufferWaitNanos: 13))
        counter.merge(PrefillWorkDiagnostics(
            executionPath: .chunked,
            scalarForwardCount: 0,
            chunkPassCount: 4,
            commandBufferCount: 5,
            commandBufferEncodeNanos: 17,
            commandBufferWaitNanos: 19))

        let diagnostics = try #require(counter.diagnostics)
        #expect(diagnostics.executionPath == .chunked)
        #expect(diagnostics.commandBufferCount == 8)
        #expect(diagnostics.commandBufferEncodeNanos == 28)
        #expect(diagnostics.commandBufferWaitNanos == 32)
    }

    @Test
    func prefillDiagnosticsDefaultCommandBufferTimingsToZero() throws {
        var counter = PrefillWorkCounter()
        counter.recordChunkPass()
        counter.recordCommandBuffers(1)

        let diagnostics = try #require(counter.diagnostics)
        #expect(diagnostics.commandBufferEncodeNanos == 0)
        #expect(diagnostics.commandBufferWaitNanos == 0)
    }

    @Test
    func mtpVerificationAcceptsCompleteProposalBlock() {
        let verification = GreedyBlockVerification(
            targetTokens: [11, 12, 13],
            proposedTokens: [11, 12, 13],
            startPosition: 20)

        #expect(verification.acceptedTokenCount == 3)
        #expect(verification.statePosition == 23)
        #expect(verification.targetTokens == [11, 12, 13])
    }

    @Test
    func mtpVerificationReplaysBoundaryAfterFirstRejection() {
        let verification = GreedyBlockVerification(
            targetTokens: [99, 12, 13],
            proposedTokens: [11, 12, 13],
            startPosition: 20)

        #expect(verification.acceptedTokenCount == 0)
        #expect(verification.statePosition == 21)
    }

    @Test
    func mtpVerificationCommitsAcceptedPrefixAndBoundary() {
        let verification = GreedyBlockVerification(
            targetTokens: [11, 22, 13],
            proposedTokens: [11, 12, 13],
            startPosition: 20)

        #expect(verification.acceptedTokenCount == 1)
        #expect(verification.statePosition == 22)
    }

    @Test
    func draftingDiagnosticsPreserveCandidateBoundaryEvidence() throws {
        let diagnostics = Qwen38DraftingDiagnostics(
            strategy: .experimentalNativeMTP,
            proposedToken: 2_104,
            targetToken: 200_637,
            matchesTarget: false,
            inputToken: 191_372,
            proposalPosition: 7,
            targetPosition: 8)
        let aggregate = DraftingDiagnosticsAggregate(
            strategy: diagnostics.strategy.rawValue,
            draftAttempts: 1,
            proposedTokens: 1,
            acceptedTokens: 0,
            rejectedTokens: 1,
            fallbackCount: 1,
            fallbackReason: "proposal-mismatch",
            lastProposedToken: diagnostics.proposedToken,
            lastTargetToken: diagnostics.targetToken,
            lastMatchesTarget: diagnostics.matchesTarget,
            lastInputToken: diagnostics.inputToken,
            lastProposalPosition: diagnostics.proposalPosition,
            lastTargetPosition: diagnostics.targetPosition)

        let encoded = try JSONEncoder().encode(aggregate)
        let decoded = try JSONDecoder().decode(
            DraftingDiagnosticsAggregate.self, from: encoded)

        #expect(decoded.lastProposedToken == 2_104)
        #expect(decoded.lastTargetToken == 200_637)
        #expect(decoded.lastMatchesTarget == false)
        #expect(decoded.lastInputToken == 191_372)
        #expect(decoded.lastProposalPosition == 7)
        #expect(decoded.lastTargetPosition == 8)
    }

    @Test
    func draftingDiagnosticsDecodeOlderAggregateWithoutBoundaryEvidence() throws {
        let legacy = Data("""
        {"strategy":"disabled","draftAttempts":0,"proposedTokens":0,
        "acceptedTokens":0,"rejectedTokens":0,"fallbackCount":0,
        "fallbackReason":null}
        """.utf8)

        let decoded = try JSONDecoder().decode(
            DraftingDiagnosticsAggregate.self, from: legacy)

        #expect(decoded.lastProposedToken == nil)
        #expect(decoded.lastTargetToken == nil)
        #expect(decoded.lastMatchesTarget == nil)
        #expect(decoded.lastInputToken == nil)
        #expect(decoded.lastProposalPosition == nil)
        #expect(decoded.lastTargetPosition == nil)
    }
}
