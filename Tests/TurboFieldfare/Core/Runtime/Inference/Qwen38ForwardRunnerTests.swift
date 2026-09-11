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
    func targetLayerCountDefaultsToFullDepthAndAcceptsBoundedPrefix() {
        #expect(Qwen38ForwardRunner.resolveTargetLayerCount(
            requested: nil,
            modelLayerCount: 48,
            pleLayer: 1) == 48)
        #expect(Qwen38ForwardRunner.resolveTargetLayerCount(
            requested: 2,
            modelLayerCount: 48,
            pleLayer: 1) == 2)
    }

    @Test
    func targetLayerCountRejectsPrefixesBeforePLEAndBeyondModel() {
        #expect(Qwen38ForwardRunner.resolveTargetLayerCount(
            requested: 1,
            modelLayerCount: 48,
            pleLayer: 1) == nil)
        #expect(Qwen38ForwardRunner.resolveTargetLayerCount(
            requested: 49,
            modelLayerCount: 48,
            pleLayer: 1) == nil)
    }

    @Test
    func semanticValidityRejectsEmptyAndAllZeroTokenSequences() {
        #expect(Qwen38SemanticValidity.from(tokenIDs: []) == .emptyTokenIDs)
        #expect(Qwen38SemanticValidity.from(tokenIDs: [0, 0, 0]) == .allZeroTokenIDs)
        #expect(Qwen38SemanticValidity.from(tokenIDs: [0, 1, 0]) == .valid)
    }

    @Test
    func semanticValidityRoundTripsThroughCodable() throws {
        let encoded = try JSONEncoder().encode(Qwen38SemanticValidity.allZeroTokenIDs)
        let decoded = try JSONDecoder().decode(
            Qwen38SemanticValidity.self, from: encoded)

        #expect(decoded == .allZeroTokenIDs)
        #expect(String(data: encoded, encoding: .utf8) == "\"invalid-all-zero-token-ids\"")
    }

    @Test
    func pleRejectsPackedWeightsWithZeroAffineCompanions() throws {
        let context = try MetalContext()
        let buffer = try #require(context.device.makeBuffer(
            length: 20, options: .storageModeShared))
        buffer.contents().assumingMemoryBound(to: UInt8.self)[0] = 0x11
        let projection = Qwen38PLEQuantizedProjection(
            weights: buffer,
            scales: buffer,
            scalesOffset: 16,
            biases: buffer,
            biasesOffset: 18)

        #expect {
            try projection.validateCompanions(
                rows: 1, columns: 32, field: "testProjection")
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else {
                return false
            }
            return detail.contains("zero affine companions")
        }
    }

    @Test
    func batchLogitLayoutProvidesContiguousPerPositionRows() throws {
        let layout = try Qwen38BatchLogitLayout(
            tokenCount: 3,
            vocabularySize: 7)

        #expect(layout.tokenCount == 3)
        #expect(layout.vocabularySize == 7)
        #expect(layout.rowByteStride == 14)
        #expect(layout.byteCount == 42)
        #expect(try layout.offset(for: 0) == 0)
        #expect(try layout.offset(for: 1) == 14)
        #expect(try layout.offset(for: 2) == 28)
    }

    @Test
    func batchLogitLayoutRejectsInvalidRows() throws {
        let layout = try Qwen38BatchLogitLayout(
            tokenCount: 2,
            vocabularySize: 7)

        #expect(layout.byteCount == 28)
        #expect(try layout.offset(for: 1) == 14)
        #expect {
            try layout.offset(for: 2)
        } throws: { error in
            guard case PrefillError.chunkedUnsupported(let reason) = error else {
                return false
            }
            return reason.contains("outside 2 rows")
        }
    }

    @Test
    func finalHeadUsesMixedRowsOnlyForBatchLogits() throws {
        let context = try MetalContext()
        let finalHidden = try #require(context.device.makeBuffer(
            length: 8, options: .storageModeShared))
        let mixedInput = try #require(context.device.makeBuffer(
            length: 8, options: .storageModeShared))
        let logitsRows = try #require(context.device.makeBuffer(
            length: 8, options: .storageModeShared))

        let batchInput = Qwen38ForwardRunner.finalHeadInput(
            logitsRows: logitsRows,
            finalHidden: finalHidden,
            mixedInput: mixedInput)
        let scalarInput = Qwen38ForwardRunner.finalHeadInput(
            logitsRows: nil,
            finalHidden: finalHidden,
            mixedInput: mixedInput)

        #expect(batchInput === mixedInput)
        #expect(scalarInput === finalHidden)
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

    @Test
    func stageCaptureRoundTripsOwnedBoundaryMetadata() throws {
        let capture = Qwen38StageCapture(
            layerIndex: 1,
            stage: "layer-output",
            tokenPosition: 7,
            inputToken: 123,
            shape: [4, 2560],
            values: [Float16(1), Float16(-2)])

        let encoded = try JSONEncoder().encode(capture)
        let decoded = try JSONDecoder().decode(
            Qwen38StageCapture.self, from: encoded)

        #expect(decoded == capture)
    }

    @Test
    func routerDiagnosticsRoundTripOwnedRoutePayload() throws {
        let diagnostics = Qwen38RouterDiagnostics(
            layerIndex: 0,
            tokenIndex: 1,
            routerLogits: [1.0, -2.0],
            selectedExperts: [7, 11],
            routeWeightBits: [0x3c00, 0x3800])

        let encoded = try JSONEncoder().encode(diagnostics)
        let decoded = try JSONDecoder().decode(
            Qwen38RouterDiagnostics.self, from: encoded)

        #expect(decoded == diagnostics)
    }

    @Test
    func targetBoundarySnapshotDefaultsToNoStageCapturesOrRouterDiagnostics() {
        let snapshot = Qwen38TargetBoundarySnapshot(
            targetPosition: 7,
            inputToken: 123,
            streamCount: 4,
            hiddenSize: 2560,
            targetHiddenStreams: [],
            rawTargetHiddenStreams: nil)

        #expect(snapshot.stageCaptures.isEmpty)
        #expect(snapshot.routerDiagnostics == nil)
    }
}
