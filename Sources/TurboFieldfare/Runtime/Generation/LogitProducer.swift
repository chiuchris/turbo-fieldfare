import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementations provide this contract; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

public protocol ForwardRunner: LogitProducer {
    var maxContext: Int { get }
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

public protocol PromptStateSnapshotting: ContinuableLogitProducer {
    func savePromptState()
    func restorePromptState(expectedPosition: Int) throws
}

public protocol FusedGreedyLogitProducer: LogitProducer {
    var usesFusedGreedyHead: Bool { get }
    var lastGreedyToken: UInt32 { get }
}

/// A generation-wide summary of the explicitly enabled drafting arm.
public struct DraftingDiagnosticsAggregate: Codable, Sendable, Equatable {
    public let strategy: String
    public let draftAttempts: Int
    public let proposedTokens: Int
    public let acceptedTokens: Int
    public let rejectedTokens: Int
    public let fallbackCount: Int
    public let fallbackReason: String?
    public let lastProposedToken: Int32?
    public let lastTargetToken: Int32?
    public let lastMatchesTarget: Bool?
    public let lastInputToken: Int32?
    public let lastProposalPosition: Int?
    public let lastTargetPosition: Int?

    public init(strategy: String,
                draftAttempts: Int,
                proposedTokens: Int,
                acceptedTokens: Int,
                rejectedTokens: Int,
                fallbackCount: Int,
                fallbackReason: String?,
                lastProposedToken: Int32? = nil,
                lastTargetToken: Int32? = nil,
                lastMatchesTarget: Bool? = nil,
                lastInputToken: Int32? = nil,
                lastProposalPosition: Int? = nil,
                lastTargetPosition: Int? = nil) {
        self.strategy = strategy
        self.draftAttempts = draftAttempts
        self.proposedTokens = proposedTokens
        self.acceptedTokens = acceptedTokens
        self.rejectedTokens = rejectedTokens
        self.fallbackCount = fallbackCount
        self.fallbackReason = fallbackReason
        self.lastProposedToken = lastProposedToken
        self.lastTargetToken = lastTargetToken
        self.lastMatchesTarget = lastMatchesTarget
        self.lastInputToken = lastInputToken
        self.lastProposalPosition = lastProposalPosition
        self.lastTargetPosition = lastTargetPosition
    }
}

/// Supplies one target-checked greedy proposal from an explicitly enabled
/// drafting producer. Returning nil keeps the target logits authoritative.
public protocol DraftingLogitProducer: LogitProducer {
    var draftingDiagnostics: DraftingDiagnosticsAggregate { get }
    func takeDraftCandidate() -> Int32?
}

public struct GreedyBlockVerification: Sendable, Equatable {
    public let targetTokens: [Int32]
    public let acceptedTokenCount: Int
    public let statePosition: Int

    init(targetTokens: [Int32],
         proposedTokens: [Int32],
         startPosition: Int) {
        precondition(!proposedTokens.isEmpty, "proposedTokens must not be empty")
        precondition(targetTokens.count == proposedTokens.count,
                     "target and proposed token counts must match")
        let accepted = zip(targetTokens, proposedTokens)
            .prefix { target, proposed in target == proposed }
            .count
        self.targetTokens = targetTokens
        self.acceptedTokenCount = accepted
        self.statePosition = startPosition + min(proposedTokens.count, accepted + 1)
    }
}

public protocol GreedyBlockVerifyingLogitProducer: LogitProducer {
    func verifyGreedyBlock(boundaryToken: Int32,
                           proposedTokens: ArraySlice<Int32>,
                           startPosition: Int,
                           config: PrefillRuntimeConfig) async throws
        -> GreedyBlockVerification
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed
    public let work: PrefillWorkDiagnostics?

    public init(newPosition: Int,
                seed: PrefillSeed,
                work: PrefillWorkDiagnostics? = nil) {
        self.newPosition = newPosition
        self.seed = seed
        self.work = work
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

protocol MultimodalPrefillRunner: LogitProducer {
    func prefillMultimodal(input: MultimodalPrefillInput,
                           startPosition: Int,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult
}
