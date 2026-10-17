import Testing
@testable import TurboFieldfareServerCore

/// A cancelled request used to be excluded from logging entirely, on the
/// reasoning that a client hanging up is not a server fault. That is true, and it
/// left the request's last line as `generating` forever: reading the log, a
/// running request, an abandoned one and a crashed one were indistinguishable.
@Suite struct ServerLogTests {
    @Test func aCancelledRequestReadsAsCancelledRatherThanFailed() {
        let line = ServerLog.cancelledMessage(id: "chatcmpl-abc",
                                              phase: "generating",
                                              duration: .milliseconds(5369))

        #expect(line.contains("chatcmpl-abc"))
        #expect(line.contains("cancelled by client"))
        #expect(line.contains("phase=generating"))
        #expect(line.contains("5.369s"))
        // Distinct from a failure, so an abandoned client does not read as a
        // server fault in the log or in anything counting error lines.
        #expect(!line.contains("failed"))
    }

    @Test func completionLogsNumericStagesWithoutPayload() {
        let completion = ServerCompletion(
            content: "private completion text",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(
                promptTokens: 3,
                completionTokens: 2,
                totalTokens: 5,
                cachedTokens: 0),
            requestPreparationSeconds: 0.125,
            prefillSeconds: 1.25,
            decodeSeconds: 2.5)
        let line = ServerLog.completedMessage(
            id: "chatcmpl-abc",
            duration: .seconds(5),
            completion: completion)

        #expect(line.contains("request_prepare=0.125s"))
        #expect(line.contains("prefill=1.250s"))
        #expect(line.contains("decode=2.500s"))
        #expect(!line.contains(completion.content))
    }
}
