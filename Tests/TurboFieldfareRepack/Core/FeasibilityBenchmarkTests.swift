import Foundation
import Metal
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct FeasibilityBenchmarkTests {
    @Test func reportRequiresAllPhase0Evidence() throws {
        let report = makeReport()

        #expect(throws: Never.self) {
            try report.validate()
        }

        let encoded = try JSONEncoder().encode(report)
        var object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "hardware")
        let missingHardware = try JSONSerialization.data(withJSONObject: object)

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(
                FeasibilityBenchmarkReport.self,
                from: missingHardware)
        }
    }

    @Test func validationRejectsIncompleteHitRateTrace() throws {
        var report = makeReport()
        report = FeasibilityBenchmarkReport(
            hardware: report.hardware,
            memory: report.memory,
            io: report.io,
            timing: report.timing,
            compute: report.compute,
            scenarios: Array(report.scenarios.dropLast()),
            baseline: report.baseline,
            projection: report.projection)

        #expect(throws: FeasibilityBenchmarkValidationError.self) {
            try report.validate()
        }
    }

    @Test func projectionScalesFourLayerBundleToFortyLayers() {
        let projection = FeasibilityBenchmarkReport.project(
            bundleDecodeMilliseconds: 20,
            bundlePrefillMilliseconds: 100,
            representativeExpertHitRate: 0.5)

        #expect(projection.projectedDecodeTokensPerSecond == 5)
        #expect(projection.projectedPrefillMilliseconds == 1_000)
    }

    @Test func runnerTouchesPayloadAndCompletesMetalWork() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("f0b-runner-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root,
                                                withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let payload = root.appendingPathComponent("payload.bin")
        try Data(repeating: 0x5A, count: 64 * 1024).write(to: payload)

        let run = try FeasibilityBenchmarkRunner.run(
            payloadPath: payload.path,
            warmupRuns: 1,
            measuredRuns: 1)

        #expect(run.scenarios.map(\.expertHitRate) == [0, 0.25, 0.5, 0.75])
        #expect(run.io.expertBytesRead > 0)
        #expect(run.io.preadOperations > 0)
        #expect(run.compute.outputVocabularySize == 248_320)
        #expect(run.compute.outputBytesTouched > 0)
        #expect(run.compute.commandBuffersCompleted == 2)
    }

    private func makeReport() -> FeasibilityBenchmarkReport {
        let scenarios = [0.0, 0.25, 0.5, 0.75].map {
            FeasibilityBenchmarkScenario(
                expertHitRate: $0,
                medianMilliseconds: 20,
                p95Milliseconds: 25,
                bytesPerToken: 4_096,
                requiredSSDBandwidthBytesPerSecond: 204_800)
        }
        return FeasibilityBenchmarkReport(
            hardware: FeasibilityBenchmarkHardware(
                machine: "test-machine",
                operatingSystem: "macOS",
                metalDevice: "test-metal"),
            memory: FeasibilityBenchmarkMemory(
                physicalFootprintBytes: 1,
                residentBytes: 1,
                pressure: "normal"),
            io: FeasibilityBenchmarkIO(
                expertBytesRead: 4_096,
                preadOperations: 1,
                elapsedMilliseconds: 1),
            timing: FeasibilityBenchmarkTiming(
                warmupRuns: 1,
                measuredRuns: 5,
                bundleLayers: 4,
                totalLayers: 40,
                prefillTokens: 512),
            compute: FeasibilityBenchmarkCompute(
                decodeBundleMedianMilliseconds: 20,
                decodeBundleP95Milliseconds: 25,
                prefillBundleMedianMilliseconds: 100,
                prefillBundleP95Milliseconds: 125,
                outputVocabularySize: 248_320,
                outputBytesTouched: 1,
                commandBuffersCompleted: 2),
            scenarios: scenarios,
            baseline: FeasibilityBenchmarkBaseline(
                model: "qwen35-9b",
                medianTokensPerSecond: 25,
                p95TokensPerSecond: 24),
            projection: FeasibilityBenchmarkProjection(
                representativeExpertHitRate: 0.5,
                projectedDecodeTokensPerSecond: 5,
                projectedPrefillMilliseconds: 1_000))
    }
}