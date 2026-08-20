import Foundation

public struct FeasibilityBenchmarkHardware: Codable, Sendable, Equatable {
    public let machine: String
    public let operatingSystem: String
    public let metalDevice: String

    public init(machine: String,
                operatingSystem: String,
                metalDevice: String) {
        self.machine = machine
        self.operatingSystem = operatingSystem
        self.metalDevice = metalDevice
    }
}

public struct FeasibilityBenchmarkMemory: Codable, Sendable, Equatable {
    public let physicalFootprintBytes: UInt64
    public let residentBytes: UInt64
    public let pressure: String

    public init(physicalFootprintBytes: UInt64,
                residentBytes: UInt64,
                pressure: String) {
        self.physicalFootprintBytes = physicalFootprintBytes
        self.residentBytes = residentBytes
        self.pressure = pressure
    }
}

public struct FeasibilityBenchmarkIO: Codable, Sendable, Equatable {
    public let expertBytesRead: UInt64
    public let preadOperations: Int
    public let elapsedMilliseconds: Double

    public init(expertBytesRead: UInt64,
                preadOperations: Int,
                elapsedMilliseconds: Double) {
        self.expertBytesRead = expertBytesRead
        self.preadOperations = preadOperations
        self.elapsedMilliseconds = elapsedMilliseconds
    }
}

public struct FeasibilityBenchmarkTiming: Codable, Sendable, Equatable {
    public let warmupRuns: Int
    public let measuredRuns: Int
    public let bundleLayers: Int
    public let totalLayers: Int
    public let prefillTokens: Int

    public init(warmupRuns: Int,
                measuredRuns: Int,
                bundleLayers: Int,
                totalLayers: Int,
                prefillTokens: Int) {
        self.warmupRuns = warmupRuns
        self.measuredRuns = measuredRuns
        self.bundleLayers = bundleLayers
        self.totalLayers = totalLayers
        self.prefillTokens = prefillTokens
    }
}

public struct FeasibilityBenchmarkCompute: Codable, Sendable, Equatable {
    public let decodeBundleMedianMilliseconds: Double
    public let decodeBundleP95Milliseconds: Double
    public let prefillBundleMedianMilliseconds: Double
    public let prefillBundleP95Milliseconds: Double
    public let outputVocabularySize: Int
    public let outputBytesTouched: UInt64
    public let commandBuffersCompleted: Int

    public init(decodeBundleMedianMilliseconds: Double,
                decodeBundleP95Milliseconds: Double,
                prefillBundleMedianMilliseconds: Double,
                prefillBundleP95Milliseconds: Double,
                outputVocabularySize: Int,
                outputBytesTouched: UInt64,
                commandBuffersCompleted: Int) {
        self.decodeBundleMedianMilliseconds = decodeBundleMedianMilliseconds
        self.decodeBundleP95Milliseconds = decodeBundleP95Milliseconds
        self.prefillBundleMedianMilliseconds = prefillBundleMedianMilliseconds
        self.prefillBundleP95Milliseconds = prefillBundleP95Milliseconds
        self.outputVocabularySize = outputVocabularySize
        self.outputBytesTouched = outputBytesTouched
        self.commandBuffersCompleted = commandBuffersCompleted
    }
}

public struct FeasibilityBenchmarkScenario: Codable, Sendable, Equatable {
    public let expertHitRate: Double
    public let medianMilliseconds: Double
    public let p95Milliseconds: Double
    public let bytesPerToken: UInt64
    public let requiredSSDBandwidthBytesPerSecond: Double

    public init(expertHitRate: Double,
                medianMilliseconds: Double,
                p95Milliseconds: Double,
                bytesPerToken: UInt64,
                requiredSSDBandwidthBytesPerSecond: Double) {
        self.expertHitRate = expertHitRate
        self.medianMilliseconds = medianMilliseconds
        self.p95Milliseconds = p95Milliseconds
        self.bytesPerToken = bytesPerToken
        self.requiredSSDBandwidthBytesPerSecond =
            requiredSSDBandwidthBytesPerSecond
    }
}

public struct FeasibilityBenchmarkBaseline: Codable, Sendable, Equatable {
    public let model: String
    public let medianTokensPerSecond: Double
    public let p95TokensPerSecond: Double

    public init(model: String,
                medianTokensPerSecond: Double,
                p95TokensPerSecond: Double) {
        self.model = model
        self.medianTokensPerSecond = medianTokensPerSecond
        self.p95TokensPerSecond = p95TokensPerSecond
    }
}

public struct FeasibilityBenchmarkProjection: Codable, Sendable, Equatable {
    public let representativeExpertHitRate: Double
    public let projectedDecodeTokensPerSecond: Double
    public let projectedPrefillMilliseconds: Double

    public init(representativeExpertHitRate: Double,
                projectedDecodeTokensPerSecond: Double,
                projectedPrefillMilliseconds: Double) {
        self.representativeExpertHitRate = representativeExpertHitRate
        self.projectedDecodeTokensPerSecond = projectedDecodeTokensPerSecond
        self.projectedPrefillMilliseconds = projectedPrefillMilliseconds
    }
}

public struct FeasibilityBenchmarkReport: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let hardware: FeasibilityBenchmarkHardware
    public let memory: FeasibilityBenchmarkMemory
    public let io: FeasibilityBenchmarkIO
    public let timing: FeasibilityBenchmarkTiming
    public let compute: FeasibilityBenchmarkCompute
    public let scenarios: [FeasibilityBenchmarkScenario]
    public let baseline: FeasibilityBenchmarkBaseline
    public let projection: FeasibilityBenchmarkProjection

    public init(schemaVersion: Int = currentSchemaVersion,
                hardware: FeasibilityBenchmarkHardware,
                memory: FeasibilityBenchmarkMemory,
                io: FeasibilityBenchmarkIO,
                timing: FeasibilityBenchmarkTiming,
                compute: FeasibilityBenchmarkCompute,
                scenarios: [FeasibilityBenchmarkScenario],
                baseline: FeasibilityBenchmarkBaseline,
                projection: FeasibilityBenchmarkProjection) {
        self.schemaVersion = schemaVersion
        self.hardware = hardware
        self.memory = memory
        self.io = io
        self.timing = timing
        self.compute = compute
        self.scenarios = scenarios
        self.baseline = baseline
        self.projection = projection
    }

    public func validate() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "unsupported schema version \(schemaVersion)")
        }
        guard !hardware.machine.isEmpty,
              !hardware.operatingSystem.isEmpty,
              !hardware.metalDevice.isEmpty else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "hardware identity is required")
        }
        guard memory.physicalFootprintBytes > 0,
              memory.residentBytes > 0,
              !memory.pressure.isEmpty else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "memory measurements are required")
        }
        guard io.expertBytesRead > 0,
              io.preadOperations > 0,
              io.elapsedMilliseconds > 0 else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "I/O measurements are required")
        }
        guard timing.warmupRuns > 0,
              timing.measuredRuns > 0,
              timing.bundleLayers > 0,
              timing.totalLayers > 0,
              timing.prefillTokens > 0 else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "timing configuration is required")
        }
                guard compute.decodeBundleMedianMilliseconds > 0,
                            compute.decodeBundleP95Milliseconds >=
                                compute.decodeBundleMedianMilliseconds,
                            compute.prefillBundleMedianMilliseconds > 0,
                            compute.prefillBundleP95Milliseconds >=
                                compute.prefillBundleMedianMilliseconds,
                            compute.outputVocabularySize == 248_320,
                            compute.outputBytesTouched > 0,
                            compute.commandBuffersCompleted > 0 else {
                        throw FeasibilityBenchmarkValidationError.invalid(
                                "compute measurements are required")
                }
        guard baseline.model == "qwen35-9b",
              baseline.medianTokensPerSecond > 0,
              baseline.p95TokensPerSecond > 0 else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "same-run qwen35-9b baseline is required")
        }
        let expectedHitRates: Set<Double> = [0, 0.25, 0.5, 0.75]
        let actualHitRates = Set(scenarios.map(\.expertHitRate))
        guard actualHitRates == expectedHitRates,
              scenarios.count == expectedHitRates.count else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "scenarios must contain 0%, 25%, 50%, and 75% hit rates")
        }
        for scenario in scenarios {
            guard (0...1).contains(scenario.expertHitRate),
                  scenario.medianMilliseconds > 0,
                  scenario.p95Milliseconds >= scenario.medianMilliseconds,
                  scenario.bytesPerToken > 0,
                  scenario.requiredSSDBandwidthBytesPerSecond > 0 else {
                throw FeasibilityBenchmarkValidationError.invalid(
                    "scenario contains an invalid measurement")
            }
        }
        guard expectedHitRates.contains(projection.representativeExpertHitRate),
              projection.projectedDecodeTokensPerSecond > 0,
              projection.projectedPrefillMilliseconds > 0 else {
            throw FeasibilityBenchmarkValidationError.invalid(
                "projection is incomplete")
        }
    }

    public static func project(bundleDecodeMilliseconds: Double,
                               bundlePrefillMilliseconds: Double,
                               representativeExpertHitRate: Double,
                               bundleLayers: Int = 4,
                               totalLayers: Int = 40) ->
        FeasibilityBenchmarkProjection {
        let layerScale = Double(totalLayers) / Double(bundleLayers)
        return FeasibilityBenchmarkProjection(
            representativeExpertHitRate: representativeExpertHitRate,
            projectedDecodeTokensPerSecond:
                1_000 / (bundleDecodeMilliseconds * layerScale),
            projectedPrefillMilliseconds:
                bundlePrefillMilliseconds * layerScale)
    }
}

public enum FeasibilityBenchmarkValidationError: Error, Equatable,
    CustomStringConvertible {
    case invalid(String)

    public var description: String {
        switch self {
        case .invalid(let message): return message
        }
    }
}