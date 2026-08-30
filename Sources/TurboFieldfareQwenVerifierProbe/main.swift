import Darwin
import Foundation
import Metal
import TurboFieldfare

private let usage = """
Usage:
  TurboFieldfareQwenVerifierProbe \
    --model <model.gturbo> \
    --prompt-tokens <id,id,...> \
    --proposed-tokens <id,id,...> \
    [--max-context <tokens>] \
    [--chunk-tokens <32|64|128|256>] \
    [--prefill-mode <batch|scalar>] \
    [--compare-prefill-modes] \
    [--expert-cache-slots <10|...> (default 32)] \
    [--expert-cache-policy <lfu|lru>]
"""

private enum PrefillMode: String, Sendable {
    case batch
    case scalar
}

private enum ArgumentError: Error, CustomStringConvertible {
    case helpRequested
    case invalid(String)

    var description: String {
        switch self {
        case .helpRequested:
            return usage
        case .invalid(let detail):
            return detail
        }
    }
}

private struct Arguments {
    let modelURL: URL
    let promptTokens: [Int32]
    let proposedTokens: [Int32]
    let maxContext: Int
    let chunkTokens: Int
    let prefillMode: PrefillMode
    let comparePrefillModes: Bool
    let expertCacheSlots: Int
    let expertCachePolicy: ExpertCachePolicy

    static func parse(_ raw: [String]) throws -> Arguments {
        var modelPath: String?
        var promptTokens: [Int32]?
        var proposedTokens: [Int32]?
        var maxContext = 4_096
        var chunkTokens = 128
        var prefillMode: PrefillMode = .batch
        var comparePrefillModes = false
        var expertCacheSlots = RuntimeConfiguration.defaultExpertCacheSlots
        var expertCachePolicy: ExpertCachePolicy = .lfu
        var index = 0

        while index < raw.count {
            let option = raw[index]
            if option == "--help" || option == "-h" {
                throw ArgumentError.helpRequested
            }
            if option == "--compare-prefill-modes" {
                comparePrefillModes = true
                index += 1
                continue
            }
            guard index + 1 < raw.count else {
                throw ArgumentError.invalid("missing value for \(option)")
            }
            let value = raw[index + 1]
            switch option {
            case "--model":
                modelPath = value
            case "--prompt-tokens":
                promptTokens = try parseTokenList(value, option: option)
            case "--proposed-tokens":
                proposedTokens = try parseTokenList(value, option: option)
            case "--max-context":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgumentError.invalid("\(option) must be a positive integer")
                }
                maxContext = parsed
            case "--chunk-tokens":
                guard let parsed = Int(value),
                      PrefillRuntimeConfig.allowedChunkTokens.contains(parsed) else {
                    throw ArgumentError.invalid(
                        "\(option) must be one of \(PrefillRuntimeConfig.allowedChunkTokens)")
                }
                chunkTokens = parsed
            case "--prefill-mode":
                guard let parsed = PrefillMode(rawValue: value) else {
                    throw ArgumentError.invalid("\(option) must be batch or scalar")
                }
                prefillMode = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value), parsed >= 10 else {
                    throw ArgumentError.invalid(
                        "\(option) must be an integer greater than or equal to 10")
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                guard let parsed = ExpertCachePolicy(rawValue: value) else {
                    throw ArgumentError.invalid("\(option) must be lfu or lru")
                }
                expertCachePolicy = parsed
            default:
                throw ArgumentError.invalid("unknown argument: \(option)")
            }
            index += 2
        }

        guard let modelPath, !modelPath.isEmpty else {
            throw ArgumentError.invalid("--model is required")
        }
        guard let promptTokens, !promptTokens.isEmpty else {
            throw ArgumentError.invalid("--prompt-tokens must contain at least one token")
        }
        guard let proposedTokens, !proposedTokens.isEmpty else {
            throw ArgumentError.invalid("--proposed-tokens must contain at least one token")
        }
        guard promptTokens.count <= chunkTokens else {
            throw ArgumentError.invalid(
                "prompt token count must not exceed --chunk-tokens")
        }
        guard maxContext >= promptTokens.count + proposedTokens.count + 1 else {
            throw ArgumentError.invalid(
                "--max-context is too small for the prompt and verification block")
        }

        return Arguments(
            modelURL: URL(fileURLWithPath: modelPath, isDirectory: true),
            promptTokens: promptTokens,
            proposedTokens: proposedTokens,
            maxContext: maxContext,
            chunkTokens: chunkTokens,
            prefillMode: prefillMode,
            comparePrefillModes: comparePrefillModes,
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy)
    }

    private static func parseTokenList(_ raw: String,
                                       option: String) throws -> [Int32] {
        let fields = raw.split(separator: ",", omittingEmptySubsequences: false)
        guard !fields.isEmpty else {
            throw ArgumentError.invalid("\(option) must be a comma-separated token list")
        }
        return try fields.map { field in
            guard !field.isEmpty, let token = Int32(field), token >= 0 else {
                throw ArgumentError.invalid(
                    "\(option) contains an invalid token ID: \(field)")
            }
            return token
        }
    }
}

private struct ProbeModeRun {
    let mode: PrefillMode
    let setupSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let decodeGPUActiveNanos: UInt64
    let decodeTimingSamples: [Qwen38DecodeTimingSample]
    let prefillWork: PrefillWorkDiagnostics?
    let emittedTokens: [Int32]
    let statePosition: Int
    let logitTrace: [[Float16]]
}

private struct PrefillParityResult: Codable {
    let comparedModes: [String]
    let logitsMatch: Bool
    let maxAbsoluteLogitDifference: Float
    let emittedTokensMatch: Bool
    let continuationPositionsMatch: Bool
    let batchEmittedTokens: [Int32]
    let scalarEmittedTokens: [Int32]
}

private struct ProbeResult: Codable {
    let prefillMode: String
    let boundaryToken: Int32
    let targetTokens: [Int32]
    let acceptedTokenCount: Int
    let emittedTokens: [Int32]
    let statePosition: Int
    let seedCaptureBytes: Int
    let verificationCaptureBytes: Int
    let setupSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let decodeGPUActiveNanos: UInt64
    let decodeTimingSamples: [Qwen38DecodeTimingSample]
    let prefillWork: PrefillWorkDiagnostics?
    let endToEndSeconds: Double
    let prefillParity: PrefillParityResult?
}

private func elapsedSeconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func greedyToken(from logits: MTLBuffer, vocabularySize: Int) -> Int32 {
    let values = logits.contents().assumingMemoryBound(to: Float16.self)
    var bestIndex = 0
    var bestValue = values[0]
    for index in 1..<vocabularySize where values[index] > bestValue {
        bestIndex = index
        bestValue = values[index]
    }
    return Int32(bestIndex)
}

private func copyLogits(_ logits: MTLBuffer, count: Int) -> [Float16] {
    let values = logits.contents().assumingMemoryBound(to: Float16.self)
    return Array(UnsafeBufferPointer(start: values, count: count))
}

private func oppositeMode(_ mode: PrefillMode) -> PrefillMode {
    mode == .batch ? .scalar : .batch
}

private func runMode(arguments: Arguments,
                     mode: PrefillMode,
                     model: Model,
                     context: MetalContext) async throws -> ProbeModeRun {
    let setupStart = DispatchTime.now().uptimeNanoseconds
    let runner = try Qwen38ForwardRunner(
        model: model,
        context: context,
        maxContext: arguments.maxContext)
    guard let logits = context.device.makeBuffer(
        length: model.config.vocabSize * MemoryLayout<Float16>.stride,
        options: .storageModeShared) else {
        throw NSError(
            domain: "TurboFieldfareQwenVerifierProbe",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "failed to allocate logits buffer"])
    }
    let setupSeconds = elapsedSeconds(since: setupStart)
    let prefillStart = DispatchTime.now().uptimeNanoseconds
    let prefillConfig = PrefillRuntimeConfig.production(
        chunkTokens: arguments.chunkTokens)
    let prefillWork: PrefillWorkDiagnostics?
    switch mode {
    case .batch:
        let prefillResult = try await runner.prefillChunked(
            tokens: arguments.promptTokens[...],
            startPosition: 0,
            outputMode: .logits,
            config: prefillConfig,
            into: logits,
            onProgress: { _ in })
        prefillWork = prefillResult.work
    case .scalar:
        prefillWork = nil
        for token in arguments.promptTokens {
            try await runner.produce(
                token: token,
                position: runner.continuationPosition,
                into: logits)
        }
    }
    let prefillSeconds = elapsedSeconds(since: prefillStart)

    var logitTrace = [copyLogits(logits, count: model.config.vocabSize)]
    let decodeStart = DispatchTime.now().uptimeNanoseconds
    var emittedTokens: [Int32] = []
    var decodeTimingSamples: [Qwen38DecodeTimingSample] = []
    var nextToken = greedyToken(from: logits, vocabularySize: model.config.vocabSize)
    for _ in arguments.proposedTokens {
        emittedTokens.append(nextToken)
        try await runner.produce(
            token: nextToken,
            position: runner.continuationPosition,
            into: logits)
        guard let timing = runner.lastDecodeTiming else {
            throw NSError(
                domain: "TurboFieldfareQwenVerifierProbe",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "decode timing sample was not recorded"])
        }
        decodeTimingSamples.append(timing)
        logitTrace.append(copyLogits(logits, count: model.config.vocabSize))
        nextToken = greedyToken(from: logits, vocabularySize: model.config.vocabSize)
    }
    let decodeSeconds = elapsedSeconds(since: decodeStart)
    let decodeTokensPerSecond = decodeSeconds > 0
        ? Double(emittedTokens.count) / decodeSeconds
        : 0
    let decodeGPUActiveNanos = decodeTimingSamples.reduce(0) {
        $0 + $1.gpuActiveNanos
    }
    return ProbeModeRun(
        mode: mode,
        setupSeconds: setupSeconds,
        prefillSeconds: prefillSeconds,
        decodeSeconds: decodeSeconds,
        decodeTokensPerSecond: decodeTokensPerSecond,
        decodeGPUActiveNanos: decodeGPUActiveNanos,
        decodeTimingSamples: decodeTimingSamples,
        prefillWork: prefillWork,
        emittedTokens: emittedTokens,
        statePosition: runner.continuationPosition,
        logitTrace: logitTrace)
}

private func parityResult(batch: ProbeModeRun,
                          scalar: ProbeModeRun) -> PrefillParityResult {
    var maxAbsoluteDifference: Float = 0
    guard batch.logitTrace.count == scalar.logitTrace.count else {
        return PrefillParityResult(
            comparedModes: [PrefillMode.batch.rawValue, PrefillMode.scalar.rawValue],
            logitsMatch: false,
            maxAbsoluteLogitDifference: .infinity,
            emittedTokensMatch: batch.emittedTokens == scalar.emittedTokens,
            continuationPositionsMatch: batch.statePosition == scalar.statePosition,
            batchEmittedTokens: batch.emittedTokens,
            scalarEmittedTokens: scalar.emittedTokens)
    }
    for (batchRow, scalarRow) in zip(batch.logitTrace, scalar.logitTrace) {
        guard batchRow.count == scalarRow.count else {
            maxAbsoluteDifference = .infinity
            break
        }
        for (batchValue, scalarValue) in zip(batchRow, scalarRow) {
            maxAbsoluteDifference = max(
                maxAbsoluteDifference,
                abs(Float(batchValue) - Float(scalarValue)))
        }
    }
    return PrefillParityResult(
        comparedModes: [PrefillMode.batch.rawValue, PrefillMode.scalar.rawValue],
        logitsMatch: maxAbsoluteDifference <= 0.02,
        maxAbsoluteLogitDifference: maxAbsoluteDifference,
        emittedTokensMatch: batch.emittedTokens == scalar.emittedTokens,
        continuationPositionsMatch: batch.statePosition == scalar.statePosition,
        batchEmittedTokens: batch.emittedTokens,
        scalarEmittedTokens: scalar.emittedTokens)
}

private func run(_ rawArguments: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(rawArguments)
    } catch ArgumentError.helpRequested {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    do {
        let endToEndStart = DispatchTime.now().uptimeNanoseconds
        let setupStart = DispatchTime.now().uptimeNanoseconds
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: arguments.modelURL,
            device: context.device,
            expecting: .qwen38FlashNextText,
            streamingMode: .pread(slotCount: arguments.expertCacheSlots),
            expertCachePolicy: arguments.expertCachePolicy)
        let baseSetupSeconds = elapsedSeconds(since: setupStart)
        let primary = try await runMode(
            arguments: arguments,
            mode: arguments.prefillMode,
            model: model,
            context: context)
        var prefillParity: PrefillParityResult?
        if arguments.comparePrefillModes {
            let secondary = try await runMode(
                arguments: arguments,
                mode: oppositeMode(arguments.prefillMode),
                model: model,
                context: context)
            let batch = primary.mode == .batch ? primary : secondary
            let scalar = primary.mode == .scalar ? primary : secondary
            prefillParity = parityResult(batch: batch, scalar: scalar)
        }
        let endToEndSeconds = elapsedSeconds(since: endToEndStart)
        let result = ProbeResult(
            prefillMode: primary.mode.rawValue,
            boundaryToken: arguments.promptTokens[arguments.promptTokens.count - 1],
            targetTokens: primary.emittedTokens,
            acceptedTokenCount: primary.emittedTokens.count,
            emittedTokens: primary.emittedTokens,
            statePosition: primary.statePosition,
            seedCaptureBytes: 0,
            verificationCaptureBytes: 0,
            setupSeconds: baseSetupSeconds + primary.setupSeconds,
            prefillSeconds: primary.prefillSeconds,
            decodeSeconds: primary.decodeSeconds,
            decodeTokensPerSecond: primary.decodeTokensPerSecond,
            decodeGPUActiveNanos: primary.decodeGPUActiveNanos,
            decodeTimingSamples: primary.decodeTimingSamples,
            prefillWork: primary.prefillWork,
            endToEndSeconds: endToEndSeconds,
            prefillParity: prefillParity)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(result))
        FileHandle.standardOutput.write(Data("\n".utf8))
        return 0
    } catch {
        printError("verification failed: \(error)")
        return 1
    }
}

exit(await run(Array(CommandLine.arguments.dropFirst())))
