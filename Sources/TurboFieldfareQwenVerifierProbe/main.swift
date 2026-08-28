import Darwin
import Foundation
import Metal
import TurboFieldfare

private let usage = """
Usage:
  TurboFieldfareQwenVerifierProbe \
    --model <model.gturbo> \
    --prompt-tokens <id,id,...> \
    --proposed-tokens <id,id,id,id,id,id,id> \
    [--max-context <tokens>] \
    [--chunk-tokens <32|64|128|256>]
"""

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

    static func parse(_ raw: [String]) throws -> Arguments {
        var modelPath: String?
        var promptTokens: [Int32]?
        var proposedTokens: [Int32]?
        var maxContext = 4_096
        var chunkTokens = 128
        var index = 0

        while index < raw.count {
            let option = raw[index]
            if option == "--help" || option == "-h" {
                throw ArgumentError.helpRequested
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
        guard let proposedTokens, proposedTokens.count == 7 else {
            throw ArgumentError.invalid("--proposed-tokens must contain exactly seven tokens")
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
            chunkTokens: chunkTokens)
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

private struct ProbeResult: Codable {
    let boundaryToken: Int32
    let targetTokens: [Int32]
    let acceptedTokenCount: Int
    let emittedTokens: [Int32]
    let statePosition: Int
    let seedCaptureBytes: Int
    let verificationCaptureBytes: Int
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
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: arguments.modelURL,
            device: context.device,
            expecting: .qwen38FlashNextText)
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
        let prefillConfig = PrefillRuntimeConfig.production(
            chunkTokens: arguments.chunkTokens)
        _ = try await runner.prefillChunked(
            tokens: arguments.promptTokens[...],
            startPosition: 0,
            outputMode: .logits,
            config: prefillConfig,
            into: logits,
            onProgress: { _ in })

        var emittedTokens: [Int32] = []
        var nextToken = greedyToken(from: logits, vocabularySize: model.config.vocabSize)
        for _ in arguments.proposedTokens {
            emittedTokens.append(nextToken)
            try await runner.produce(
                token: nextToken,
                position: runner.continuationPosition,
                into: logits)
            nextToken = greedyToken(from: logits, vocabularySize: model.config.vocabSize)
        }
        let result = ProbeResult(
            boundaryToken: arguments.promptTokens[arguments.promptTokens.count - 1],
            targetTokens: emittedTokens,
            acceptedTokenCount: emittedTokens.count,
            emittedTokens: emittedTokens,
            statePosition: runner.continuationPosition,
            seedCaptureBytes: 0,
            verificationCaptureBytes: 0)
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
