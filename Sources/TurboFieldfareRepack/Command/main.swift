import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
    TurboFieldfareRepack --source <hugging-face-snapshot> --output <model.gturbo> [--overwrite]
    TurboFieldfareRepack --output <model.gturbo> [--model gemma4|qwen36|qwen38|qwen38-mtplx] [--overwrite] [--resume] [--remote-concurrency <1-8>] [--resident-concurrency <1-8>] [--range-chunk-mib <1-256>]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --verify-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --activate-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --remove-vision-install
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --discard-partial
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --help

The installer streams the selected supported checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume or removed with --discard-partial.

The optional image companion pack installs beside an existing text model and
is bound to it. Without the pack the text runtime is unchanged; image input is
simply unavailable.
"""

private struct Arguments {
    var output: String?
    var source: String?
    var model = "gemma4"
    var remoteConcurrency = 1
    var residentConcurrency = 1
    var rangeChunkBytes = RemoteChunkPolicy.defaultBytes
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    var visionOutput: String?
    var textModel: String?
    var verifyVisionInstall = false
    var activateVisionInstall = false
    var removeVisionInstall = false

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--source":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.source = values[index + 1]
                index += 2
            case "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.model = values[index + 1]
                index += 2
            case "--remote-concurrency":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let concurrency = Int(values[index + 1]) else {
                    throw ParseError.invalidMode("invalid remote concurrency")
                }
                parsed.remoteConcurrency = concurrency
                index += 2
            case "--range-chunk-mib":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let mebibytes = Int(values[index + 1]),
                      (1...RemoteChunkPolicy.maxBytes / (1024 * 1024))
                          .contains(mebibytes) else {
                    throw ParseError.invalidMode("invalid range chunk size")
                }
                parsed.rangeChunkBytes = mebibytes * 1024 * 1024
                index += 2
            case "--resident-concurrency":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let concurrency = Int(values[index + 1]),
                      (1...8).contains(concurrency) else {
                    throw ParseError.invalidMode("invalid resident concurrency")
                }
                parsed.residentConcurrency = concurrency
                index += 2
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--verify-vision-install":
                parsed.verifyVisionInstall = true
                index += 1
            case "--activate-vision-install":
                parsed.activateVisionInstall = true
                index += 1
            case "--remove-vision-install":
                parsed.removeVisionInstall = true
                index += 1
            case "--vision-output":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.visionOutput = values[index + 1]
                index += 2
            case "--text-model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.textModel = values[index + 1]
                index += 2
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        let visionModes = [parsed.verifyVisionInstall,
                           parsed.activateVisionInstall,
                           parsed.removeVisionInstall].filter { $0 }.count
        guard visionModes <= 1 else {
            throw ParseError.invalidMode("vision install modes are mutually exclusive")
        }
        if visionModes == 1 || parsed.visionOutput != nil {
            guard parsed.visionOutput != nil else {
                throw ParseError.missingRequired("--vision-output")
            }
            guard parsed.output == nil, parsed.inputGTurbo == nil,
                  parsed.source == nil, !parsed.verifyInstall else {
                throw ParseError.invalidMode(
                    "vision install operations do not accept text install arguments")
            }
            // Discard runs first below, so accepting it alongside another mode
            // would silently perform the discard and exit 0 without ever doing
            // what was asked.
            guard !(parsed.discardPartial && visionModes == 1) else {
                throw ParseError.invalidMode(
                    "--discard-partial is mutually exclusive with the other vision "
                        + "install operations")
            }
            if parsed.removeVisionInstall || parsed.discardPartial {
                guard parsed.textModel == nil, !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output")
                }
            } else if parsed.verifyVisionInstall || parsed.activateVisionInstall {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
                // Neither reads a download, so a transfer flag here is a
                // request this mode cannot honour rather than a no-op.
                guard !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output and "
                            + "--text-model")
                }
            } else {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
            }
            return parsed
        }
        guard parsed.textModel == nil else {
            throw ParseError.invalidMode("--text-model requires --vision-output")
        }
        if parsed.source != nil {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil,
                  !parsed.verifyInstall,
                  !parsed.resume,
                  !parsed.discardPartial else {
                throw ParseError.invalidMode(
                    "--source only accepts --output and --overwrite")
            }
            return parsed
        }
        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func runVisionInstall(_ arguments: Arguments) async -> Int32? {
    guard let visionOutput = arguments.visionOutput else { return nil }

    if arguments.discardPartial {
        do {
            try RemoteVisionPackInstaller.discardPartial(outputDirectory: visionOutput)
            print("Discarded saved image-pack download for \(visionOutput)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.removeVisionInstall {
        do {
            try RemoteVisionPackInstaller.removeInstalled(outputDirectory: visionOutput)
            print("Removed image pack \(visionOutput)")
            return 0
        } catch {
            printError("remove-vision-install failed: \(error)")
            return 1
        }
    }

    guard let textModel = arguments.textModel else { return 2 }

    if arguments.verifyVisionInstall {
        do {
            let verification = try VisionPackVerifier.verify(
                directory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                installedDirectory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                textModelDirectory: URL(fileURLWithPath: textModel, isDirectory: true),
                verifyWeights: true)
            print("Verified image pack \(visionOutput)")
            print("Bound to text model \(textModel)")
            print("Text manifest sha256 \(verification.compatibleTextManifestSha256)")
            return 0
        } catch {
            printError("verify-vision-install failed: \(error)")
            return 1
        }
    }

    if arguments.activateVisionInstall {
        do {
            try RemoteVisionPackInstaller.activatePrepared(
                outputDirectory: visionOutput,
                textModelDirectory: textModel,
                repoID: SupportedModelSource.repoID,
                requestedRevision: SupportedModelSource.revision)
            print("Activated image pack \(visionOutput)")
            return 0
        } catch {
            printError("activate-vision-install failed: \(error)")
            return 1
        }
    }

    let options = RemoteVisionPackInstallOptions(
        repoID: SupportedModelSource.repoID,
        revision: SupportedModelSource.revision,
        textModelDirectory: textModel,
        outputDirectory: visionOutput,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        overwrite: arguments.overwrite,
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        try await RemoteVisionPackInstaller(options: options).run(
            progress: { progress($0) })
        print("Installed image pack \(visionOutput)")
        print("Text model: \(textModel)")
        return 0
    } catch {
        printError("vision install failed: \(error)")
        return 1
    }
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if let code = await runVisionInstall(arguments) {
        return code
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    if let source = arguments.source {
        let options = LocalSnapshotRepackOptions(
            snapshotDirectory: source,
            outputDirectory: output,
            overwrite: arguments.overwrite,
            residentConcurrency: arguments.residentConcurrency)
        do {
            let progress = InstallProgressReporter()
            let result = try await LocalSnapshotRepacker(options: options).run(
                progress: { progress($0) })
            print("Installed local snapshot")
            print("Source index sha256: \(result.sourceIndexSHA256)")
            print("Model: \(result.outputDirectory)")
            return 0
        } catch {
            printError("local install failed: \(error)")
            return 1
        }
    }
    guard let profile = SupportedModelSource.profile(forName: arguments.model) else {
        printError("error: unknown model profile \(arguments.model) (expected gemma4, qwen36, qwen38, or qwen38-mtplx)")
        return 2
    }
    let options = profile.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume,
        remoteConcurrency: arguments.remoteConcurrency,
        residentConcurrency: arguments.residentConcurrency,
        rangeChunkBytes: arguments.rangeChunkBytes)
    do {
        let progress = InstallProgressReporter()
        let result = try await RemoteStreamingRepacker(options: options).run(
            progress: { progress($0) })
        print("Installed \(profile.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
