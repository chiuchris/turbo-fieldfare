import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareFeasibility --snapshot <directory> [--output <file>]
    TurboFieldfareFeasibility --repo <owner/name> --revision <commit> --work-dir <directory> [--output <file>]
    TurboFieldfareFeasibility --repo <owner/name> --revision <commit> --work-dir <directory> --materialize [--sample-experts <count>] [--max-range-bytes <bytes>] [--output <file>]
"""

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ arguments: [String]) async -> Int32 {
    var snapshot: String?
    var repo: String?
    var revision: String?
    var workDirectory: String?
    var output: String?
    var materialize = false
    var sampleExpertCount = FeasibilityProbe.defaultSampleExpertCount
    var maxRangeBytes = FeasibilityProbe.defaultMaxRangeBytes
    var index = 1
    while index < arguments.count {
        switch arguments[index] {
        case "--help":
            print(usage)
            return 0
        case "--materialize":
            materialize = true
            index += 1
        case "--snapshot", "--repo", "--revision", "--work-dir", "--output":
            guard index + 1 < arguments.count else {
                printError("missing value for \(arguments[index])\n\n\(usage)")
                return 2
            }
            switch arguments[index] {
            case "--snapshot":
                snapshot = arguments[index + 1]
            case "--repo":
                repo = arguments[index + 1]
            case "--revision":
                revision = arguments[index + 1]
            case "--work-dir":
                workDirectory = arguments[index + 1]
            default:
                output = arguments[index + 1]
            }
            index += 2
        case "--sample-experts", "--max-range-bytes":
            guard index + 1 < arguments.count,
                  let value = Int(arguments[index + 1]), value > 0 else {
                printError("invalid numeric value for \(arguments[index])\n\n\(usage)")
                return 2
            }
            if arguments[index] == "--sample-experts" {
                sampleExpertCount = value
            } else {
                maxRangeBytes = value
            }
            index += 2
        default:
            printError("unknown argument: \(arguments[index])\n\n\(usage)")
            return 2
        }
    }

    guard (snapshot != nil) != (repo != nil) else {
        printError("choose exactly one of --snapshot or --repo\n\n\(usage)")
        return 2
    }
    if repo != nil && (revision == nil || workDirectory == nil) {
        printError("remote scans require --revision and --work-dir\n\n\(usage)")
        return 2
    }
    if materialize && repo == nil {
        printError("--materialize requires --repo\n\n\(usage)")
        return 2
    }
    do {
        let data: Data
        if let snapshot {
            let inventory = try FeasibilityInventory.scan(snapshotDirectory: snapshot)
            data = try JSONEncoder.prettySorted.encode(inventory)
        } else if materialize {
            let report = try await FeasibilityProbe.runRemote(
                repoID: repo!,
                revision: revision!,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                workingDirectory: workDirectory!,
                sampleExpertCount: sampleExpertCount,
                maxRangeBytes: maxRangeBytes)
            data = try JSONEncoder.prettySorted.encode(report)
        } else {
            let inventory = try await FeasibilityInventory.scanRemote(
                repoID: repo!,
                revision: revision!,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                workingDirectory: workDirectory!)
            data = try JSONEncoder.prettySorted.encode(inventory)
        }
        if let output {
            try data.write(to: URL(fileURLWithPath: output), options: .atomic)
        } else {
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
        return 0
    } catch {
        printError("feasibility inventory failed: \(error)")
        return 1
    }
}

private extension JSONEncoder {
    static var prettySorted: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

exit(await run(CommandLine.arguments))