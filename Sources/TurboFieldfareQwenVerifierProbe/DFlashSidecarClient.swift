import Foundation
import TurboFieldfare

final class DFlashSidecarClient {
    private struct ResetRequest: Encodable {
        let op = "reset"
    }

    private struct DraftRequest: Encodable {
        let op = "draft"
        let lastBonus: Int32
        let captureLayerIDs: [Int]
        let captureRows: Int
        let hiddenSize: Int
        let hiddenF16: [UInt16]

        enum CodingKeys: String, CodingKey {
            case op
            case lastBonus = "last_bonus"
            case captureLayerIDs = "capture_layer_ids"
            case captureRows = "capture_rows"
            case hiddenSize = "hidden_size"
            case hiddenF16 = "hidden_f16"
        }
    }

    private struct Response: Decodable {
        let ok: Bool
        let proposals: [Int32]?
        let error: String?
    }

    private enum ClientError: LocalizedError {
        case closed
        case responseTooLarge
        case invalidResponse(String)
        case invalidProposalCount(Int)

        var errorDescription: String? {
            switch self {
            case .closed:
                return "DFlash sidecar closed its response pipe"
            case .responseTooLarge:
                return "DFlash sidecar response exceeded the 1 MiB limit"
            case .invalidResponse(let detail):
                return "DFlash sidecar request failed: \(detail)"
            case .invalidProposalCount(let count):
                return "DFlash sidecar returned \(count) proposals; expected 7"
            }
        }
    }

    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(executableURL: URL, checkpointURL: URL, bindingsURL: URL) throws {
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let process = Process()
        process.executableURL = executableURL
        process.arguments = [
            "--checkpoint", checkpointURL.path,
            "--bindings", bindingsURL.path,
        ]
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = FileHandle.standardError
        try process.run()

        self.process = process
        self.input = inputPipe.fileHandleForWriting
        self.output = outputPipe.fileHandleForReading
    }

    deinit {
        try? input.close()
        if process.isRunning {
            process.terminate()
        }
    }

    func reset() throws {
        let response: Response = try exchange(ResetRequest())
        try validate(response)
    }

    func draft(lastBonus: Int32, capture: QwenDFlashHiddenCapture) throws -> [Int32] {
        guard capture.data.count.isMultiple(of: MemoryLayout<UInt16>.stride) else {
            throw ClientError.invalidResponse("hidden capture byte count is not Float16-aligned")
        }
        let bytes = [UInt8](capture.data)
        let hidden = stride(from: 0, to: bytes.count, by: 2).map { offset in
            UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
        }
        let request = DraftRequest(
            lastBonus: lastBonus,
            captureLayerIDs: capture.captureLayerIDs,
            captureRows: capture.rowCount,
            hiddenSize: capture.hiddenSize,
            hiddenF16: hidden
        )
        let response: Response = try exchange(request)
        try validate(response)
        guard let proposals = response.proposals, proposals.count == 7 else {
            throw ClientError.invalidProposalCount(response.proposals?.count ?? 0)
        }
        return proposals
    }

    private func exchange<Request: Encodable>(_ request: Request) throws -> Response {
        guard process.isRunning else { throw ClientError.closed }
        var data = try encoder.encode(request)
        data.append(0x0A)
        try input.write(contentsOf: data)
        let responseLine = try readResponseLine()
        return try decoder.decode(Response.self, from: responseLine)
    }

    private func readResponseLine() throws -> Data {
        var data = Data()
        while true {
            guard let byte = try output.read(upToCount: 1), !byte.isEmpty else {
                throw ClientError.closed
            }
            if byte[byte.startIndex] == 0x0A {
                return data
            }
            data.append(byte)
            guard data.count <= 1_048_576 else {
                throw ClientError.responseTooLarge
            }
        }
    }

    private func validate(_ response: Response) throws {
        guard response.ok else {
            throw ClientError.invalidResponse(response.error ?? "unspecified sidecar error")
        }
    }
}
