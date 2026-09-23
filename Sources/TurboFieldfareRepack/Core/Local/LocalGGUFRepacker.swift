import Foundation

public struct LocalGGUFRepackOptions: Sendable {
    public let sourceGGUF: String
    public let outputDirectory: String
    public let overwrite: Bool
    public let rangeChunkBytes: Int
    public let residentConcurrency: Int

    public init(sourceGGUF: String,
                outputDirectory: String,
                overwrite: Bool = false,
                rangeChunkBytes: Int = 1 * 1024 * 1024,
                residentConcurrency: Int = 1) {
        self.sourceGGUF = sourceGGUF
        self.outputDirectory = outputDirectory
        self.overwrite = overwrite
        self.rangeChunkBytes = rangeChunkBytes
        self.residentConcurrency = residentConcurrency
    }
}

public final class LocalGGUFRepacker {
    private let options: LocalGGUFRepackOptions

    public init(options: LocalGGUFRepackOptions) {
        self.options = options
    }

    public func run(
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> LocalSnapshotRepackResult {
        let sourceURL = URL(fileURLWithPath: options.sourceGGUF)
        let document = try GGUFDocument.load(from: sourceURL)
        let temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory(),
                                     isDirectory: true)
            .appendingPathComponent("turbofieldfare-gguf-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory,
                                                withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        try GGUFSnapshotWriter.write(document: document,
                                     sourceURL: sourceURL,
                                     directory: temporaryDirectory)
        let snapshotOptions = LocalSnapshotRepackOptions(
            snapshotDirectory: temporaryDirectory.path,
            outputDirectory: options.outputDirectory,
            overwrite: options.overwrite,
            rangeChunkBytes: options.rangeChunkBytes,
            residentConcurrency: options.residentConcurrency,
            minFreeReserveBytes: 0)
        return try await LocalSnapshotRepacker(options: snapshotOptions).run(
            progress: progress)
    }
}

private enum LocalGGUFRepackError: Error, CustomStringConvertible {
    case unsupportedMTP(String)
    case unmappedTensor(String)
    case duplicateTensor(String)
    case invalidShape(String)
    case unsupportedArchitecture(String)
    case missingTensor(String)
    case invalidMetadata(String)

    var description: String {
        switch self {
        case .unsupportedMTP(let name):
            return "GGUF contains native MTP tensor \(name); ordinary conversion refuses MTP artifacts"
        case .unmappedTensor(let name):
            return "GGUF tensor has no Qwen3.6 runtime mapping: \(name)"
        case .duplicateTensor(let name):
            return "GGUF maps multiple tensors to \(name)"
        case .invalidShape(let detail):
            return "invalid GGUF conversion shape: \(detail)"
        case .unsupportedArchitecture(let architecture):
            return "GGUF conversion supports qwen35moe, got \(architecture)"
        case .missingTensor(let name):
            return "GGUF conversion requires tensor \(name)"
        case .invalidMetadata(let detail):
            return "invalid GGUF conversion metadata: \(detail)"
        }
    }
}

private enum GGUFSnapshotWriter {
    private enum PayloadKind {
        case weight
        case scales
        case biases
    }

    private struct GeneratedTensor {
        let name: String
        let dtype: String
        let shape: [UInt64]
        let source: GGUFTensor
        let kind: PayloadKind
        let byteCount: UInt64
    }

    private struct GeneratedGroup {
        let canonicalName: String
        let source: GGUFTensor
        let quantized: Bool
        let tensors: [GeneratedTensor]
    }

    static func write(document: GGUFDocument,
                      sourceURL: URL,
                      directory: URL) throws {
        guard document.architecture == "qwen35moe" else {
            throw LocalGGUFRepackError.unsupportedArchitecture(document.architecture)
        }
        let groups = try makeGroups(document: document)
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        try writeSafetensors(document: document, groups: groups,
                             sourceURL: sourceURL, url: weightsURL)
        try writeIndex(groups: groups, directory: directory)
        try writeConfig(groups: groups, directory: directory)
    }

    private static func makeGroups(document: GGUFDocument) throws -> [GeneratedGroup] {
        var names = Set<String>()
        return try document.tensors.map { source in
            if source.name.hasPrefix("mtp.") {
                throw LocalGGUFRepackError.unsupportedMTP(source.name)
            }
            guard let canonical = GGUFNameMapper.canonical(source.name) else {
                throw LocalGGUFRepackError.unmappedTensor(source.name)
            }
            guard names.insert(canonical).inserted else {
                throw LocalGGUFRepackError.duplicateTensor(canonical)
            }
            let shape = Array(source.shape.reversed())
            guard !shape.isEmpty else {
                throw LocalGGUFRepackError.invalidShape("empty shape for \(source.name)")
            }
            if isQuantized(source.type) {
                guard let inputWidth = source.shape.first,
                      inputWidth >= 32, inputWidth % 32 == 0 else {
                    throw LocalGGUFRepackError.invalidShape(
                        "quantized width must be divisible by 32 for \(source.name)")
                }
                let packedShape = Array(shape.dropLast()) + [inputWidth / 8]
                let companionShape = Array(shape.dropLast()) + [inputWidth / 32]
                let elements = try product(shape, name: source.name)
                let weightBytes = try checkedMultiply(elements / 8, 4,
                                                      name: source.name)
                let companionBytes = try checkedMultiply(elements / 32, 2,
                                                         name: source.name)
                let tensors = [
                    GeneratedTensor(name: canonical, dtype: "U32", shape: packedShape,
                                    source: source, kind: .weight,
                                    byteCount: weightBytes),
                    GeneratedTensor(name: canonical + ".scales", dtype: "BF16",
                                    shape: companionShape, source: source, kind: .scales,
                                    byteCount: companionBytes),
                    GeneratedTensor(name: canonical + ".biases", dtype: "BF16",
                                    shape: companionShape, source: source, kind: .biases,
                                    byteCount: companionBytes),
                ]
                return GeneratedGroup(canonicalName: canonical, source: source,
                                      quantized: true, tensors: tensors)
            }
            let elements = try product(shape, name: source.name)
            return GeneratedGroup(
                canonicalName: canonical,
                source: source,
                quantized: false,
                tensors: [GeneratedTensor(name: canonical, dtype: "BF16", shape: shape,
                                          source: source, kind: .weight,
                                          byteCount: try checkedMultiply(elements, 2,
                                                                         name: source.name))])
        }
    }

    private static func writeSafetensors(document: GGUFDocument,
                                         groups: [GeneratedGroup],
                                         sourceURL: URL,
                                         url: URL) throws {
        var header: [String: Any] = ["__metadata__": ["format": "pt"]]
        var offset: UInt64 = 0
        for group in groups {
            for tensor in group.tensors {
                guard let begin = Int(exactly: offset),
                      let end = Int(exactly: offset + tensor.byteCount) else {
                    throw LocalGGUFRepackError.invalidShape(
                        "safetensors offset exceeds Int for \(tensor.name)")
                }
                header[tensor.name] = [
                    "dtype": tensor.dtype,
                    "shape": tensor.shape.map { Int($0) },
                    "data_offsets": [begin, end],
                ]
                offset += tensor.byteCount
            }
        }
        var headerData = try JSONSerialization.data(withJSONObject: header,
                                                     options: [.sortedKeys])
        while (8 + headerData.count) % 8 != 0 {
            headerData.append(0x20)
        }
        let created = FileManager.default.createFile(atPath: url.path, contents: nil)
        guard created else { throw CocoaError(.fileNoSuchFile) }
        let writer = try FileHandle(forWritingTo: url)
        defer { try? writer.close() }
        try writer.write(contentsOf: littleEndian(UInt64(headerData.count)))
        try writer.write(contentsOf: headerData)

        for group in groups {
            let values = try decodedValues(group.source, document: document, from: sourceURL)
            let payloads: [PayloadKind: Data]
            if group.quantized {
                let result = try quantizedPayloads(values: values,
                                                   sourceShape: group.source.shape,
                                                   name: group.source.name)
                payloads = [.weight: result.weight, .scales: result.scales,
                            .biases: result.biases]
            } else {
                payloads = [.weight: bf16Payload(values)]
            }
            for tensor in group.tensors {
                guard let payload = payloads[tensor.kind],
                      UInt64(payload.count) == tensor.byteCount else {
                    throw LocalGGUFRepackError.invalidShape(
                        "generated payload size mismatch for \(tensor.name)")
                }
                try writer.write(contentsOf: payload)
            }
        }
    }

    private static func writeIndex(groups: [GeneratedGroup], directory: URL) throws {
        var weightMap: [String: String] = [:]
        for group in groups {
            for tensor in group.tensors {
                weightMap[tensor.name] = "model.safetensors"
            }
        }
        let data = try JSONSerialization.data(
            withJSONObject: ["weight_map": weightMap], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("model.safetensors.index.json"),
                       options: [.atomic])
    }

    private static func writeConfig(groups: [GeneratedGroup], directory: URL) throws {
        guard let embedding = groups.first(where: {
            $0.source.name == "token_embd.weight"
        }) else {
            throw LocalGGUFRepackError.missingTensor("token_embd.weight")
        }
        let hiddenSize = try intValue(embedding.source.shape.first,
                                      name: "hidden size")
        let vocabSize = try intValue(embedding.source.shape.last,
                                     name: "vocabulary size")
        let layerGroups = groups.compactMap { group -> (Int, GGUFTensor)? in
            let parts = group.source.name.split(separator: ".")
            guard parts.count > 1, parts[0] == "blk", let index = Int(parts[1]) else {
                return nil
            }
            return (index, group.source)
        }
        let layerCount = (layerGroups.map(\.0).max() ?? 39) + 1
        let expertTensor = groups.first(where: {
            $0.source.name == "blk.0.ffn_gate_exps.weight"
        })?.source
        let expertCount = try intValue(expertTensor?.shape.last ?? 256,
                                       name: "expert count")
        let moeIntermediate = try intValue(expertTensor?.shape.dropFirst().first ?? 512,
                                           name: "expert intermediate size")
        let sharedTensor = groups.first(where: {
            $0.source.name == "blk.0.ffn_gate_shexp.weight"
        })?.source
        let sharedIntermediate = try intValue(sharedTensor?.shape.dropFirst().first
                                              ?? UInt64(moeIntermediate),
                                              name: "shared intermediate size")
        let headDim = 128
        let textConfig: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "hidden_size": hiddenSize,
            "intermediate_size": sharedIntermediate,
            "shared_expert_intermediate_size": sharedIntermediate,
            "moe_intermediate_size": moeIntermediate,
            "num_attention_heads": hiddenSize / headDim,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": headDim,
            "global_head_dim": headDim,
            "vocab_size": vocabSize,
            "sliding_window": 1,
            "num_hidden_layers": layerCount,
            "num_experts": expertCount,
            "num_experts_per_tok": 8,
            "top_k_experts": 8,
            "tie_word_embeddings": false,
            "attention_k_eq_v": false,
            "hidden_act": "silu",
            "layer_types": Array(repeating: "full_attention", count: layerCount),
            "rope_parameters": [
                "full_attention": ["rope_theta": 1_000_000.0,
                                   "partial_rotary_factor": 1.0],
                "sliding_attention": ["rope_theta": 1_000_000.0],
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        ]
        let config: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": textConfig,
            "quantization": ["bits": 4, "group_size": 32, "mode": "affine"],
        ]
        let data = try JSONSerialization.data(withJSONObject: config,
                                               options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("config.json"),
                       options: [.atomic])
    }

    private static func decodedValues(_ tensor: GGUFTensor,
                                      document: GGUFDocument,
                                      from url: URL) throws -> [Float] {
        let elements = try product(tensor.shape, name: tensor.name)
        let payload = try document.readPayload(for: tensor, from: url,
                                               maxBytes: UInt64(Int.max))
        return try GGUFDecoder.decode(payload, type: tensor.type, elementCount: elements)
    }

    private static func quantizedPayloads(values: [Float],
                                          sourceShape: [UInt64],
                                          name: String) throws
        -> (weight: Data, scales: Data, biases: Data) {
        guard let inputWidth = sourceShape.first,
              inputWidth >= 32, inputWidth % 32 == 0 else {
            throw LocalGGUFRepackError.invalidShape("quantized width for \(name)")
        }
        let width = try intValue(inputWidth, name: name)
        guard values.count % width == 0 else {
            throw LocalGGUFRepackError.invalidShape("row width for \(name)")
        }
        var weight = Data(repeating: 0, count: values.count / 2)
        var scales = Data()
        var biases = Data()
        scales.reserveCapacity(values.count / 16)
        biases.reserveCapacity(values.count / 16)
        for rowStart in stride(from: 0, to: values.count, by: width) {
            for groupStart in stride(from: 0, to: width, by: 32) {
                let group = values[(rowStart + groupStart)..<(rowStart + groupStart + 32)]
                guard group.allSatisfy(\.isFinite) else {
                    throw LocalGGUFRepackError.invalidShape("non-finite value in \(name)")
                }
                let minimum = group.min() ?? 0
                let maximum = group.max() ?? 0
                let scale = maximum == minimum ? Float(1) : (maximum - minimum) / 15
                appendBF16(scale, to: &scales)
                appendBF16(minimum, to: &biases)
                for index in 0..<32 {
                    let value = values[rowStart + groupStart + index]
                    let quantized = maximum == minimum
                        ? 0
                        : max(0, min(15, Int(((value - minimum) / scale).rounded())))
                    let outputIndex = rowStart + groupStart + index
                    let byteIndex = outputIndex / 2
                    if outputIndex.isMultiple(of: 2) {
                        weight[byteIndex] = UInt8(quantized)
                    } else {
                        weight[byteIndex] |= UInt8(quantized << 4)
                    }
                }
            }
        }
        var packed = Data()
        packed.reserveCapacity(weight.count / 4)
        for index in stride(from: 0, to: weight.count, by: 4) {
            let word = UInt32(weight[index])
                | UInt32(weight[index + 1]) << 8
                | UInt32(weight[index + 2]) << 16
                | UInt32(weight[index + 3]) << 24
            packed.append(contentsOf: littleEndian(word))
        }
        return (packed, scales, biases)
    }

    private static func bf16Payload(_ values: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(values.count * 2)
        for value in values {
            appendBF16(value, to: &data)
        }
        return data
    }

    private static func isQuantized(_ type: GGUFTensorType) -> Bool {
        switch type {
        case .f32, .f16, .bf16: return false
        default: return true
        }
    }

    private static func product(_ shape: [UInt64], name: String) throws -> Int {
        var result = 1
        for dimension in shape {
            guard let value = Int(exactly: dimension) else {
                throw LocalGGUFRepackError.invalidShape("element count for \(name)")
            }
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw LocalGGUFRepackError.invalidShape("element count for \(name)")
            }
            result = next
        }
        return result
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int, name: String) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow, value >= 0, let result = UInt64(exactly: value) else {
            throw LocalGGUFRepackError.invalidShape("byte count for \(name)")
        }
        return result
    }

    private static func intValue(_ value: UInt64?, name: String) throws -> Int {
        guard let value, let result = Int(exactly: value), result > 0 else {
            throw LocalGGUFRepackError.invalidMetadata(name)
        }
        return result
    }

    private static func littleEndian<T: FixedWidthInteger>(_ value: T) -> Data {
        var little = value.littleEndian
        return Data(bytes: &little, count: MemoryLayout<T>.size)
    }

    private static func appendBF16(_ value: Float, to data: inout Data) {
        let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        data.append(contentsOf: littleEndian(bits))
    }
}
