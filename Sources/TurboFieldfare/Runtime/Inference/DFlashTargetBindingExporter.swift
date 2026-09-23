import Foundation
import Metal

public enum DFlashTargetBindingExporter {
    public static func export(model: Model, to destination: URL) throws {
        guard model.config.modelFamily == .qwen36MoeText else {
            throw ExportError.unsupportedModelFamily
        }
        guard !model.config.tieWordEmbeddings else {
            throw ExportError.tiedEmbeddings
        }

        try export(
            embedding: model.embedding,
            lmHead: model.lmHead,
            vocabSize: model.config.vocabSize,
            hiddenSize: model.config.hiddenSize,
            to: destination
        )
    }

    static func export(
        embedding: TensorView,
        lmHead: TensorView,
        vocabSize: Int,
        hiddenSize: Int,
        to destination: URL
    ) throws {
        let embeddingPayload = try payload(
            embedding,
            name: "target.embed_tokens",
            vocabSize: vocabSize,
            hiddenSize: hiddenSize
        )
        let lmHeadPayload = try payload(
            lmHead,
            name: "target.lm_head",
            vocabSize: vocabSize,
            hiddenSize: hiddenSize
        )
        let tensors = embeddingPayload.tensors + lmHeadPayload.tensors
        let metadata = embeddingPayload.metadata.merging(lmHeadPayload.metadata) { _, new in new }
        let header = try makeHeader(tensors: tensors, metadata: metadata)

        let fileManager = FileManager.default
        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).tmp")
        guard fileManager.createFile(atPath: temporary.path, contents: nil) else {
            throw ExportError.couldNotCreateOutput
        }

        do {
            let handle = try FileHandle(forWritingTo: temporary)
            defer { try? handle.close() }
            try handle.write(contentsOf: header)
            for tensor in tensors {
                let start = Int(tensor.offset)
                let length = Int(tensor.length)
                let pointer = tensor.buffer.contents().advanced(by: Int(tensor.bufferOffset))
                try handle.write(contentsOf: Data(bytesNoCopy: pointer, count: length, deallocator: .none))
                guard start + length <= tensors.reduce(0, { $0 + Int($1.length) }) else {
                    throw ExportError.invalidTensorRange(tensor.name)
                }
            }
            try handle.synchronize()
            if fileManager.fileExists(atPath: destination.path) {
                _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
            } else {
                try fileManager.moveItem(at: temporary, to: destination)
            }
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw error
        }
    }

    private struct TensorSlice {
        let name: String
        let dtype: String
        let shape: [Int]
        let buffer: MTLBuffer
        let bufferOffset: UInt64
        let length: UInt64
        var offset: UInt64 = 0
    }

    private struct TensorPayload {
        let tensors: [TensorSlice]
        let metadata: [String: String]
    }

    private enum ExportError: Error {
        case unsupportedModelFamily
        case tiedEmbeddings
        case invalidShape(String)
        case invalidQuantization(String)
        case invalidTensorRange(String)
        case couldNotCreateOutput
    }

    private static func payload(
        _ view: TensorView,
        name: String,
        vocabSize: Int,
        hiddenSize: Int
    ) throws -> TensorPayload {
        guard Int(view.shape.0) == vocabSize,
              Int(view.shape.1) == hiddenSize,
              view.shape.2 == 1,
              view.shape.3 == 1 else {
            throw ExportError.invalidShape(name)
        }
        guard let quantization = view.quantization,
              quantization.bits == 4,
              quantization.groupSize == 64,
              view.length > 0,
              view.scaleLength > 0,
              view.biasLength > 0,
              view.scaleLength.isMultiple(of: 2),
              view.biasLength.isMultiple(of: 2) else {
            throw ExportError.invalidQuantization(name)
        }

        try validateRange(view.buffer, offset: view.offset, length: view.length, name: name)
        try validateRange(view.buffer, offset: view.scaleOffset, length: view.scaleLength, name: "\(name).scales")
        try validateRange(view.buffer, offset: view.biasOffset, length: view.biasLength, name: "\(name).biases")

        let shape = "\(vocabSize),\(hiddenSize)"
        return TensorPayload(
            tensors: [
                TensorSlice(name: "\(name).weight", dtype: "U8", shape: [Int(view.length)],
                            buffer: view.buffer, bufferOffset: view.offset, length: view.length),
                TensorSlice(name: "\(name).scales", dtype: "BF16", shape: [Int(view.scaleLength / 2)],
                            buffer: view.buffer, bufferOffset: view.scaleOffset, length: view.scaleLength),
                TensorSlice(name: "\(name).biases", dtype: "BF16", shape: [Int(view.biasLength / 2)],
                            buffer: view.buffer, bufferOffset: view.biasOffset, length: view.biasLength),
            ],
            metadata: [
                "\(name).logical_shape": shape,
                "\(name).quantization_bits": "4",
                "\(name).quantization_group_size": "64",
                "\(name).packing": "gturbo_q4_affine",
            ]
        )
    }

    private static func validateRange(_ buffer: MTLBuffer, offset: UInt64, length: UInt64, name: String) throws {
        guard offset <= UInt64(buffer.length),
              length <= UInt64(buffer.length) - offset else {
            throw ExportError.invalidTensorRange(name)
        }
    }

    private static func makeHeader(tensors: [TensorSlice], metadata: [String: String]) throws -> Data {
        var entries: [String: Any] = ["__metadata__": metadata]
        var offset: UInt64 = 0
        for index in tensors.indices {
            var tensor = tensors[index]
            tensor.offset = offset
            let end = offset + tensor.length
            entries[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            ]
            offset = end
        }

        var json = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        while !json.count.isMultiple(of: 8) {
            json.append(0x20)
        }
        var header = Data()
        var headerLength = UInt64(json.count).littleEndian
        withUnsafeBytes(of: &headerLength) { header.append(contentsOf: $0) }
        header.append(json)
        return header
    }
}
