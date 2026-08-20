import Darwin
import Foundation
import Metal

public struct FeasibilityBenchmarkRun: Sendable, Equatable {
    public let io: FeasibilityBenchmarkIO
    public let compute: FeasibilityBenchmarkCompute
    public let scenarios: [FeasibilityBenchmarkScenario]

    public init(io: FeasibilityBenchmarkIO,
                compute: FeasibilityBenchmarkCompute,
                scenarios: [FeasibilityBenchmarkScenario]) {
        self.io = io
        self.compute = compute
        self.scenarios = scenarios
    }
}

public enum FeasibilityBenchmarkRunner {
    public static let hiddenSize = 2_048
    public static let outputVocabularySize = 248_320
    public static let bundleLayers = 4
    public static let totalLayers = 40
    public static let prefillTokens = 512
    public static let defaultWarmupRuns = 1
    public static let defaultMeasuredRuns = 5

    public static func run(payloadPath: String,
                           warmupRuns: Int = defaultWarmupRuns,
                           measuredRuns: Int = defaultMeasuredRuns) throws
        -> FeasibilityBenchmarkRun {
        guard warmupRuns > 0, measuredRuns > 0 else {
            throw RepackError.configurationInvalid(
                detail: "benchmark run counts must be positive")
        }
        let ioRuns = try runIO(payloadPath: payloadPath,
                       measuredRuns: measuredRuns)
        let computeRuns = try runCompute(warmupRuns: warmupRuns,
                                         measuredRuns: measuredRuns)
        let scenarios = ioRuns.traces.map { trace in
            let median = computeRuns.decodeMedianMilliseconds
                + trace.medianMilliseconds
            let p95 = computeRuns.decodeP95Milliseconds
                + trace.p95Milliseconds
            let bytes = trace.bytesPerToken
            return FeasibilityBenchmarkScenario(
                expertHitRate: trace.expertHitRate,
                medianMilliseconds: median,
                p95Milliseconds: max(p95, median),
                bytesPerToken: max(bytes, 1),
                    requiredSSDBandwidthBytesPerSecond:
                        Double(max(bytes, 1)) / (max(trace.medianMilliseconds, 0.001) / 1_000))
        }
        return FeasibilityBenchmarkRun(
            io: FeasibilityBenchmarkIO(
                expertBytesRead: ioRuns.bytesPerToken,
                preadOperations: ioRuns.preadOperations,
                elapsedMilliseconds: ioRuns.medianMilliseconds),
            compute: FeasibilityBenchmarkCompute(
                decodeBundleMedianMilliseconds:
                    computeRuns.decodeMedianMilliseconds,
                decodeBundleP95Milliseconds: computeRuns.decodeP95Milliseconds,
                prefillBundleMedianMilliseconds:
                    computeRuns.prefillMedianMilliseconds,
                prefillBundleP95Milliseconds:
                    computeRuns.prefillP95Milliseconds,
                outputVocabularySize: outputVocabularySize,
                outputBytesTouched: computeRuns.outputBytesTouched,
                commandBuffersCompleted: computeRuns.commandBuffersCompleted),
            scenarios: scenarios)
    }

    private struct IORun {
        let bytesPerToken: UInt64
        let preadOperations: Int
        let medianMilliseconds: Double
        let p95Milliseconds: Double
        let traces: [IOTrace]
    }

    private struct IOTrace {
        let expertHitRate: Double
        let bytesPerToken: UInt64
        let medianMilliseconds: Double
        let p95Milliseconds: Double
    }

    private static func runIO(payloadPath: String,
                              measuredRuns: Int) throws -> IORun {
        let fd = try Posix.openRead(payloadPath)
        defer { close(fd) }
        let fileSize = try Posix.fileSize(fd: fd, path: payloadPath)
        guard fileSize > 0 else {
            throw RepackError.configurationInvalid(
                detail: "benchmark payload is empty")
        }
        let fillBytes = min(fileSize, UInt64(Posix.pageSize * 16))
        let slotCount = 8
        var traces: [IOTrace] = []
        var coldRun: (samples: [Double], operations: Int, bytes: UInt64)?
        for hitRate in [0.0, 0.25, 0.5, 0.75] {
            let missCount = Int(Double(slotCount) * (1.0 - hitRate))
            var samples: [Double] = []
            var operations = 0
            var bytes: UInt64 = 0
            var checksum: UInt8 = 0
            for _ in 0..<measuredRuns {
                let start = DispatchTime.now().uptimeNanoseconds
                for slot in 0..<missCount {
                    let rawOffset = UInt64(slot) * fillBytes
                    let offset = rawOffset - rawOffset % UInt64(Posix.pageSize)
                    let available = fileSize - min(offset, fileSize)
                    let count = Int(min(fillBytes, available))
                    guard count > 0 else { continue }
                    var buffer = Data(count: count)
                    try buffer.withUnsafeMutableBytes { raw in
                        try Posix.preadAll(fd: fd,
                                           path: payloadPath,
                                           buf: raw.baseAddress!,
                                           count: raw.count,
                                           offset: offset)
                        checksum ^= raw.load(as: UInt8.self)
                    }
                    operations += 1
                    bytes += UInt64(count)
                }
                let elapsed = DispatchTime.now().uptimeNanoseconds - start
                samples.append(Double(elapsed) / 1_000_000)
            }
            guard checksum != 255 || bytes == 0 else {
                throw RepackError.configurationInvalid(
                    detail: "I/O fill was not observed")
            }
            let trace = IOTrace(
                expertHitRate: hitRate,
                bytesPerToken: max(bytes / UInt64(measuredRuns), 1),
                medianMilliseconds: median(samples),
                p95Milliseconds: percentile(samples, 0.95))
            traces.append(trace)
            if hitRate == 0 {
                coldRun = (samples, operations, bytes)
            }
        }
        guard let coldRun else {
            throw RepackError.configurationInvalid(
                detail: "cold I/O trace was not measured")
        }
        return IORun(bytesPerToken: max(coldRun.bytes / UInt64(measuredRuns), 1),
                     preadOperations: coldRun.operations,
                     medianMilliseconds: median(coldRun.samples),
                     p95Milliseconds: percentile(coldRun.samples, 0.95),
                     traces: traces)
    }

    private struct ComputeRun {
        let decodeMedianMilliseconds: Double
        let decodeP95Milliseconds: Double
        let prefillMedianMilliseconds: Double
        let prefillP95Milliseconds: Double
        let outputBytesTouched: UInt64
        let commandBuffersCompleted: Int
    }

    private static func runCompute(warmupRuns: Int,
                                   measuredRuns: Int) throws -> ComputeRun {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw RepackError.configurationInvalid(
                detail: "Metal device or queue is unavailable")
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: shaderSource,
                                             options: options)
        let hiddenPipeline = try makePipeline(
            device: device,
            library: library,
            name: "qwen36_feasibility_hidden")
        let headPipeline = try makePipeline(
            device: device,
            library: library,
            name: "qwen36_feasibility_lm_head")
        let input = try makeInputBuffer(device: device,
                                        tokenCount: prefillTokens)
        let hidden = try makeBuffer(device: device,
                                    length: prefillTokens * hiddenSize * 2)
        let logits = try makeBuffer(device: device,
                                    length: outputVocabularySize * 2)
        let decodeSamples = try measure(
            queue: queue,
            hiddenPipeline: hiddenPipeline,
            headPipeline: headPipeline,
            input: input,
            hidden: hidden,
            logits: logits,
            tokenCount: 1,
            warmupRuns: warmupRuns,
            measuredRuns: measuredRuns)
        let prefillSamples = try measure(
            queue: queue,
            hiddenPipeline: hiddenPipeline,
            headPipeline: headPipeline,
            input: input,
            hidden: hidden,
            logits: logits,
            tokenCount: prefillTokens,
            warmupRuns: warmupRuns,
            measuredRuns: measuredRuns)
        return ComputeRun(
            decodeMedianMilliseconds: median(decodeSamples.samples),
            decodeP95Milliseconds: percentile(decodeSamples.samples, 0.95),
            prefillMedianMilliseconds: median(prefillSamples.samples),
            prefillP95Milliseconds: percentile(prefillSamples.samples, 0.95),
            outputBytesTouched: UInt64(logits.length),
            commandBuffersCompleted:
                decodeSamples.completed + prefillSamples.completed)
    }

    private struct ComputeSamples {
        let samples: [Double]
        let completed: Int
    }

    private static func measure(queue: MTLCommandQueue,
                               hiddenPipeline: MTLComputePipelineState,
                               headPipeline: MTLComputePipelineState,
                               input: MTLBuffer,
                               hidden: MTLBuffer,
                               logits: MTLBuffer,
                               tokenCount: Int,
                               warmupRuns: Int,
                               measuredRuns: Int) throws -> ComputeSamples {
        for _ in 0..<warmupRuns {
            _ = try encodeAndWait(queue: queue,
                                  hiddenPipeline: hiddenPipeline,
                                  headPipeline: headPipeline,
                                  input: input,
                                  hidden: hidden,
                                  logits: logits,
                                  tokenCount: tokenCount)
        }
        var samples: [Double] = []
        var completed = 0
        for _ in 0..<measuredRuns {
            let result = try encodeAndWait(queue: queue,
                                           hiddenPipeline: hiddenPipeline,
                                           headPipeline: headPipeline,
                                           input: input,
                                           hidden: hidden,
                                           logits: logits,
                                           tokenCount: tokenCount)
            samples.append(result.elapsedMilliseconds)
            completed += 1
            guard result.touchedBytes > 0 else {
                throw RepackError.configurationInvalid(
                    detail: "Metal output buffer was not touched")
            }
        }
        return ComputeSamples(samples: samples, completed: completed)
    }

    private struct CommandResult {
        let elapsedMilliseconds: Double
        let touchedBytes: Int
    }

    private static func encodeAndWait(queue: MTLCommandQueue,
                                      hiddenPipeline: MTLComputePipelineState,
                                      headPipeline: MTLComputePipelineState,
                                      input: MTLBuffer,
                                      hidden: MTLBuffer,
                                      logits: MTLBuffer,
                                      tokenCount: Int) throws -> CommandResult {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let hiddenEncoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw RepackError.configurationInvalid(
                detail: "Metal command buffer could not be created")
        }
        var tokenCountValue = UInt32(tokenCount)
        var hiddenSizeValue = UInt32(hiddenSize)
        var vocabularyValue = UInt32(outputVocabularySize)
        hiddenEncoder.setComputePipelineState(hiddenPipeline)
        hiddenEncoder.setBuffer(input, offset: 0, index: 0)
        hiddenEncoder.setBuffer(hidden, offset: 0, index: 1)
        hiddenEncoder.setBytes(&tokenCountValue,
                               length: MemoryLayout<UInt32>.size,
                               index: 2)
        hiddenEncoder.setBytes(&hiddenSizeValue,
                               length: MemoryLayout<UInt32>.size,
                               index: 3)
        hiddenEncoder.dispatchThreads(
            MTLSize(width: hiddenSize, height: tokenCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(hiddenPipeline.maxTotalThreadsPerThreadgroup, 256),
                                           height: 1,
                                           depth: 1))
        hiddenEncoder.endEncoding()
        guard let headEncoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw RepackError.configurationInvalid(
                detail: "Metal head encoder could not be created")
        }
        headEncoder.setComputePipelineState(headPipeline)
        headEncoder.setBuffer(hidden,
                              offset: (tokenCount - 1) * hiddenSize * 2,
                              index: 0)
        headEncoder.setBuffer(logits, offset: 0, index: 1)
        headEncoder.setBytes(&hiddenSizeValue,
                             length: MemoryLayout<UInt32>.size,
                             index: 2)
        headEncoder.setBytes(&vocabularyValue,
                             length: MemoryLayout<UInt32>.size,
                             index: 3)
        headEncoder.dispatchThreads(
            MTLSize(width: outputVocabularySize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(headPipeline.maxTotalThreadsPerThreadgroup, 256),
                                           height: 1,
                                           depth: 1))
        headEncoder.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        let output = logits.contents().assumingMemoryBound(to: UInt16.self)
        var touched = 0
        for index in 0..<min(16, logits.length / MemoryLayout<UInt16>.size) {
            touched |= Int(output[index])
        }
        return CommandResult(elapsedMilliseconds: Double(elapsed) / 1_000_000,
                             touchedBytes: touched == 0 ? 0 : logits.length)
    }

    private static func makePipeline(device: MTLDevice,
                                     library: MTLLibrary,
                                     name: String) throws
        -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw RepackError.configurationInvalid(
                detail: "Metal benchmark function missing: \(name)")
        }
        return try device.makeComputePipelineState(function: function)
    }

    private static func makeBuffer(device: MTLDevice,
                                   length: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length,
                                             options: .storageModeShared)
        else {
            throw RepackError.configurationInvalid(
                detail: "Metal benchmark buffer allocation failed")
        }
        memset(buffer.contents(), 0, length)
        return buffer
    }

    private static func makeInputBuffer(device: MTLDevice,
                                        tokenCount: Int) throws -> MTLBuffer {
        let buffer = try makeBuffer(device: device,
                                    length: tokenCount * hiddenSize * 2)
        let values = buffer.contents().assumingMemoryBound(to: UInt16.self)
        for index in 0..<(tokenCount * hiddenSize) {
            values[index] = UInt16(Float16((index % 31) - 15).bitPattern)
        }
        return buffer
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private static func percentile(_ values: [Double], _ fraction: Double)
        -> Double {
        let sorted = values.sorted()
        let index = min(sorted.count - 1,
                        Int(Double(sorted.count - 1) * fraction))
        return sorted[index]
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void qwen36_feasibility_hidden(
        device const half* input [[buffer(0)]],
        device half* hidden [[buffer(1)]],
        constant uint& token_count [[buffer(2)]],
        constant uint& hidden_size [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= hidden_size || gid.y >= token_count) return;
        float value = float(input[gid.y * hidden_size + gid.x]);
        for (uint layer = 0; layer < 4; ++layer) {
            float recurrent = 0.0f;
            for (uint key = 0; key < 16; ++key) {
                recurrent = fma(value, 0.001f * float(key + 1), recurrent);
            }
            float attention = 0.0f;
            for (uint head = 0; head < 16; ++head) {
                attention = fma(value, 0.002f * float(head + 1), attention);
            }
            float routed = 0.0f;
            for (uint expert = 0; expert < 8; ++expert) {
                float gate = 1.0f / (1.0f + exp(-value * 0.01f));
                routed += gate * (value + float(expert) * 0.0001f);
            }
            float shared_gate = 1.0f / (1.0f + exp(-value));
            value = tanh(recurrent + attention + routed * shared_gate);
        }
        hidden[gid.y * hidden_size + gid.x] = half(value);
    }

    kernel void qwen36_feasibility_lm_head(
        device const half* hidden [[buffer(0)]],
        device half* logits [[buffer(1)]],
        constant uint& hidden_size [[buffer(2)]],
        constant uint& vocabulary_size [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= vocabulary_size) return;
        float value = 0.0f;
        for (uint index = 0; index < hidden_size; ++index) {
            float weight = float((index + gid) % 17) * 0.0001f - 0.0008f;
            value = fma(float(hidden[index]), weight, value);
        }
        logits[gid] = half(value);
    }
    """
}