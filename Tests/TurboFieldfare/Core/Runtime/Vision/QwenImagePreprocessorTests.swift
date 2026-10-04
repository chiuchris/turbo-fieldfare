import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

@Suite struct QwenImagePreprocessorTests {
    @Test func geometryUsesPinnedPixelBudgetsAndPatchAlignment() throws {
        let small = try QwenImageGeometry(sourceWidth: 32, sourceHeight: 32)
        #expect(small.processedWidth == 256)
        #expect(small.processedHeight == 256)
        #expect(small.patchGridWidth == 16)
        #expect(small.patchGridHeight == 16)
        #expect(small.tokenCount == 64)

        let large = try QwenImageGeometry(sourceWidth: 5_000, sourceHeight: 5_000)
        #expect(large.processedWidth.isMultiple(of: QwenImageGeometry.alignment))
        #expect(large.processedHeight.isMultiple(of: QwenImageGeometry.alignment))
        #expect(large.processedWidth * large.processedHeight
                <= QwenImageGeometry.maximumPixels)
        #expect(large.processedWidth * large.processedHeight
                >= QwenImageGeometry.minimumPixels)
    }

    @Test func preprocessesMergeOrderedChannelLastPatchesWithQwenNormalization() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-vision-\(UUID().uuidString).png")
        try writeSolidPNG(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let result = try QwenImagePreprocessor(device: device).preprocess(fileURL: url)
        #expect(result.geometry.processedWidth == 256)
        #expect(result.geometry.processedHeight == 256)
        #expect(result.patchesBF16.length
                == result.geometry.patchCount
                    * QwenImageGeometry.patchVectorDimension
                    * MemoryLayout<UInt16>.stride)

        let positions = result.positionsInt32x2.contents().bindMemory(
            to: Int32.self, capacity: result.geometry.patchCount * 2)
        let expected: [(Int32, Int32)] = [
            (0, 0), (1, 0), (0, 1), (1, 1),
            (2, 0), (3, 0), (2, 1), (3, 1),
        ]
        for (index, position) in expected.enumerated() {
            #expect(positions[index * 2] == position.0)
            #expect(positions[index * 2 + 1] == position.1)
        }

        let patch = result.patchesBF16.contents().bindMemory(
            to: UInt16.self, capacity: QwenImageGeometry.patchVectorDimension)
        let temporalFrameSize = 3 * QwenImageGeometry.patchSize
            * QwenImageGeometry.patchSize
        #expect(abs(Quantization.bf16ToFloat(patch[0]) - 1) < 0.01)
        #expect(Quantization.bf16ToFloat(patch[1]) < 0)
        #expect(Quantization.bf16ToFloat(patch[2]) < 0)
        #expect(patch[3] == patch[0])
        #expect(patch[temporalFrameSize] == patch[0])
    }

    @Test func admissionGeometryMatchesPreprocessedGeometry() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-vision-admission-\(UUID().uuidString).png")
        try writeSolidPNG(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = QwenImagePreprocessor(device: device)
        let admission = try preprocessor.admissionGeometry(fileURL: url)
        let processed = try preprocessor.preprocess(fileURL: url)

        #expect(admission == processed.geometry)
    }

    private func writeSolidPNG(to url: URL) throws {
        let side = 256
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: side,
                height: side,
                bitsPerComponent: 8,
                bytesPerRow: side * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw VisionImageError.allocationFailed
        }
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw VisionImageError.decodeFailed
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw VisionImageError.decodeFailed
        }
    }
}
