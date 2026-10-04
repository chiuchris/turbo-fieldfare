import CoreGraphics
import Foundation
import ImageIO
import Metal

public struct QwenImageGeometry: Equatable, Sendable {
    public static let patchSize = 16
    public static let temporalPatchSize = 2
    public static let mergeSize = 2
    public static let alignment = patchSize * mergeSize
    public static let minimumPixels = 65_536
    public static let maximumPixels = 16_777_216
    public static let maximumAspectRatio = 200
    public static let patchVectorDimension = 3 * temporalPatchSize * patchSize * patchSize

    public let sourceWidth: Int
    public let sourceHeight: Int
    public let processedWidth: Int
    public let processedHeight: Int

    public var patchGridWidth: Int { processedWidth / Self.patchSize }
    public var patchGridHeight: Int { processedHeight / Self.patchSize }
    public var patchCount: Int { patchGridWidth * patchGridHeight }
    public var tokenCount: Int { patchCount / (Self.mergeSize * Self.mergeSize) }

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        minimumPixels: Int = Self.minimumPixels,
        maximumPixels: Int = Self.maximumPixels
    ) throws {
        guard sourceWidth > 0, sourceHeight > 0,
              minimumPixels > 0, maximumPixels >= minimumPixels else {
            throw VisionImageError.invalidMetadata("invalid Qwen image geometry")
        }
        let aspectRatio = Double(max(sourceWidth, sourceHeight))
            / Double(min(sourceWidth, sourceHeight))
        guard aspectRatio <= Double(Self.maximumAspectRatio) else {
            throw VisionImageError.invalidMetadata(
                "image aspect ratio exceeds Qwen's limit of \(Self.maximumAspectRatio)")
        }

        var height = Self.nearestAligned(sourceHeight)
        var width = Self.nearestAligned(sourceWidth)
        let roundedPixels = try Self.checkedMultiply(height, width)
        if roundedPixels > maximumPixels {
            let scale = sqrt(Double(roundedPixels) / Double(maximumPixels))
            height = max(Self.alignment,
                         Self.floorAligned(Double(sourceHeight) / scale))
            width = max(Self.alignment,
                        Self.floorAligned(Double(sourceWidth) / scale))
        } else if roundedPixels < minimumPixels {
            let scale = sqrt(Double(minimumPixels) / Double(roundedPixels))
            height = max(Self.alignment,
                         Self.ceilAligned(Double(sourceHeight) * scale))
            width = max(Self.alignment,
                        Self.ceilAligned(Double(sourceWidth) * scale))
        }
        let processedPixels = try Self.checkedMultiply(height, width)
        guard processedPixels <= maximumPixels else {
            throw VisionImageError.invalidMetadata(
                "Qwen image resize exceeds its maximum pixel budget")
        }
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.processedWidth = width
        self.processedHeight = height
    }

    private static func nearestAligned(_ dimension: Int) -> Int {
        max(alignment,
            Int((Double(dimension) / Double(alignment)).rounded(.toNearestOrEven))
                * alignment)
    }

    private static func floorAligned(_ dimension: Double) -> Int {
        Int(floor(dimension / Double(alignment))) * alignment
    }

    private static func ceilAligned(_ dimension: Double) -> Int {
        Int(ceil(dimension / Double(alignment))) * alignment
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw VisionImageError.invalidMetadata("Qwen image dimensions overflow")
        }
        return value
    }
}

public struct QwenImagePixels {
    public let patchesBF16: MTLBuffer
    public let positionsInt32x2: MTLBuffer
    public let metadata: VisionImageMetadata
    public let geometry: QwenImageGeometry
    public let wallNanoseconds: UInt64
    public let allocatedBytes: Int
}

public final class QwenImagePreprocessor {
    private static let normalizedBF16 = (0...255).map {
        Quantization.bf16Bits(Float($0) / 127.5 - 1)
    }

    private let device: MTLDevice
    private let metadataReader: ImageMetadataReader

    public init(
        device: MTLDevice,
        limits: VisionImageLimits = VisionImageLimits()
    ) {
        self.device = device
        self.metadataReader = ImageMetadataReader(limits: limits)
    }

    public func admissionGeometry(fileURL: URL) throws -> QwenImageGeometry {
        let image = try VisionImageSource(fileURL: fileURL)
        let opened = try image.open(
            maximumEncodedBytes: metadataReader.limits.maximumEncodedBytes)
        let metadata = try metadataReader.read(
            opened: opened, verifyStreamCompleteness: false)
        return try QwenImageGeometry(
            sourceWidth: metadata.orientedWidth,
            sourceHeight: metadata.orientedHeight)
    }

    public func preprocess(fileURL: URL) throws -> QwenImagePixels {
        let started = ContinuousClock.now
        let image = try VisionImageSource(fileURL: fileURL)
        let opened = try image.open(
            maximumEncodedBytes: metadataReader.limits.maximumEncodedBytes)
        let metadata = try metadataReader.read(opened: opened)
        let geometry = try QwenImageGeometry(
            sourceWidth: metadata.orientedWidth,
            sourceHeight: metadata.orientedHeight)
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize:
                max(metadata.orientedWidth, metadata.orientedHeight),
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceShouldAllowFloat: false,
        ]
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(
            opened.source, 0, thumbnailOptions as CFDictionary) else {
            throw VisionImageError.decodeFailed
        }

        let sourceRowBytes = try checkedMultiply(decoded.width, 4)
        let sourceBytes = try checkedMultiply(sourceRowBytes, decoded.height)
        let sourceRGBA = UnsafeMutableRawPointer.allocate(
            byteCount: sourceBytes, alignment: 64)
        sourceRGBA.initializeMemory(as: UInt8.self, repeating: 255, count: sourceBytes)
        defer { sourceRGBA.deallocate() }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let drawing = CGContext(
                data: sourceRGBA,
                width: decoded.width,
                height: decoded.height,
                bitsPerComponent: 8,
                bytesPerRow: sourceRowBytes,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw VisionImageError.allocationFailed
        }
        drawing.setFillColor(CGColor(gray: 1, alpha: 1))
        drawing.fill(CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        drawing.draw(decoded, in: CGRect(x: 0, y: 0,
                                         width: decoded.width, height: decoded.height))
        withExtendedLifetime(drawing) {}

        let rowBytes = try checkedMultiply(geometry.processedWidth, 4)
        let rgbaBytes = try checkedMultiply(rowBytes, geometry.processedHeight)
        let rgba = UnsafeMutableRawPointer.allocate(byteCount: rgbaBytes, alignment: 64)
        defer { rgba.deallocate() }
        let resizeScratchBytes = TorchBicubicResize.resize(
            source: sourceRGBA.assumingMemoryBound(to: UInt8.self),
            sourceWidth: decoded.width,
            sourceHeight: decoded.height,
            sourceRowBytes: sourceRowBytes,
            destination: rgba.assumingMemoryBound(to: UInt8.self),
            destinationWidth: geometry.processedWidth,
            destinationHeight: geometry.processedHeight,
            destinationRowBytes: rowBytes)

        let patchElements = try checkedMultiply(
            geometry.patchCount, QwenImageGeometry.patchVectorDimension)
        let patchBytes = try checkedMultiply(patchElements, MemoryLayout<UInt16>.stride)
        let positionElements = try checkedMultiply(geometry.patchCount, 2)
        let positionBytes = try checkedMultiply(positionElements, MemoryLayout<Int32>.stride)
        guard let patches = device.makeBuffer(
                length: patchBytes, options: .storageModeShared),
              let positions = device.makeBuffer(
                length: positionBytes, options: .storageModeShared) else {
            throw VisionImageError.allocationFailed
        }
        patchify(
            rgba: rgba.assumingMemoryBound(to: UInt8.self),
            rowBytes: rowBytes,
            geometry: geometry,
            patches: patches,
            positions: positions)

        return QwenImagePixels(
            patchesBF16: patches,
            positionsInt32x2: positions,
            metadata: metadata,
            geometry: geometry,
            wallNanoseconds: nanoseconds(started.duration(to: .now)),
            allocatedBytes: sourceBytes * 2 + rgbaBytes + resizeScratchBytes
                + patchBytes + positionBytes)
    }

    private func patchify(
        rgba: UnsafePointer<UInt8>,
        rowBytes: Int,
        geometry: QwenImageGeometry,
        patches: MTLBuffer,
        positions: MTLBuffer
    ) {
        let patchPointer = patches.contents().bindMemory(
            to: UInt16.self,
            capacity: geometry.patchCount * QwenImageGeometry.patchVectorDimension)
        let positionPointer = positions.contents().bindMemory(
            to: Int32.self, capacity: geometry.patchCount * 2)
        let patchSize = QwenImageGeometry.patchSize
        let mergeSize = QwenImageGeometry.mergeSize
        let gridWidth = geometry.patchGridWidth
        let gridHeight = geometry.patchGridHeight
        var row = 0

        for groupY in 0..<(gridHeight / mergeSize) {
            for groupX in 0..<(gridWidth / mergeSize) {
                for offsetY in 0..<mergeSize {
                    for offsetX in 0..<mergeSize {
                        let patchX = groupX * mergeSize + offsetX
                        let patchY = groupY * mergeSize + offsetY
                        positionPointer[row * 2] = Int32(patchX)
                        positionPointer[row * 2 + 1] = Int32(patchY)
                        var output = row * QwenImageGeometry.patchVectorDimension
                        for _ in 0..<QwenImageGeometry.temporalPatchSize {
                            for pixelY in 0..<patchSize {
                                let inputRow = (patchY * patchSize + pixelY) * rowBytes
                                for pixelX in 0..<patchSize {
                                    let inputPixel = inputRow + (patchX * patchSize + pixelX) * 4
                                    for channel in 0..<3 {
                                        patchPointer[output] = Self.normalizedBF16[
                                            Int(rgba[inputPixel + channel])]
                                        output += 1
                                    }
                                }
                            }
                        }
                        row += 1
                    }
                }
            }
        }
    }

    private func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw VisionImageError.invalidMetadata("Qwen image buffer size overflow")
        }
        return value
    }

    private func nanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        return UInt64(max(0, components.seconds)) * 1_000_000_000
            + UInt64(max(0, components.attoseconds / 1_000_000_000))
    }
}
