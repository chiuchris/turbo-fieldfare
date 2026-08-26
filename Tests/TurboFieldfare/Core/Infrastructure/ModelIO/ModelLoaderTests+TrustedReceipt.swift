import Foundation
import Darwin
import Metal
import Testing

@testable import TurboFieldfare

extension ModelLoaderTests {
  @Test func receiptReaderRejectsFIFOWithoutBlocking() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let receipt = dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
    #expect(mkfifo(receipt.path, 0o600) == 0)

    #expect(throws: ModelError.self) {
      _ = try VerifiedInstallReceiptReader.load(directoryURL: dir)
    }
  }

  @Test func trustedReceiptModeFallsBackWhenReceiptIsMissing() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    var stats = ModelLoadStats()
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt,
      loadStats: &stats)

    #expect(model.integrityPolicy == .fullSha256)
    #expect(stats.eagerSha256Nanos > 0)
  }

  @Test func trustedReceiptReaderRejectsOversizedMetadataBeforeDecode() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)

    #expect {
      _ = try VerifiedInstallReceiptReader.load(directoryURL: dir, maxBytes: 16)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid(let detail) = error {
        return detail.contains("metadata cap")
      }
      return false
    }
  }

  @Test func trustedReceiptModeFallsBackForMalformedReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let receipt = dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
    try Data("{".utf8).write(to: receipt)
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

  @Test func trustedReceiptModeFallsBackForUnsupportedSchema() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["schemaVersion"] = 1
    }
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

  @Test func trustedReceiptModeFallsBackForSameSizeLayerMutation() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let layerURL =
      dir
      .appendingPathComponent("packed_experts")
      .appendingPathComponent("layer_00.bin")
    try Self.flipByte(in: layerURL, at: 64)
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
    #expect {
      _ = try model.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  @Test func trustedReceiptModeFallsBackForMismatchedArtifactDigest() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      var files = root["files"] as! [String: Any]
      var weights = files["model_weights.bin"] as! [String: Any]
      weights["sha256"] = String(repeating: "0", count: 64)
      files["model_weights.bin"] = weights
      root["files"] = files
    }
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

  @Test func trustedReceiptModeFallsBackForResidentIdentityChange() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let weights = dir.appendingPathComponent("model_weights.bin")
    try FileManager.default.setAttributes(
      [.modificationDate: Date(timeIntervalSince1970: 1)],
      ofItemAtPath: weights.path)
    let device = try #require(MTLCreateSystemDefaultDevice())
    var stats = ModelLoadStats()

    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt,
      loadStats: &stats)
    #expect(model.integrityPolicy == .fullSha256)
    #expect(stats.eagerSha256Nanos > 0)
  }

  @Test func trustedReceiptModeHashesLayerChangedAfterLoad() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)

    let layerURL =
      dir
      .appendingPathComponent("packed_experts")
      .appendingPathComponent("layer_00.bin")
    try Self.flipByte(in: layerURL, at: 64)
    #expect {
      _ = try model.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  @Test func trustedReceiptModeFallsBackForWrongSizedLayerFile() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let layerURL =
      dir
      .appendingPathComponent("packed_experts")
      .appendingPathComponent("layer_00.bin")
    let handle = try FileHandle(forWritingTo: layerURL)
    try handle.truncate(atOffset: 1024)
    try handle.close()

    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
    #expect {
      _ = try model.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.tensorSizeMismatch = error { return true }
      return false
    }
  }

  @Test func trustedReceiptModeSkipsEagerSHAAndReportsValidationTiming() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())
    var stats = ModelLoadStats()
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt,
      loadStats: &stats)
    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
    #expect(stats.manifestSha256Nanos > 0)
    #expect(stats.receiptValidationNanos > 0)
    #expect(stats.eagerSha256Nanos == 0)
  }

  @Test func trustedReceiptModeFallsBackForExtraReceiptFileEntry() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      var files = root["files"] as! [String: Any]
      files["unexpected.bin"] = ["size": 0, "sha256": String(repeating: "0", count: 64)]
      root["files"] = files
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

  @Test func trustedReceiptModeFallsBackForMissingReceiptFileEntry() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      var files = root["files"] as! [String: Any]
      files.removeValue(forKey: "packed_experts/layer_00.bin")
      root["files"] = files
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

  @Test func trustedReceiptModeFallsBackForStaleManifestBinding() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["manifestSha256"] = String(repeating: "0", count: 64)
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

  @Test func trustedReceiptModeFallsBackForDifferentModelDirectoryBinding() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["modelDirectoryPath"] =
        dir
        .deletingLastPathComponent()
        .appendingPathComponent("other.gturbo")
        .standardizedFileURL
        .path
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.integrityPolicy == .fullSha256)
  }

}
