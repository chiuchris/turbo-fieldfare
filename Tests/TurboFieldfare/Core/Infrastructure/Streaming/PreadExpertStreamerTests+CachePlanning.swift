import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

extension PreadExpertStreamerTests {
  @Test func leasedPlanPreventsEvictionAndDirectWritesUntilReleased() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 2)
    _ = try streamer.loadExpertsCached(experts: [0])
    let plan = streamer.planExpertsCached(experts: [0, 1])
    let lease = try streamer.pinExpertCachePlan(plan)
    defer { lease.release() }
    let results = try streamer.executeExpertCachePlan(plan)

    #expect(streamer.planExpertsCachedIfPossible(experts: [2]) == nil)
    #expect(throws: ExpertCachePlanError.self) {
      _ = try streamer.loadExpert(layer: 0, expert: 2, slot: plan.assignedSlots[0])
    }
    for index in plan.experts.indices {
      #expect(Self.bytes(of: results[index].buffer, offset: 0, count: Self.expertStride)
        .allSatisfy { $0 == Self.tagByte(plan.experts[index]) })
    }
    lease.release()
    lease.release()
    let next = try #require(streamer.planExpertsCachedIfPossible(experts: [2, 3]))
    _ = try streamer.executeExpertCachePlan(next)
  }

  @Test func staleAndForeignPlansCannotBePinnedOrExecuted() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 1)
    let other = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 1)
    let stale = streamer.planExpertsCached(experts: [0])
    let current = streamer.planExpertsCached(experts: [1])
    #expect(throws: ExpertCachePlanError.self) { _ = try streamer.pinExpertCachePlan(stale) }
    #expect(throws: ExpertCachePlanError.self) { _ = try streamer.executeExpertCachePlan(stale) }
    #expect(throws: ExpertCachePlanError.self) { _ = try other.pinExpertCachePlan(current) }
    _ = try streamer.executeExpertCachePlan(current)
  }

  @Test func independentLeasesReleaseOnlyTheirOwnPins() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 1)
    _ = try streamer.loadExpertsCached(experts: [0])
    let plan = streamer.planExpertsCached(experts: [0])
    var first: ExpertCacheLease? = try streamer.pinExpertCachePlan(plan)
    let second = try streamer.pinExpertCachePlan(plan)
    #expect(first != nil)
    first = nil
    #expect(streamer.planExpertsCachedIfPossible(experts: [1]) == nil)
    second.release()
    #expect(streamer.planExpertsCachedIfPossible(experts: [1]) != nil)
  }

  @Test func concurrentChurnCannotOverwriteLeasedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 1)
    let results = try streamer.loadExpertsCached(experts: [0])
    let plan = streamer.planExpertsCached(experts: [0])
    let lease = try streamer.pinExpertCachePlan(plan)
    defer { lease.release() }

    DispatchQueue.concurrentPerform(iterations: 64) { _ in
      do {
        let reader = try streamer.pinExpertCachePlan(plan)
        defer { reader.release() }
        #expect(streamer.planExpertsCachedIfPossible(experts: [1]) == nil)
        #expect(throws: ExpertCachePlanError.self) {
          _ = try streamer.loadExpert(layer: 0, expert: 1, slot: 0)
        }
        reader.release()
        reader.release()
      } catch {
        Issue.record("concurrent reader failed: \(error)")
      }
    }
    #expect(Self.bytes(of: results[0].buffer, offset: 0, count: Self.expertStride)
      .allSatisfy { $0 == Self.tagByte(0) })
    lease.release()
    _ = try streamer.loadExpertsCached(experts: [1])
  }

  @Test func directWriteInvalidatesPreviouslyPlannedCacheHits() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 1)
    _ = try streamer.loadExpertsCached(experts: [0])
    let oldHit = streamer.planExpertsCached(experts: [0])
    _ = try streamer.loadExpert(layer: 0, expert: 1, slot: 0)
    #expect(throws: ExpertCachePlanError.self) { _ = try streamer.pinExpertCachePlan(oldHit) }
    #expect(throws: ExpertCachePlanError.self) { _ = try streamer.executeExpertCachePlan(oldHit) }
    let refill = streamer.planExpertsCached(experts: [0])
    #expect(refill.hits == 0)
    let results = try streamer.executeExpertCachePlan(refill)
    #expect(Self.bytes(of: results[0].buffer, offset: 0, count: Self.expertStride)
      .allSatisfy { $0 == Self.tagByte(0) })
  }

  @Test func failedReadReleasesExecutionPinsWithoutPublishingResidency() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 2)
    let plan = streamer.planExpertsCached(experts: [0, 1])
    let consumer = try streamer.pinExpertCachePlan(plan)
    defer { consumer.release() }
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.truncate(atOffset: Self.streamOffset)
    #expect(throws: (any Error).self) { _ = try streamer.executeExpertCachePlan(plan) }
    #expect(streamer.planExpertsCachedIfPossible(experts: [2, 3]) == nil)
    consumer.release()
    let retry = try #require(streamer.planExpertsCachedIfPossible(experts: [0, 1]))
    #expect(retry.hits == 0)
    #expect(retry.misses == [0, 1])
  }

  @Test func cachedBatchWithoutExecutorLoadsTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let results = try streamer.loadExpertsCached(experts: [3, 1, 2])
    for (index, result) in results.enumerated() {
      let expert = [3, 1, 2][index]
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func adviseExpertsDoesNotChangeLoadedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let experts = [0, 2, 3]

    let advice = streamer.adviseExperts(experts: experts)
    #expect(advice.requested == experts.count)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == experts.count)
    #endif

    let results = try streamer.loadExpertsCached(experts: experts)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func adviseExpertMissesSkipsResidentSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let advice = streamer.adviseExpertMisses(experts: [0, 1, 2])

    #expect(advice.requested == 2)
    #expect(advice.calls == 1)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == 1)
    #endif
  }

  @Test func plannedCacheLoadExecutesSameMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)

    #expect(plan.hits == 1)
    #expect(plan.misses.map { experts[$0] } == [1, 2])

    let results = try streamer.executeExpertCachePlan(plan)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func plannedCacheDiagnosticsMeasureOnlyMissReads() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)
    let execution = try streamer.executeExpertCachePlanWithDiagnostics(plan)

    #expect(execution.readDiagnostics.readCount == 2)
    #expect(execution.readDiagnostics.totalNanos > 0)
    #expect(execution.readDiagnostics.maxNanos > 0)
    #expect(execution.readDiagnostics.maxNanos <= execution.readDiagnostics.totalNanos)
    for (index, result) in execution.buffers.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func plannedCacheBuffersExposeReservedSlotsBeforeExecute() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)
    let reserved = streamer.expertCachePlanBuffers(plan)

    let hitBytes = Self.bytes(of: reserved[0].buffer, offset: 0, count: Self.expertStride)
    #expect(hitBytes.allSatisfy { $0 == Self.tagByte(0) })

    let executed = try streamer.executeExpertCachePlan(plan)
    for i in 0..<experts.count {
      #expect(reserved[i].buffer === executed[i].buffer)
      let got = Self.bytes(of: executed[i].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[i]) })
    }
  }

  @Test func plannedCacheAvoidsInFlightSlotsForHitsAndMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let warmed = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCached(
      experts: [0, 2],
      avoidingSlots: [0, 1])

    #expect(plan.assignedSlots == [0, 2])
    #expect(plan.hits == 1)
    #expect(plan.misses == [1])

    let executed = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in plan.experts.enumerated() {
      let got = Self.bytes(of: executed[index].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }

    let avoidedBytes = Self.bytes(of: warmed[0].buffer, offset: 0, count: Self.expertStride)
    #expect(avoidedBytes.allSatisfy { $0 == Self.tagByte(0) })
  }

  @Test func plannedCacheReturnsNilWhenMissesCannotAvoidInFlightSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCachedIfPossible(
      experts: [0, 2, 3, 4],
      avoidingSlots: [0, 1])

    #expect(plan == nil)
  }

  @Test func plannedCacheAdaptiveExecutionHandlesSerialAndZeroMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let serialPlan = streamer.planExpertsCached(experts: [0, 2])
    let serialExecution = try streamer.executeExpertCachePlanWithDiagnostics(serialPlan)

    #expect(serialPlan.misses.count == 1)
    #expect(serialExecution.readDiagnostics.readCount == 1)
    #expect(serialExecution.readDiagnostics.totalNanos > 0)
    #expect(Self.bytes(
      of: serialExecution.buffers[1].buffer,
      offset: 0,
      count: Self.expertStride).allSatisfy { $0 == Self.tagByte(2) })

    let zeroMissPlan = streamer.planExpertsCached(experts: [0, 2])
    let zeroMissExecution = try streamer.executeExpertCachePlanWithDiagnostics(zeroMissPlan)

    #expect(zeroMissPlan.misses.isEmpty)
    #expect(zeroMissExecution.readDiagnostics == ExpertReadDiagnostics())
    for index in zeroMissPlan.experts.indices {
      #expect(zeroMissExecution.buffers[index].buffer === serialExecution.buffers[index].buffer)
    }
  }

  @Test func plannedCacheParallelExecutionPreservesAllMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let experts = [0, 1, 2, 3]

    let plan = streamer.planExpertsCached(experts: experts)
    let execution = try streamer.executeExpertCachePlanWithDiagnostics(plan)

    #expect(plan.misses == experts.indices.map { $0 })
    #expect(execution.readDiagnostics.readCount == experts.count)
    for (index, expert) in experts.enumerated() {
      let got = Self.bytes(
        of: execution.buffers[index].buffer,
        offset: 0,
        count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func lruEvictsLeastRecentlyUsedSlot() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path),
      device: device,
      slotCount: 2,
      cachePolicy: .lru)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [1])

    let plan = streamer.planExpertsCached(experts: [2])
    #expect(plan.assignedSlots == [0])
    #expect(plan.misses == [0])
  }

  @Test func lfuEvictsLowerUseCountSlot() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path),
      device: device,
      slotCount: 2,
      cachePolicy: .lfu)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [1])

    let plan = streamer.planExpertsCached(experts: [2])
    #expect(plan.assignedSlots == [1])
    #expect(plan.misses == [0])
  }

}
