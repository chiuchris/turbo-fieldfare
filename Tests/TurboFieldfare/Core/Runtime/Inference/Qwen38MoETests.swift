import Darwin
import Testing
@testable import TurboFieldfare

@Suite
struct Qwen38MoETests {
    @Test
    func qwen38RoutingGeometryMatchesCanonicalConfiguration() {
        let config = ArchConfig.qwen38FlashNextText

        #expect(Qwen38MoE.topK == 10)
        #expect(Qwen38MoE.numExperts == 512)
        #expect(Qwen38MoE.canonicalHiddenSize == 2560)
        #expect(Qwen38MoE.canonicalIntermediateSize == 640)
        #expect(config.topKExperts == Qwen38MoE.topK)
        #expect(config.numExperts == Qwen38MoE.numExperts)
        #expect(config.moeIntermediateSize == Qwen38MoE.canonicalIntermediateSize)
        #expect(config.intermediateSize == config.moeIntermediateSize)
        #expect(config.qwen38Architecture?.ngramSplitParts == 128)
        #expect(config.qwen38Architecture?.ngramVocabSizeDivisor == 128)
    }

    @Test
    func qwen36RoutingConstantsRemainTopEight() {
        #expect(QwenMoE.topK == 8)
        #expect(QwenMoE.qwen38TopK == 10)
        #expect(QwenMoE.qwen38NumExperts == 512)
    }

    @Test
    func rejectsRoutedExpertCountThatDoesNotMatchTopK() throws {
        let context = try MetalContext()
        let moe = try Qwen38MoE(context: context)
        let buffer = try #require(
            context.device.makeBuffer(length: 1, options: .storageModeShared))
        let view = TensorView(
            buffer: buffer,
            offset: 0,
            length: 1,
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (0, 0, 0, 0),
            dtype: 0)

        #expect {
            _ = try moe.makeRoutedArgumentBuffer(
                experts: Array(repeating: view, count: Qwen38MoE.topK - 1))
        } throws: { error in
            if case ModelError.archMismatch(let field, let expected, let actual) = error {
                return field == "qwen38RoutedExperts.count"
                    && expected == "10"
                    && actual == "9"
            }
            return false
        }
    }

    @Test
    func preservesTopTenSlotsWithNonZeroExpertOffsets() throws {
        let context = try MetalContext()
        let moe = try Qwen38MoE(context: context)
        let buffer = try #require(
            context.device.makeBuffer(length: 8192, options: .storageModeShared))
        var experts: [TensorView] = []
        for index in 0..<Qwen38MoE.topK {
            let offset: UInt64 = UInt64(64 + index * 128)
            let scaleOffset: UInt64 = UInt64(2048 + index * 64)
            let biasOffset: UInt64 = UInt64(3072 + index * 64)
            experts.append(TensorView(
                buffer: buffer,
                offset: offset,
                length: 64,
                scaleOffset: scaleOffset,
                scaleLength: 64,
                biasOffset: biasOffset,
                biasLength: 64,
                shape: (1, 2560, 0, 0),
                dtype: 0))
        }

        let first = try moe.makeRoutedArgumentBuffer(
            layer: 2, slot: 3, experts: experts)
        let repeated = try moe.makeRoutedArgumentBuffer(
            layer: 2, slot: 3, experts: experts)
        let differentSlot = try moe.makeRoutedArgumentBuffer(
            layer: 2, slot: 4, experts: experts)
        let differentLayer = try moe.makeRoutedArgumentBuffer(
            layer: 3, slot: 3, experts: experts)

        #expect(first === repeated)
        #expect(first !== differentSlot)
        #expect(first !== differentLayer)
    }

    @Test
    func subsetPhaseOneMatchesFullPhaseOneAtOriginalRouteSlots() throws {
        let context = try MetalContext()
        let moe = try Qwen38MoE(context: context)
        let expertBytes = 1_048_576
        let expertBuffer = try #require(context.device.makeBuffer(
            length: expertBytes, options: .storageModeShared))
        memset(expertBuffer.contents(), 0, expertBytes)
        let expert = TensorView(
            buffer: expertBuffer,
            offset: 0,
            length: UInt64(expertBytes),
            scaleOffset: 0,
            scaleLength: 0,
            biasOffset: 0,
            biasLength: 0,
            shape: (0, 0, 0, 0),
            dtype: 0)
        let routed = Array(repeating: expert, count: Qwen38MoE.topK)
        let argumentBuffer = try moe.makeRoutedArgumentBuffer(
            layer: 0, slot: 0, experts: routed)
        let input = try #require(context.device.makeBuffer(
            length: Qwen38MoE.canonicalHiddenSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        memset(input.contents(), 0, input.length)
        let activationBytes = Qwen38MoE.topK * Qwen38MoE.canonicalIntermediateSize
            * MemoryLayout<Float16>.stride
        let fullActivations = try #require(context.device.makeBuffer(
            length: activationBytes, options: .storageModeShared))
        let splitActivations = try #require(context.device.makeBuffer(
            length: activationBytes, options: .storageModeShared))
        let routedResources = routed.map(\.buffer)
        let offsets = MoEExpertOffsets(
            gateWOff: 0, gateSOff: 0, gateBOff: 0,
            upWOff: 0, upSOff: 0, upBOff: 0,
            downWOff: 0, downSOff: 0, downBOff: 0)
        let hitSlots: [UInt32] = [1, 4]
        let missSlots: [UInt32] = [8]

        let fullCommand = try #require(context.queue.makeCommandBuffer())
        moe.encodeRoutedPhase1(
            commandBuffer: fullCommand,
            routedArgumentBuffer: argumentBuffer,
            routedResources: routedResources,
            routedOffsets: offsets,
            input: input,
            activations: fullActivations,
            hiddenSize: UInt32(Qwen38MoE.canonicalHiddenSize),
            intermediateSize: UInt32(Qwen38MoE.canonicalIntermediateSize))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.status == .completed)

        let sentinel = Float16(-7)
        splitActivations.contents().bindMemory(to: Float16.self,
            capacity: activationBytes / MemoryLayout<Float16>.stride)
            .update(repeating: sentinel,
                    count: activationBytes / MemoryLayout<Float16>.stride)
        for activeSlots in [hitSlots, missSlots] {
            let splitCommand = try #require(context.queue.makeCommandBuffer())
            moe.encodeRoutedPhase1Subset(
                commandBuffer: splitCommand,
                routedArgumentBuffer: argumentBuffer,
                routedResources: routedResources,
                routedOffsets: offsets,
                input: input,
                activations: splitActivations,
                activeSlots: activeSlots,
                hiddenSize: UInt32(Qwen38MoE.canonicalHiddenSize),
                intermediateSize: UInt32(Qwen38MoE.canonicalIntermediateSize))
            splitCommand.commit()
            splitCommand.waitUntilCompleted()
            #expect(splitCommand.status == .completed)
        }

        let rowCount = Qwen38MoE.canonicalIntermediateSize
        let full = fullActivations.contents().bindMemory(to: Float16.self,
            capacity: Qwen38MoE.topK * rowCount)
        let split = splitActivations.contents().bindMemory(to: Float16.self,
            capacity: Qwen38MoE.topK * rowCount)
        for slot in 0..<Qwen38MoE.topK {
            let expected = (hitSlots + missSlots).map(Int.init).contains(slot)
                ? Array(UnsafeBufferPointer(start: full + slot * rowCount, count: rowCount))
                : Array(repeating: sentinel, count: rowCount)
            #expect(Array(UnsafeBufferPointer(start: split + slot * rowCount,
                                              count: rowCount)) == expected,
                    "slot \(slot) activation row")
        }
    }
}
