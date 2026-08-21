import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QwenElementwiseTests {
    @Test func deltaParametersMatchReference() throws {
        let context = try MetalContext()
        let kernels = try QwenElementwise(context: context)
        let a = try #require(Fp16Buffer.make(context.device, values: [0.2, -0.4]))
        let betaInput = try #require(Fp16Buffer.make(context.device, values: [0.7, -0.3]))
        let aLog = try #require(context.device.makeBuffer(
            bytes: [-1.0 as Float, -0.5],
            length: 2 * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let dtBias = try #require(context.device.makeBuffer(
            bytes: [0.1 as Float, -0.2],
            length: 2 * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let decay = try #require(context.device.makeBuffer(
            length: 2 * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let beta = try #require(context.device.makeBuffer(
            length: 2 * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernels.encodeDeltaParameters(commandBuffer: commandBuffer,
                                      a: a,
                                      betaInput: betaInput,
                                      aLog: aLog,
                                      dtBias: dtBias,
                                      decay: decay,
                                      beta: beta,
                                      count: 2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actualDecay = decay.contents().assumingMemoryBound(to: Float.self)
        let actualBeta = beta.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<2 {
            let aValue = [0.2, -0.4][index]
            let expectedDecay = -Float(Foundation.exp(Double([-1.0, -0.5][index])))
                * Float(Foundation.log1p(Foundation.exp(
                    Double(aValue + [0.1, -0.2][index]))))
            let expectedBeta = Float(1 / (1 + Foundation.exp(
                -Double([0.7, -0.3][index]))))
            #expect(abs(actualDecay[index] - expectedDecay) < 0.001)
            #expect(abs(actualBeta[index] - expectedBeta) < 0.001)
        }
    }

    @Test func gatedNormUsesResidentWeightOffsetAndResidualAdds() throws {
        let context = try MetalContext()
        let kernels = try QwenElementwise(context: context)
        let inputValues: [Float] = [1, -2, 0.5, 3, -1, 2]
        let gateValues: [Float] = [0.2, -0.4, 0.8, -0.3, 0.6, -0.7]
        let input = try #require(Fp16Buffer.make(context.device, values: inputValues))
        let gate = try #require(Fp16Buffer.make(context.device, values: gateValues))
        let residentWeights = try #require(
            Fp16Buffer.make(context.device, values: [99, 1, 2, 3]))
        let normalized = try #require(Fp16Buffer.make(context.device, count: 6))
        let lhs = try #require(Fp16Buffer.make(context.device, values: [1, 2, 3, 4, 5, 6]))
        let rhs = try #require(Fp16Buffer.make(context.device, values: [0.5, 1, 1.5, 2, 2.5, 3]))
        let residual = try #require(Fp16Buffer.make(context.device, count: 6))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernels.encodeGatedNorm(commandBuffer: commandBuffer,
                                 input: input,
                                 gate: gate,
                                 weight: residentWeights,
                                 weightOffset: MemoryLayout<Float16>.stride,
                                 output: normalized,
                                 headCount: 2,
                                 headDimension: 3,
                                 epsilon: 1e-6)
        kernels.encodeResidualAdd(commandBuffer: commandBuffer,
                                  lhs: lhs,
                                  rhs: rhs,
                                  output: residual,
                                  count: 6)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actualNorm = Fp16Buffer.read(normalized, count: 6)
        for head in 0..<2 {
            let start = head * 3
            let sum = (0..<3).reduce(Float(0)) {
                $0 + inputValues[start + $1] * inputValues[start + $1]
            }
            let inverse = 1 / sqrt(sum / 3 + 1e-6)
            for offset in 0..<3 {
                let index = start + offset
                let silu = gateValues[index] / (1 + exp(-gateValues[index]))
                let expected = inputValues[index] * inverse
                    * [1, 2, 3][offset] * silu
                #expect(abs(actualNorm[index] - Float(Float16(expected))) < 0.01)
            }
        }
        let actualResidual = Fp16Buffer.read(residual, count: 6)
        #expect(actualResidual == [1.5, 3, 4.5, 6, 7.5, 9].map { Float(Float16($0)) })
    }
}