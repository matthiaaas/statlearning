import Metal
import MetalPerformanceShaders

func perform() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal is not supported on this device.")
    }

    let commandQueue = device.makeCommandQueue()!

    let m = 2
    let n = 3
    let k = 4

    // m x n
    let A : [Float] = [1, 2, 3,
                       4, 5, 6]

    // n x k
    let B : [Float] = [10, 11, 12, 13,
                       20, 21, 22, 23,
                       30, 31, 32, 33]

    guard let bufferA = device.makeBuffer(bytes:A, length: m * n * MemoryLayout<Float>.size, options: .storageModeShared),
          let bufferB = device.makeBuffer(bytes:B, length: n * k * MemoryLayout<Float>.size, options: .storageModeShared),
          let bufferC = device.makeBuffer(length: m * k * MemoryLayout<Float>.size, options: .storageModeShared) else {
        fatalError("Failed to create buffers.")
    }

    let matrixMultiplication = MPSMatrixMultiplication(
        device: device,
        transposeLeft: false,
        transposeRight: false,
        resultRows: m,
        resultColumns: k,
        interiorColumns: n,
        alpha: 1.0,
        beta: 0.0
    )

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        fatalError("Failed to create command buffer.")
    }

    matrixMultiplication.encode(
        commandBuffer: commandBuffer,
        leftMatrix: MPSMatrix(buffer: bufferA, descriptor: MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * MemoryLayout<Float>.size, dataType: .float32)),
        rightMatrix: MPSMatrix(buffer: bufferB, descriptor: MPSMatrixDescriptor(rows: n, columns: k, rowBytes: k * MemoryLayout<Float>.size, dataType: .float32)),
        resultMatrix: MPSMatrix(buffer: bufferC, descriptor: MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * MemoryLayout<Float>.size, dataType: .float32))
    )

    commandBuffer.commit()

    commandBuffer.waitUntilCompleted()

    let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: m * k)
    let resultBuffer = UnsafeBufferPointer(start: resultPointer, count: m * k)
    let resultArray = Array(resultBuffer)

    print("\nResult Matrix C (2x4):")
    for i in 0..<m {
        print(resultArray[i*k..<(i+1)*k])
    }
}

perform()
