#include "mps.h"

@import Foundation;
@import Metal;
@import MetalPerformanceShaders;
#include <string.h>

MTLDevice_t mtl_create_system_default_device_c(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            NSLog(@"[ARC] Metal device created: %@", device.name);
            return (__bridge_retained void *)device;
        } else {
            NSLog(@"[ARC] Failed to create Metal device.");
            return NULL;
        }
    }
}

void mtl_release_device_c(MTLDevice_t device_handle) {
    if (device_handle == NULL) return;
    id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)device_handle;
    NSLog(@"[ARC] Released Metal device: %@", device.name);
    (void)device;
}

const char* mtl_get_device_name_c(MTLDevice_t device_handle) {
    if (device_handle == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
        return strdup([device.name UTF8String]);
    }
}

MTLCommandQueue_t mtl_make_command_queue_c(MTLDevice_t device_handle) {
    if (device_handle == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        return (__bridge_retained void *)commandQueue;
    }
}

MTLCommandBuffer_t mtl_make_command_buffer_c(MTLCommandQueue_t command_queue_handle) {
    if (command_queue_handle == NULL) return NULL;
    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)command_queue_handle;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        return (__bridge_retained void *)commandBuffer;
    }
}

void mtl_commit_command_buffer_c(MTLCommandBuffer_t command_buffer_handle) {
    if (command_buffer_handle == NULL) return;
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)command_buffer_handle;
        [commandBuffer commit];
    }
}

void mtl_wait_until_completed_c(MTLCommandBuffer_t command_buffer_handle) {
    if (command_buffer_handle == NULL) return;
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)command_buffer_handle;
        [commandBuffer waitUntilCompleted];
    }
}

MTLBuffer_t mtl_make_buffer_with_length_c(MTLDevice_t device_handle, unsigned long length) {
    if (device_handle == NULL || length == 0) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        return (__bridge_retained void *)buffer;
    }
}

MTLBuffer_t mtl_make_buffer_with_bytes_c(MTLDevice_t device_handle, const void *bytes, unsigned long length) {
    if (device_handle == NULL || bytes == NULL || length == 0) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
        id<MTLBuffer> buffer = [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
        return (__bridge_retained void *)buffer;
    }
}

void mtl_get_buffer_contents_c(MTLBuffer_t buffer_handle, void *dst, unsigned long length) {
    if (buffer_handle == NULL || dst == NULL || length == 0) return;
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_handle;
        void *src = [buffer contents];
        memcpy(dst, src, length);
    }
}

MPSMatrixMultiplication_t mps_alloc_matrix_multiplication_c(MTLDevice_t device_handle,
                                                            bool transposeLeft,
                                                            bool transposeRight,
                                                            unsigned long resultRows,
                                                            unsigned long resultColumns,
                                                            unsigned long interiorColumns,
                                                            float alpha,
                                                            float beta) {
    if (device_handle == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
        MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
                                              initWithDevice:device transposeLeft:transposeLeft transposeRight:transposeRight
                                                  resultRows:resultRows resultColumns:resultColumns interiorColumns:interiorColumns
                                                       alpha:alpha beta:beta];
        return (__bridge_retained void *)matMul;
    }
}

void mps_encode_matrix_multiplication_to_command_buffer_c(MPSMatrixMultiplication_t matMul_handle,
                                                            MTLCommandBuffer_t commandBuffer_handle,
                                                            MPSMatrix_t leftMatrix_handle,
                                                            MPSMatrix_t rightMatrix_handle,
                                                            MPSMatrix_t resultMatrix_handle) {
    if (matMul_handle == NULL || commandBuffer_handle == NULL || leftMatrix_handle == NULL || rightMatrix_handle == NULL || resultMatrix_handle == NULL) return;
    @autoreleasepool {
        MPSMatrixMultiplication *matMul = (__bridge MPSMatrixMultiplication *)matMul_handle;
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer_handle;
        MPSMatrix *leftMatrix = (__bridge MPSMatrix *)leftMatrix_handle;
        MPSMatrix *rightMatrix = (__bridge MPSMatrix *)rightMatrix_handle;
        MPSMatrix *resultMatrix = (__bridge MPSMatrix *)resultMatrix_handle;

        [matMul encodeToCommandBuffer:commandBuffer leftMatrix:leftMatrix rightMatrix:rightMatrix resultMatrix:resultMatrix];
    }
}

MPSMatrixDescriptor_t mps_create_matrix_descriptor_c(unsigned long rows, unsigned long columns, unsigned long rowBytes) {
    @autoreleasepool {
        MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:rowBytes dataType:MPSDataTypeFloat32];
        return (__bridge_retained void *)descriptor;
    }
}

MPSMatrix_t mps_create_matrix_c(MTLBuffer_t buffer_handle, MPSMatrixDescriptor_t descriptor_handle) {
    if (buffer_handle == NULL || descriptor_handle == NULL) return NULL;
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_handle;
        MPSMatrixDescriptor *descriptor = (__bridge MPSMatrixDescriptor *)descriptor_handle;
        MPSMatrix *matrix = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
        return (__bridge_retained void *)matrix;
    }
}

// void perform_matmul_on_buffers(MTLDevice_t device_handle, MTLCommandQueue_t command_queue_handle, MTLBuffer_t bufferA_handle, MTLBuffer_t bufferB_handle, MTLBuffer_t bufferC_handle, int m, int n, int k) {
//     @autoreleasepool {
//         if (command_queue_handle == NULL) {
//             NSLog(@"Invalid Metal command queue.");
//             return;
//         }

//         if (device_handle == NULL) {
//             NSLog(@"Invalid Metal device.");
//             return;
//         }

//         if (bufferA_handle == NULL || bufferB_handle == NULL || bufferC_handle == NULL) {
//             NSLog(@"Invalid Metal buffers.");
//             return;
//         }

//         id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)device_handle;
//         id<MTLCommandQueue> commandQueue = (__bridge_transfer id<MTLCommandQueue>)command_queue_handle;

//         int m = 2, n = 3, k = 4;

//         float matrixAData[] = {1, 2, 3, 4, 5, 6};
//         float matrixBData[] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
//         float resultC[m*k]; // 2 * 4 = 8

//         NSUInteger bufferASize = m * n * sizeof(float);
//         NSUInteger bufferBSize = n * k * sizeof(float);
//         NSUInteger bufferCSize = m * k * sizeof(float);

//         id<MTLBuffer> bufferA = (__bridge_transfer id<MTLBuffer>)bufferA_handle;
//         id<MTLBuffer> bufferB = (__bridge_transfer id<MTLBuffer>)bufferB_handle;
//         id<MTLBuffer> bufferC = (__bridge_transfer id<MTLBuffer>)bufferC_handle;

//         MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
//                                               initWithDevice:device transposeLeft:false transposeRight:false
//                                                   resultRows:m resultColumns:k interiorColumns:n
//                                                        alpha:1.0 beta:0.0];

//         id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

//         MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
//         MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
//         MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];

//         MPSMatrix *mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
//         MPSMatrix *mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
//         MPSMatrix *mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

//         [matMul encodeToCommandBuffer:commandBuffer leftMatrix:mpsMatrixA rightMatrix:mpsMatrixB resultMatrix:mpsMatrixC];

//         [commandBuffer commit];
//         [commandBuffer waitUntilCompleted];

//         void *resultPtr = [bufferC contents];        
//         memcpy(resultC, resultPtr, bufferCSize);

//         NSLog(@"Result Matrix C:");
//         for (int i = 0; i < m; ++i) {
//             NSMutableString *rowString = [NSMutableString string];
//             for (int j = 0; j < k; ++j) {
//                 [rowString appendFormat:@"%f ", resultC[i * k + j]];
//             }
//             NSLog(@"%@", rowString);
//         }
//     }
// }

void perform_matmul_on_command_queue(MTLDevice_t device_handle, MTLCommandQueue_t command_queue_handle) {
    @autoreleasepool {
        if (command_queue_handle == NULL) {
            NSLog(@"Invalid Metal command queue.");
            return;
        }

        if (device_handle == NULL) {
            NSLog(@"Invalid Metal device.");
            return;
        }

        id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)device_handle;
        id<MTLCommandQueue> commandQueue = (__bridge_transfer id<MTLCommandQueue>)command_queue_handle;

        int m = 2, n = 3, k = 4;

        float matrixAData[] = {1, 2, 3, 4, 5, 6};
        float matrixBData[] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
        float resultC[m*k]; // 2 * 4 = 8

        NSUInteger bufferASize = m * n * sizeof(float);
        NSUInteger bufferBSize = n * k * sizeof(float);
        NSUInteger bufferCSize = m * k * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:matrixAData length:bufferASize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:matrixBData length:bufferBSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:bufferCSize options:MTLResourceStorageModeShared];

        MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
                                              initWithDevice:device transposeLeft:false transposeRight:false
                                                  resultRows:m resultColumns:k interiorColumns:n
                                                       alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix *mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        [matMul encodeToCommandBuffer:commandBuffer leftMatrix:mpsMatrixA rightMatrix:mpsMatrixB resultMatrix:mpsMatrixC];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        void *resultPtr = [bufferC contents];        
        memcpy(resultC, resultPtr, bufferCSize);

        NSLog(@"Result Matrix C:");
        for (int i = 0; i < m; ++i) {
            NSMutableString *rowString = [NSMutableString string];
            for (int j = 0; j < k; ++j) {
                [rowString appendFormat:@"%f ", resultC[i * k + j]];
            }
            NSLog(@"%@", rowString);
        }
    }
}

void perform_matmul_with_device_c(MTLDevice_t device_handle) {
    @autoreleasepool {
        if (device_handle == NULL) {
            NSLog(@"Invalid Metal device.");
            return;
        }

        id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)device_handle;

        int m = 2, n = 3, k = 4;

        float matrixAData[] = {1, 2, 3, 4, 5, 6};
        float matrixBData[] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
        float resultC[m*k]; // 2 * 4 = 8

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        NSUInteger bufferASize = m * n * sizeof(float);
        NSUInteger bufferBSize = n * k * sizeof(float);
        NSUInteger bufferCSize = m * k * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:matrixAData length:bufferASize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:matrixBData length:bufferBSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:bufferCSize options:MTLResourceStorageModeShared];

        MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
                                              initWithDevice:device transposeLeft:false transposeRight:false
                                                  resultRows:m resultColumns:k interiorColumns:n
                                                       alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix *mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        [matMul encodeToCommandBuffer:commandBuffer leftMatrix:mpsMatrixA rightMatrix:mpsMatrixB resultMatrix:mpsMatrixC];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        void *resultPtr = [bufferC contents];        
        memcpy(resultC, resultPtr, bufferCSize);

        NSLog(@"Result Matrix C:");
        for (int i = 0; i < m; ++i) {
            NSMutableString *rowString = [NSMutableString string];
            for (int j = 0; j < k; ++j) {
                [rowString appendFormat:@"%f ", resultC[i * k + j]];
            }
            NSLog(@"%@", rowString);
        }
    }
}

// #import <Foundation/Foundation.h>
// #import <Metal/Metal.h>
// #import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// #include <string.h>
// #include <stdio.h>

// #include "mps.h"

// #ifdef __cplusplus
// extern "C" {
// #endif

// void perform_matmul(const float* matrixAData, const float* matrixBData, float* resultC, int m, int n, int k) {
//     @autoreleasepool {
//         id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//         id<MTLCommandQueue> commandQueue = [device newCommandQueue];

//         NSUInteger bufferASize = m * n * sizeof(float);
//         NSUInteger bufferBSize = n * k * sizeof(float);
//         NSUInteger bufferCSize = m * k * sizeof(float);

//         id<MTLBuffer> bufferA = [device newBufferWithBytes:matrixAData length:bufferASize options:MTLResourceStorageModeShared];
//         id<MTLBuffer> bufferB = [device newBufferWithBytes:matrixBData length:bufferBSize options:MTLResourceStorageModeShared];
//         id<MTLBuffer> bufferC = [device newBufferWithLength:bufferCSize options:MTLResourceStorageModeShared];

//         MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
//                                               initWithDevice:device transposeLeft:false transposeRight:false
//                                                   resultRows:m resultColumns:k interiorColumns:n
//                                                        alpha:1.0 beta:0.0];

//         id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

//         MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
//         MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
//         MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];

//         MPSMatrix *mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
//         MPSMatrix *mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
//         MPSMatrix *mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

//         [matMul encodeToCommandBuffer:commandBuffer leftMatrix:mpsMatrixA rightMatrix:mpsMatrixB resultMatrix:mpsMatrixC];

//         [commandBuffer commit];
//         [commandBuffer waitUntilCompleted];

//         void *resultPtr = [bufferC contents];
//         memcpy(resultC, resultPtr, bufferCSize);
//     }
// }

// MTLDevice_t mtl_create_system_default_device(void) {
//     @autoreleasepool {
//         id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//         if (device) {
//             NSLog(@"Metal device created: %@", device.name);
//         } else {
//             NSLog(@"Failed to create Metal device.");
//         }
//         return (__bridge_retained void *)device;
//     }
// }

// void perform_matmul_with_device(id<MTLDevice> device) {
//     @autoreleasepool {
//         if (!device) {
//             NSLog(@"Invalid Metal device.");
//             return;
//         }

//         int m = 2, n = 3, k = 4;

//         float matrixAData[] = {1, 2, 3, 4, 5, 6};
//         float matrixBData[] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
//         float resultC[m*k]; // 2 * 4 = 8

//         id<MTLCommandQueue> commandQueue = [device newCommandQueue];

//         NSUInteger bufferASize = m * n * sizeof(float);
//         NSUInteger bufferBSize = n * k * sizeof(float);
//         NSUInteger bufferCSize = m * k * sizeof(float);

//         id<MTLBuffer> bufferA = [device newBufferWithBytes:matrixAData length:bufferASize options:MTLResourceStorageModeShared];
//         id<MTLBuffer> bufferB = [device newBufferWithBytes:matrixBData length:bufferBSize options:MTLResourceStorageModeShared];
//         id<MTLBuffer> bufferC = [device newBufferWithLength:bufferCSize options:MTLResourceStorageModeShared];

//         MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc]
//                                               initWithDevice:device transposeLeft:false transposeRight:false
//                                                   resultRows:m resultColumns:k interiorColumns:n
//                                                        alpha:1.0 beta:0.0];

//         id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

//         MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
//         MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
//         MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];

//         MPSMatrix *mpsMatrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
//         MPSMatrix *mpsMatrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
//         MPSMatrix *mpsMatrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

//         [matMul encodeToCommandBuffer:commandBuffer leftMatrix:mpsMatrixA rightMatrix:mpsMatrixB resultMatrix:mpsMatrixC];

//         [commandBuffer commit];
//         [commandBuffer waitUntilCompleted];

//         void *resultPtr = [bufferC contents];
//         memcpy(resultC, resultPtr, bufferCSize);
//     }
// }


// #ifdef __cplusplus
// }
// #endif

// void do_the_matmul() {
//     int m = 2, n = 3, k = 4;

//     float matrixA[] = {1, 2, 3, 4, 5, 6};
//     float matrixB[] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
//     float resultC[m*k]; // 2 * 4 = 8

//     perform_matmul(matrixA, matrixB, resultC, m, n, k);

//     printf("Result Matrix C:\n");
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < k; ++j) {
//             printf("%f ", resultC[i * k + j]);
//         }
//         printf("\n");
//     }
// }
