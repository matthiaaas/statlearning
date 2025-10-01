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

