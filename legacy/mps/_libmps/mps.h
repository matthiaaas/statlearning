#ifndef MPS_H
#define MPS_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *MTLDevice_t;
    MTLDevice_t mtl_create_system_default_device_c(void);
    void mtl_release_device_c(MTLDevice_t device_handle);
    const char *mtl_get_device_name_c(MTLDevice_t device_handle);

    typedef void *MTLCommandQueue_t;
    MTLCommandQueue_t mtl_make_command_queue_c(MTLDevice_t device_handle);

    typedef void *MTLCommandBuffer_t;
    MTLCommandBuffer_t mtl_make_command_buffer_c(MTLCommandQueue_t command_queue_handle);
    void mtl_commit_command_buffer_c(MTLCommandBuffer_t command_buffer_handle);
    void mtl_wait_until_completed_c(MTLCommandBuffer_t command_buffer_handle);

    typedef void *MTLBuffer_t;
    MTLBuffer_t mtl_make_buffer_with_length_c(MTLDevice_t device_handle, unsigned long length);
    MTLBuffer_t mtl_make_buffer_with_bytes_c(MTLDevice_t device_handle, const void *bytes, unsigned long length);
    void mtl_get_buffer_contents_c(MTLBuffer_t buffer_handle, void *resultC, unsigned long bufferCSize);

    typedef void *MPSMatrixMultiplication_t;
    MPSMatrixMultiplication_t mps_alloc_matrix_multiplication_c(MTLDevice_t device_handle,
                                                                bool transposeLeft,
                                                                bool transposeRight,
                                                                unsigned long rows,
                                                                unsigned long columns,
                                                                unsigned long innerDimension,
                                                                float alpha,
                                                                float beta);
    void mps_encode_matrix_multiplication_to_command_buffer_c(MPSMatrixMultiplication_t kernel_handle,
                                                              MTLCommandQueue_t command_queue_handle,
                                                              MTLBuffer_t left_matrix_handle,
                                                              MTLBuffer_t right_matrix_handle,
                                                              MTLBuffer_t result_matrix_handle);

    typedef void *MPSMatrixDescriptor_t;
    MPSMatrixDescriptor_t mps_create_matrix_descriptor_c(unsigned long rows, unsigned long columns, unsigned long rowBytes);

    typedef void *MPSMatrix_t;
    MPSMatrix_t mps_create_matrix_c(MTLBuffer_t buffer_handle, MPSMatrixDescriptor_t descriptor_handle);

#ifdef __cplusplus
}
#endif

#endif
