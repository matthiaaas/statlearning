#include "mps_engine.h"

@import Foundation;
@import Metal;
@import MetalPerformanceShaders;
@import MetalPerformanceShadersGraph;

static id<MTLDevice> get_device() {
    static id<MTLDevice> device = nil;
    if (!device) {
        device = MTLCreateSystemDefaultDevice();
    }
    return device;
}

struct MPSGraphHandle {
    MPSGraph* graph;
};
struct MPSTensorHandle {
    MPSGraphTensor* tensor;
};
struct MPSTensorDataHandle {
    MPSGraphTensorData* data;
};


NSMutableArray<NSNumber*>* collect_shape_of_rank_from_c(const int* shape, int rank) {
    NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray arrayWithCapacity:rank];
    for (int i = 0; i < rank; ++i) {
        [shapeArray addObject:@(shape[i])];
    }
    return shapeArray;
}

NSUInteger compute_total_elements_from_shape(const int* shape, int rank) {
    NSUInteger totalElements = 1;
    for (int i = 0; i < rank; ++i) {
        totalElements *= shape[i];
    }
    return totalElements;
}

GraphHandle mps_create_graph_c(void) {
    MPSGraph* graph = [[MPSGraph alloc] init];
    struct MPSGraphHandle* handle = malloc(sizeof(*handle));
    handle->graph = graph;
    return handle;
}

void mps_release_graph_c(GraphHandle handle) {
    if (handle) {
        handle->graph = nil;
        free(handle);
    }
}

TensorHandle mps_graph_placeholder_c(GraphHandle graph_handle, const int* shape, int rank) {
    NSMutableArray<NSNumber*>* shapeArray = collect_shape_of_rank_from_c(shape, rank);

    MPSGraphTensor* tensor = [graph_handle->graph placeholderWithShape:shapeArray dataType:MPSDataTypeFloat32 name:nil];

    struct MPSTensorHandle* handle = malloc(sizeof(*handle));
    handle->tensor = tensor;
    return handle;
}

TensorHandle mps_graph_attach_addition_c(GraphHandle graph_handle, TensorHandle a, TensorHandle b) {
    MPSGraphTensor* result_tensor = [graph_handle->graph additionWithPrimaryTensor:a->tensor
                                                                  secondaryTensor:b->tensor
                                                                             name:nil];
    struct MPSTensorHandle* handle = malloc(sizeof(*handle));
    handle->tensor = result_tensor;
    return handle;
}

void mps_release_tensor_c(TensorHandle tensor_handle) {
    if (tensor_handle) {
        tensor_handle->tensor = nil;
        free(tensor_handle);
    }
}

TensorDataHandle mps_tensor_data_from_float_array_c(const float* data, const int* shape, int rank) {
    NSMutableArray<NSNumber*>* shapeArray = collect_shape_of_rank_from_c(shape, rank);
    
    MPSNDArrayDescriptor *desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shapeArray];
    MPSNDArray *ndArray = [[MPSNDArray alloc] initWithDevice:get_device() descriptor:desc];
    [ndArray writeBytes:(void *)data strideBytes:NULL];
    MPSGraphTensorData *tensorData = [[MPSGraphTensorData alloc] initWithMPSNDArray:ndArray];

    struct MPSTensorDataHandle* handle = malloc(sizeof(*handle));
    handle->data = tensorData;
    return handle;
}

void mps_release_tensor_data_c(TensorDataHandle data_handle) {
    if (data_handle) {
        data_handle->data = nil;
        free(data_handle);
    }
}

void printMPSGraphTensorData(MPSGraphTensorData *tensorData) {
    if (!tensorData) {
        NSLog(@"Tensor data is nil");
        return;
    }
    
    MPSNDArray *ndarray = tensorData.mpsndarray;
    if (!ndarray) {
        NSLog(@"MPSNDArray is nil");
        return;
    }

    NSUInteger numberOfDimensions = ndarray.descriptor.numberOfDimensions;
    NSUInteger elementCount = 1;
    for (NSUInteger i = 0; i < numberOfDimensions; i++) {
        NSUInteger dimensionLength = [ndarray.descriptor lengthOfDimension:i];
        elementCount *= dimensionLength;
    }

    float *data = malloc(elementCount * sizeof(float));
    [ndarray readBytes:data strideBytes:nil];

    NSLog(@"Tensor data (total elements: %lu):", (unsigned long)elementCount);
    for (NSUInteger i = 0; i < elementCount; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
    free(data);
}

void mps_graph_run_forward_backward_with_feeds_c(GraphHandle graph, const TensorHandle *feed_tensors, const TensorDataHandle *feed_data, int num_feeds, TensorHandle output_tensor, TensorDataHandle output_data) {
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionaryWithCapacity:num_feeds];
    for (int i = 0; i < num_feeds; ++i) {
        feeds[feed_tensors[i]->tensor] = feed_data[i]->data;
    }

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* gradients = [graph->graph gradientForPrimaryTensor:output_tensor->tensor withTensors:feeds.allKeys name: nil];

    NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray arrayWithCapacity:1 + gradients.count];
    [targetTensors addObject:output_tensor->tensor];
    [targetTensors addObjectsFromArray:gradients.allValues];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [graph->graph runWithFeeds:feeds targetTensors:targetTensors targetOperations:nil];

    if (results[output_tensor->tensor] && output_data) {
        output_data->data = results[output_tensor->tensor];
        printMPSGraphTensorData(output_data->data);
    } else {
        NSLog(@"No output data found for the output tensor.");
    }

    for (MPSGraphTensor* gradTensor in gradients.allValues) {
        MPSGraphTensorData* gradData = results[gradTensor];
        if (gradData) {
            NSLog(@"Gradient for tensor %@:", gradTensor);
            printMPSGraphTensorData(gradData);
        } else {
            NSLog(@"No gradient data found for tensor %@", gradTensor);
        }
    }
}
