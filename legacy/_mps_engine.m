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

// --- Internal Handle Structs (Casting Targets) ---
struct MPSGraphHandle {
    MPSGraph* graph;
};
struct MPSTensorHandle {
    MPSGraphTensor* tensor;
};
struct MPSDataHandle {
    MPSNDArray* ndarray;
};


// --- Lifecycle Implementation ---
GraphHandle create_graph(void) {
    MPSGraph* graph = [[MPSGraph alloc] init];
    struct MPSGraphHandle* handle = malloc(sizeof(*handle));
    handle->graph = graph; // ARC will manage the object lifetime
    return handle;
}

void release_graph(GraphHandle handle) {
    if (handle) {
        handle->graph = nil; // Releases the Obj-C object under ARC
        free(handle);
    }
}

DataHandle create_data(const float* data, const long* shape, int rank) {
    id<MTLDevice> device = get_device();
    
    NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray arrayWithCapacity:rank];
    size_t total_elements = 1;
    for (int i = 0; i < rank; ++i) {
        [shapeArray addObject:@(shape[i])];
        total_elements *= shape[i];
    }
    
    size_t buffer_size = total_elements * sizeof(float);
    id<MTLBuffer> buffer = [device newBufferWithBytes:data length:buffer_size options:MTLResourceStorageModeShared];
    
    MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shapeArray];
    MPSNDArray* ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
    [ndarray writeBytes:(void*)data strideBytes:nil];
    
    struct MPSDataHandle* handle = malloc(sizeof(*handle));
    handle->ndarray = ndarray;
    return handle;
}

void release_data(DataHandle handle) {
    if (handle) {
        handle->ndarray = nil;
        free(handle);
    }
}

void read_data(DataHandle handle, float* buffer, size_t num_elements) {
    if (!handle) return;
    [handle->ndarray readBytes:buffer strideBytes:nil];
}


// --- Graph Building Implementation ---
TensorHandle placeholder(GraphHandle graph_handle, const long* shape, int rank) {
    NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray arrayWithCapacity:rank];
    for (int i = 0; i < rank; ++i) {
        [shapeArray addObject:@(shape[i])];
    }
    
    MPSGraphTensor* tensor = [graph_handle->graph placeholderWithShape:shapeArray dataType:MPSDataTypeFloat32 name:nil];
    
    struct MPSTensorHandle* handle = malloc(sizeof(*handle));
    handle->tensor = tensor;
    return handle;
}

TensorHandle add(GraphHandle graph_handle, TensorHandle a, TensorHandle b) {
    MPSGraphTensor* result_tensor = [graph_handle->graph additionWithPrimaryTensor:a->tensor
                                                                  secondaryTensor:b->tensor
                                                                             name:nil];
    struct MPSTensorHandle* handle = malloc(sizeof(*handle));
    handle->tensor = result_tensor;
    return handle;
}

// ... Implement multiply, relu, etc. similarly ...


// --- Execution Implementation ---
// In mps_engine.m

void run_graph(GraphHandle graph_handle,
               const GraphFeed* feeds, int num_feeds,
               TensorHandle* targets, int num_targets,
               DataHandle* output_data,
               DataHandle* gradient_data) {
    
    // This function now assumes we're calculating the gradient of the *first target*
    // with respect to all the fed inputs. This is a common autograd pattern (scalar loss).
    if (num_targets == 0) return; // Nothing to do

    // --- Graph Building: Define Gradient Operations ---

    // 1. Prepare tensors for which we want gradients (usually the inputs/weights).
    NSMutableArray<MPSGraphTensor*>* gradient_wrt_tensors = [NSMutableArray arrayWithCapacity:num_feeds];
    for (int i = 0; i < num_feeds; ++i) {
        [gradient_wrt_tensors addObject:feeds[i].placeholder->tensor];
    }

    // 2. THIS IS THE KEY CHANGE: Add gradient ops to the graph.
    // This returns new symbolic tensors for the gradients.
    // Note: The method is for a *single* primary tensor.
    MPSGraphTensor* primary_target_tensor = targets[0]->tensor;
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* gradient_graph_tensors =
    [graph_handle->graph gradientForPrimaryTensor:primary_target_tensor
                             withTensors:gradient_wrt_tensors
                   name:nil];

    // --- Execution Phase ---
    
    // 3. Combine forward targets and gradient tensors into one list for a single run.
    NSMutableArray<MPSGraphTensor*>* all_targets_array = [NSMutableArray arrayWithCapacity:(num_targets + num_feeds)];
    // Add original forward-pass targets
    for (int i = 0; i < num_targets; ++i) {
        [all_targets_array addObject:targets[i]->tensor];
    }
    // Add the new symbolic gradient tensors
    for (int i = 0; i < num_feeds; ++i) {
        MPSGraphTensor* input_tensor = feeds[i].placeholder->tensor;
        MPSGraphTensor* grad_tensor = gradient_graph_tensors[input_tensor];
        if (grad_tensor) {
            [all_targets_array addObject:grad_tensor];
        }
    }

    // 4. Prepare feeds dictionary (same as before)
    NSMutableDictionary<MPSGraphTensor*, MPSNDArray*>* feeds_dict = [NSMutableDictionary dictionaryWithCapacity:num_feeds];
    for (int i = 0; i < num_feeds; ++i) {
        feeds_dict[feeds[i].placeholder->tensor] = feeds[i].data->ndarray;
    }

    // 5. Execute the graph to get ALL results at once.
    // This is the corrected, synchronous run call.
    id<MTLCommandQueue> queue = [get_device() newCommandQueue];
    NSDictionary<MPSGraphTensor*, MPSNDArray*>* results =
        [graph_handle->graph runWithMTLCommandQueue:queue
                                              feeds:feeds_dict
                                      targetTensors:all_targets_array
                                   targetOperations:nil];

    // 6. Populate output handles from the results dictionary.
    // Forward pass results
    for (int i = 0; i < num_targets; ++i) {
        MPSNDArray* result_ndarray = results[targets[i]->tensor];
        struct MPSDataHandle* handle = malloc(sizeof(*handle));
        handle->ndarray = result_ndarray;
        output_data[i] = handle;
    }

    // Backward pass gradients
    for (int i = 0; i < num_feeds; ++i) {
        MPSGraphTensor* input_tensor = feeds[i].placeholder->tensor;
        MPSGraphTensor* grad_tensor = gradient_graph_tensors[input_tensor];
        MPSNDArray* grad_ndarray = grad_tensor ? results[grad_tensor] : nil;
        
        struct MPSDataHandle* handle = malloc(sizeof(*handle));
        handle->ndarray = grad_ndarray; // Will be nil if no gradient was computed
        gradient_data[i] = handle;
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Example usage of the MPS engine
        GraphHandle graph = create_graph();
        
        long shape[2] = {2, 2};
        TensorHandle a = placeholder(graph, shape, 2);
        TensorHandle b = placeholder(graph, shape, 2);
        TensorHandle c = add(graph, a, b);
        
        float data_a[4] = {1, 2, 3, 4};
        float data_b[4] = {5, 6, 7, 8};
        
        DataHandle data_handle_a = create_data(data_a, shape, 2);
        DataHandle data_handle_b = create_data(data_b, shape, 2);
        
        GraphFeed feeds[2] = {
            {a, data_handle_a},
            {b, data_handle_b}
        };
        
        TensorHandle targets[1] = {c};
        DataHandle output_data[1];
        DataHandle gradient_data[2];
        
        run_graph(graph, feeds, 2, targets, 1, output_data, gradient_data);
        
        // float output[4];
        // read_data(output_data[0], output, 4);
        
        // printf("Output:\n");
        // for (int i = 0; i < 4; ++i) {
        //     printf("%f ", output[i]);
        // }
        // printf("\n");
        
        // // Cleanup
        // release_data(data_handle_a);
        // release_data(data_handle_b);
        // release_data(output_data[0]);
        // release_data(gradient_data[0]);
        // release_data(gradient_data[1]);
        
        // free(a);
        // free(b);
        // free(c);
        // release_graph(graph);
    }
    return 0;
}
