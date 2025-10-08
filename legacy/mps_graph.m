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

// Reference test implementation, not part of the library
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device");
            return -1;
        }
        NSLog(@"Metal device: %@", device.name);

        MPSGraph* graph = [[MPSGraph alloc] init];

        NSArray<NSNumber *> *shape = @[@4];
        MPSGraphTensor *tensorA_placeholder = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"A"];
        MPSGraphTensor *tensorB_placeholder = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"B"];

        MPSGraphTensor *tensorC_result = [graph additionWithPrimaryTensor:tensorA_placeholder secondaryTensor:tensorB_placeholder name:@"C"];


        MPSNDArrayDescriptor *desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shape];

        float dataA[4] = {1, 2, 3, 4};
        MPSNDArray *inputA = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
        [inputA writeBytes:dataA strideBytes:nil];
        MPSGraphTensorData *dataA_tensor = [[MPSGraphTensorData alloc] initWithMPSNDArray:inputA];

        float dataB[4] = {5, 6, 7, 8};
        MPSNDArray *inputB = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
        [inputB writeBytes:dataB strideBytes:nil];
        MPSGraphTensorData *dataB_tensor = [[MPSGraphTensorData alloc] initWithMPSNDArray:inputB];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
            tensorA_placeholder: dataA_tensor,
            tensorB_placeholder: dataB_tensor
        };

        // First, define the gradient operations BEFORE running the graph
        NSDictionary<MPSGraphTensor*, MPSGraphTensor*> *gradients = [graph gradientForPrimaryTensor:tensorC_result withTensors:@[tensorA_placeholder, tensorB_placeholder] name:nil];

        // Build the list of target tensors including both forward and gradient tensors
        NSMutableArray<MPSGraphTensor*> *targetTensors = [NSMutableArray arrayWithObject:tensorC_result];
        for (MPSGraphTensor *inputTensor in gradients) {
            MPSGraphTensor *gradTensor = gradients[inputTensor];
            [targetTensors addObject:gradTensor];
        }

        // Now run the graph with all targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:targetTensors targetOperations:nil];

        MPSGraphTensorData *outputC = results[tensorC_result];
        NSLog(@"Running A + B");
        printf("A: "); printMPSGraphTensorData(dataA_tensor);
        printf("B: "); printMPSGraphTensorData(dataB_tensor);
        printf("Result C: "); printMPSGraphTensorData(outputC);

        // loop over the values in the gradients dictionary
        for (MPSGraphTensor *inputTensor in gradients) {
            MPSGraphTensor *gradTensor = gradients[inputTensor];
            MPSGraphTensorData *gradData = results[gradTensor];
            if (gradData) {
                printf("Gradient w.r.t. tensor %p: ", inputTensor);
                printMPSGraphTensorData(gradData);
            } else {
                printf("No gradient computed for tensor %p\n", inputTensor);
            }
        }
    }
}


// DataHandle create_data(const float* data, const long* shape, int rank) {
//     id<MTLDevice> device = get_device();
    
//     NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray arrayWithCapacity:rank];
//     size_t total_elements = 1;
//     for (int i = 0; i < rank; ++i) {
//         [shapeArray addObject:@(shape[i])];
//         total_elements *= shape[i];
//     }
    
//     size_t buffer_size = total_elements * sizeof(float);
//     id<MTLBuffer> buffer = [device newBufferWithBytes:data length:buffer_size options:MTLResourceStorageModeShared];
    
//     MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shapeArray];
//     MPSNDArray* ndarray = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
//     [ndarray writeBytes:(void*)data strideBytes:nil];
    
//     struct MPSDataHandle* handle = malloc(sizeof(*handle));
//     handle->ndarray = ndarray;
//     return handle;
// }

// void release_data(DataHandle handle) {
//     if (handle) {
//         handle->ndarray = nil;
//         free(handle);
//     }
// }

// void read_data(DataHandle handle, float* buffer, size_t num_elements) {
//     if (!handle) return;
//     [handle->ndarray readBytes:buffer strideBytes:nil];
// }


// // --- Graph Building Implementation ---
// TensorHandle placeholder(GraphHandle graph_handle, const long* shape, int rank) {
//     NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray arrayWithCapacity:rank];
//     for (int i = 0; i < rank; ++i) {
//         [shapeArray addObject:@(shape[i])];
//     }
    
//     MPSGraphTensor* tensor = [graph_handle->graph placeholderWithShape:shapeArray dataType:MPSDataTypeFloat32 name:nil];
    
//     struct MPSTensorHandle* handle = malloc(sizeof(*handle));
//     handle->tensor = tensor;
//     return handle;
// }

// TensorHandle add(GraphHandle graph_handle, TensorHandle a, TensorHandle b) {
//     MPSGraphTensor* result_tensor = [graph_handle->graph additionWithPrimaryTensor:a->tensor
//                                                                   secondaryTensor:b->tensor
//                                                                              name:nil];
//     struct MPSTensorHandle* handle = malloc(sizeof(*handle));
//     handle->tensor = result_tensor;
//     return handle;
// }

// // ... Implement multiply, relu, etc. similarly ...


// // --- Execution Implementation ---
// // In mps_engine.m

// void run_graph(GraphHandle graph_handle,
//                const GraphFeed* feeds, int num_feeds,
//                TensorHandle* targets, int num_targets,
//                DataHandle* output_data,
//                DataHandle* gradient_data) {
    
//     // This function now assumes we're calculating the gradient of the *first target*
//     // with respect to all the fed inputs. This is a common autograd pattern (scalar loss).
//     if (num_targets == 0) return; // Nothing to do

//     // --- Graph Building: Define Gradient Operations ---

//     // 1. Prepare tensors for which we want gradients (usually the inputs/weights).
//     NSMutableArray<MPSGraphTensor*>* gradient_wrt_tensors = [NSMutableArray arrayWithCapacity:num_feeds];
//     for (int i = 0; i < num_feeds; ++i) {
//         [gradient_wrt_tensors addObject:feeds[i].placeholder->tensor];
//     }

//     // 2. THIS IS THE KEY CHANGE: Add gradient ops to the graph.
//     // This returns new symbolic tensors for the gradients.
//     // Note: The method is for a *single* primary tensor.
//     MPSGraphTensor* primary_target_tensor = targets[0]->tensor;
//     NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* gradient_graph_tensors =
//     [graph_handle->graph gradientForPrimaryTensor:primary_target_tensor
//                              withTensors:gradient_wrt_tensors
//                    name:nil];

//     // --- Execution Phase ---
    
//     // 3. Combine forward targets and gradient tensors into one list for a single run.
//     NSMutableArray<MPSGraphTensor*>* all_targets_array = [NSMutableArray arrayWithCapacity:(num_targets + num_feeds)];
//     // Add original forward-pass targets
//     for (int i = 0; i < num_targets; ++i) {
//         [all_targets_array addObject:targets[i]->tensor];
//     }
//     // Add the new symbolic gradient tensors
//     for (int i = 0; i < num_feeds; ++i) {
//         MPSGraphTensor* input_tensor = feeds[i].placeholder->tensor;
//         MPSGraphTensor* grad_tensor = gradient_graph_tensors[input_tensor];
//         if (grad_tensor) {
//             [all_targets_array addObject:grad_tensor];
//         }
//     }

//     // 4. Prepare feeds dictionary (same as before)
//     NSMutableDictionary<MPSGraphTensor*, MPSNDArray*>* feeds_dict = [NSMutableDictionary dictionaryWithCapacity:num_feeds];
//     for (int i = 0; i < num_feeds; ++i) {
//         feeds_dict[feeds[i].placeholder->tensor] = feeds[i].data->ndarray;
//     }

//     // 5. Execute the graph to get ALL results at once.
//     // This is the corrected, synchronous run call.
//     id<MTLCommandQueue> queue = [get_device() newCommandQueue];
//     NSDictionary<MPSGraphTensor*, MPSNDArray*>* results =
//         [graph_handle->graph runWithMTLCommandQueue:queue
//                                               feeds:feeds_dict
//                                       targetTensors:all_targets_array
//                                    targetOperations:nil];

//     // 6. Populate output handles from the results dictionary.
//     // Forward pass results
//     for (int i = 0; i < num_targets; ++i) {
//         MPSNDArray* result_ndarray = results[targets[i]->tensor];
//         struct MPSDataHandle* handle = malloc(sizeof(*handle));
//         handle->ndarray = result_ndarray;
//         output_data[i] = handle;
//     }

//     // Backward pass gradients
//     for (int i = 0; i < num_feeds; ++i) {
//         MPSGraphTensor* input_tensor = feeds[i].placeholder->tensor;
//         MPSGraphTensor* grad_tensor = gradient_graph_tensors[input_tensor];
//         MPSNDArray* grad_ndarray = grad_tensor ? results[grad_tensor] : nil;
        
//         struct MPSDataHandle* handle = malloc(sizeof(*handle));
//         handle->ndarray = grad_ndarray; // Will be nil if no gradient was computed
//         gradient_data[i] = handle;
//     }
// }

// DataHandle create_data(const float *data, const long *shape, int rank);
    // void release_data(DataHandle data);
    // void read_data(DataHandle data, float *buffer, size_t num_elements);

    // TensorHandle placeholder(GraphHandle graph, const long *shape, int rank);
    // TensorHandle add(GraphHandle graph, TensorHandle a, TensorHandle b);
    // TensorHandle multiply(GraphHandle graph, TensorHandle a, TensorHandle b);
    // TensorHandle relu(GraphHandle graph, TensorHandle a);

    // typedef struct
    // {
    //     TensorHandle placeholder;
    //     DataHandle data;
    // } GraphFeed;

    // void run_graph(GraphHandle graph,
    //                const GraphFeed *feeds, int num_feeds,
    //                TensorHandle *targets, int num_targets,
    //                DataHandle *output_data,
    //                DataHandle *gradient_data);
