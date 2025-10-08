#ifndef MPS_ENGINE_H
#define MPS_ENGINE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct MPSGraphHandle *GraphHandle;
    typedef struct MPSTensorHandle *TensorHandle;
    typedef struct MPSTensorDataHandle *TensorDataHandle;

    GraphHandle mps_create_graph_c(void);
    void mps_release_graph_c(GraphHandle graph);

    TensorHandle mps_graph_placeholder_c(GraphHandle graph, const int *shape, int rank);
    TensorHandle mps_graph_attach_addition_c(GraphHandle graph, TensorHandle a, TensorHandle b);
    TensorHandle mps_graph_attach_multiplication_c(GraphHandle graph, TensorHandle a, TensorHandle b);

    void mps_release_tensor_c(TensorHandle tensor_handle);

    TensorDataHandle mps_tensor_data_from_float_array_c(const float *data, const int *shape, int rank);
    float *mps_tensor_data_to_float_array_c(TensorDataHandle data);
    void mps_release_tensor_data_c(TensorDataHandle data);

    void mps_graph_run_forward_backward_with_feeds_c(
        GraphHandle graph,
        const TensorHandle *feed_tensors,
        const TensorDataHandle *feed_data,
        int num_feeds,
        TensorHandle output_tensor,
        TensorDataHandle output_data);

#ifdef __cplusplus
}
#endif
#endif // MPS_ENGINE_H
