open! Ctypes
module Types = Types_generated

module Functions (F : Ctypes.FOREIGN) = struct
  open F

  let mps_create_graph =
    foreign "mps_create_graph_c" (void @-> returning Types.Graph_handle.t)

  let mps_release_graph =
    foreign "mps_release_graph_c" (Types.Graph_handle.t @-> returning void)

  let mps_graph_placeholder =
    foreign "mps_graph_placeholder_c"
      (Types.Graph_handle.t @-> ptr int @-> int
      @-> returning Types.Tensor_handle.t)

  let mps_graph_attach_addition =
    foreign "mps_graph_attach_addition_c"
      (Types.Graph_handle.t @-> Types.Tensor_handle.t @-> Types.Tensor_handle.t
      @-> returning Types.Tensor_handle.t)

  let mps_graph_attach_multiplication =
    foreign "mps_graph_attach_multiplication_c"
      (Types.Graph_handle.t @-> Types.Tensor_handle.t @-> Types.Tensor_handle.t
      @-> returning Types.Tensor_handle.t)

  let mps_release_tensor =
    foreign "mps_release_tensor_c" (Types.Tensor_handle.t @-> returning void)

  let mps_tensor_data_from_float_array =
    foreign "mps_tensor_data_from_float_array_c"
      (ptr float @-> ptr int @-> int @-> returning Types.Tensor_data_handle.t)

  let mps_tensor_data_to_float_array =
    foreign "mps_tensor_data_to_float_array_c"
      (Types.Tensor_data_handle.t @-> returning (ptr float))

  let mps_release_tensor_data =
    foreign "mps_release_tensor_data_c"
      (Types.Tensor_data_handle.t @-> returning void)

  let mps_graph_run_forward_backward_with_feeds =
    foreign "mps_graph_run_forward_backward_with_feeds_c"
      (Types.Graph_handle.t @-> ptr Types.Tensor_handle.t
      @-> ptr Types.Tensor_data_handle.t
      @-> int @-> Types.Tensor_handle.t @-> Types.Tensor_data_handle.t
      @-> returning void)
end
