type t = { c_ptr : C.Types.Graph_handle.t }

module Tensor = struct
  type t = C.Types.Tensor_handle.t

  let of_c_ptr c_ptr =
    Gc.finalise C.Functions.mps_release_tensor c_ptr;
    c_ptr
end

module Tensor_data = struct
  type t = { c_ptr : C.Types.Tensor_data_handle.t }

  let create (data : float list) (shape : int list) =
    let c_ptr =
      C.Functions.mps_tensor_data_from_float_array
        (Ctypes.CArray.of_list Ctypes.float data |> Ctypes.CArray.start)
        (Ctypes.CArray.of_list Ctypes.int shape |> Ctypes.CArray.start)
        (List.length shape)
    in
    Gc.finalise C.Functions.mps_release_tensor_data c_ptr;
    { c_ptr }

  let zeroes shape =
    let size = List.fold_left ( * ) 1 shape in
    create (List.init size (fun _ -> 0.)) shape
end

let create () =
  let graph = C.Functions.mps_create_graph () in
  Gc.finalise C.Functions.mps_release_graph graph;
  { c_ptr = graph }

let placeholder t (shape : int list) =
  let shape_arr = Ctypes.CArray.of_list Ctypes.int shape in
  let c_ptr =
    C.Functions.mps_graph_placeholder t.c_ptr
      (Ctypes.CArray.start shape_arr)
      (List.length shape)
  in
  let tensor = Tensor.of_c_ptr c_ptr in
  tensor

let add t a b =
  let c_ptr = C.Functions.mps_graph_attach_addition t.c_ptr a b in
  let tensor = Tensor.of_c_ptr c_ptr in
  tensor

let run_forward_backward_with_feeds t (feed_tensors : Tensor.t list)
    (feed_data : Tensor_data.t list) output_tensor (output_data : Tensor_data.t)
    =
  let num_feeds = List.length feed_tensors in
  let feed_tensor_handles = List.map (fun (t : Tensor.t) -> t) feed_tensors in
  let feed_data_handles =
    List.map (fun (d : Tensor_data.t) -> d.c_ptr) feed_data
  in
  C.Functions.mps_graph_run_forward_backward_with_feeds t.c_ptr
    (Ctypes.CArray.of_list C.Types.Tensor_handle.t feed_tensor_handles
    |> Ctypes.CArray.start)
    (Ctypes.CArray.of_list C.Types.Tensor_data_handle.t feed_data_handles
    |> Ctypes.CArray.start)
    num_feeds output_tensor output_data.c_ptr
