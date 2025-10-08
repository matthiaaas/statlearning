# `statlearning`

# Statistics

## Regression Tree

```ocaml
(* Fit *)
let data = [ ([ 1.0; 2.0 ], 1.0); ... ] in
let tree = Regression_tree.fit data ~max_depth:10 in
print_endline (Regression_tree.to_string tree ~indent:4);

(* Predict *)
let x = (List.hd data |> fst) in
print_endline (Regression_tree.predict tree x |> string_of_float)
```

## Gradient Boosting Regressor

```ocaml
(* Fit *)
let data = [ ([ 1.0; 2.0 ], 1.0); ... ] in
let regressor = Gradient_boosting_regressor.fit data ~n_estimators:8 ~learning_rate:0.1 in

let improved_regressor = Gradient_boosting_regressor.boost regressor [...]

(* Predict *)
let x = (List.hd data |> fst) in
print_endline (Gradient_boosting_regressor.predict regressor x |> string_of_float)
```

# Dataset

```ocaml
let data : (float, float) Dataset.single = [ ([ 1.0; 2.0 ], 1.0); ... ] in
```

```ocaml
let load_dataset =
  let csv = Csv.load "data/Auto.csv" in
  Dataset.single_of_csv [ 4 ] 5 csv
```

# Apple Hardware Acceleration with Metal Performance Shaders, MPS

C/Objective-C bindings to Apple [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders).

## MPSGraph

C/Objective-C bindings to higher-level [Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph?language=objc) API. This is used for automatic gradient computation in `Tensor`s.

```ocaml
let graph = Mps_graph.create () in

let tensor_a = Mps_graph.placeholder graph [ 4 ] in
let tensor_b = Mps_graph.placeholder graph [ 4 ] in
let data_a = Mps_graph.Tensor_data.create [ 1.; 2.; 3.; 4. ] [ 4 ] in
let data_b = Mps_graph.Tensor_data.create [ 10.; 20.; 30.; 40. ] [ 4 ] in

let tensor_result = Mps_graph.add tensor_a tensor_b in
let data_result = Mps_graph.Tensor_data.zeroes [ 4 ] in

Mps_graph.run_forward_backward_with_feeds graph feeds;

let result = Mps_graph.Tensor_data.to_list data_e in
List.iter (Printf.printf "%f ") result;
Printf.printf "\n";
```

## Metal/MPS Primitives (Legacy)

First iteration of binding directly to lower-level Metal and MPS primitives. **PoC-only**.

```ocaml
let device = Mtl.Device.create_system_default in
let command_queue = Mtl.CommandQueue.make device in

let buffer_a = Mtl.Buffer.of_data device [| 1.; 2.; 3.; 4. |] in
let buffer_b = Mtl.Buffer.of_data device [| 5.; 6.; 7.; 8. |] in
let buffer_c = Mtl.Buffer.of_length device 4 in

let descriptor_a = Mps.MatrixDescriptor.create ~rows:2 ~columns:2 ~row_bytes:8 in
let descriptor_b = Mps.MatrixDescriptor.create ~rows:2 ~columns:2 ~row_bytes:8 in
let descriptor_c = Mps.MatrixDescriptor.create ~rows:2 ~columns:2 ~row_bytes:8 in

let matrix_a = Mps.Matrix.of_buffer ~descriptor:descriptor_a ~buffer:buffer_a in
let matrix_b = Mps.Matrix.of_buffer ~descriptor:descriptor_b ~buffer:buffer_b in
let matrix_c = Mps.Matrix.of_buffer ~descriptor:descriptor_c ~buffer:buffer_c in

let kernel = Mps.Kernel.MatrixMultiplication.alloc ~transpose_left:false ~transpose_right:false ~rows:2 ~columns:2 ~inner_dim:2 ~alpha:1.0 ~beta:0.0 device in

let command_buffer = Mtl.CommandBuffer.make command_queue in
Mps.Kernel.MatrixMultiplication.encode kernel ~left_matrix:matrix_a ~right_matrix:matrix_b ~result_matrix:matrix_c command_buffer;
Mtl.CommandBuffer.commit command_buffer;
Mtl.CommandBuffer.wait_until_completed command_buffer;
let result = Mtl.Buffer.to_float_array buffer_c in
...
```

# Tensor & Autograd engine (+ Apple MPS backend)

```ocaml
let a = Tensor.{ value = [|2.0|] } |> Tensor.to_device Autograd.Device.Mps in
let b = Tensor.{ value = [|3.0|] } |> Tensor.to_device Autograd.Device.Mps in
let product = mul a b in
print_endline (product |> Tensor.to_device Autograd.Device.Cpu |> Tensor.to_string);
```
