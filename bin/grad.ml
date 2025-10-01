open Statlearning

let () =
  let device = Mtl.Device.create_system_default in
  let command_queue = Mtl.CommandQueue.make device in
  let buffer_a = Mtl.Buffer.of_data device [| 1.; 2.; 3.; 4. |] in
  let buffer_b = Mtl.Buffer.of_data device [| 5.; 6.; 7.; 8. |] in
  let buffer_c = Mtl.Buffer.of_length device 4 in
  let descriptor_a =
    Mps.MatrixDescriptor.create ~rows:2 ~columns:2 ~row_bytes:8
  in
  let descriptor_b =
    Mps.MatrixDescriptor.create ~rows:2 ~columns:2 ~row_bytes:8
  in
  let descriptor_c =
    Mps.MatrixDescriptor.create ~rows:2 ~columns:2 ~row_bytes:8
  in
  let matrix_a =
    Mps.Matrix.of_buffer ~descriptor:descriptor_a ~buffer:buffer_a
  in
  let matrix_b =
    Mps.Matrix.of_buffer ~descriptor:descriptor_b ~buffer:buffer_b
  in
  let matrix_c =
    Mps.Matrix.of_buffer ~descriptor:descriptor_c ~buffer:buffer_c
  in
  let kernel =
    Mps.Kernel.MatrixMultiplication.alloc ~transpose_left:false
      ~transpose_right:false ~rows:2 ~columns:2 ~inner_dim:2 ~alpha:1.0
      ~beta:0.0 device
  in
  let command_buffer = Mtl.CommandBuffer.make command_queue in
  Mps.Kernel.MatrixMultiplication.encode kernel ~left_matrix:matrix_a
    ~right_matrix:matrix_b ~result_matrix:matrix_c command_buffer;
  Mtl.CommandBuffer.commit command_buffer;
  Mtl.CommandBuffer.wait_until_completed command_buffer;
  let result = Mtl.Buffer.to_float_array buffer_c in
  Array.iter (fun x -> Printf.printf "%f " x) result;
  print_newline ();
  Mtl.Device.release device

