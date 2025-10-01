open! Ctypes
module Types = Types_generated

module Functions (F : Ctypes.FOREIGN) = struct
  open F

  let mtl_create_system_default_device =
    foreign "mtl_create_system_default_device_c"
      (void @-> returning Types.MtlDevice.t)

  let mtl_release_device =
    foreign "mtl_release_device_c" (Types.MtlDevice.t @-> returning void)

  let mtl_get_device_name =
    foreign "mtl_get_device_name_c" (Types.MtlDevice.t @-> returning string)

  let mtl_make_command_queue =
    foreign "mtl_make_command_queue_c"
      (Types.MtlDevice.t @-> returning Types.MtlCommandQueue.t)

  let mtl_make_command_buffer =
    foreign "mtl_make_command_buffer_c"
      (Types.MtlCommandQueue.t @-> returning Types.MtlCommandBuffer.t)

  let mtl_commit_command_buffer =
    foreign "mtl_commit_command_buffer_c"
      (Types.MtlCommandBuffer.t @-> returning void)

  let mtl_wait_until_completed =
    foreign "mtl_wait_until_completed_c"
      (Types.MtlCommandBuffer.t @-> returning void)

  let mtl_make_buffer_with_length =
    foreign "mtl_make_buffer_with_length_c"
      (Types.MtlDevice.t @-> ulong @-> returning Types.MtlBuffer.t)

  let mtl_make_buffer_with_bytes =
    foreign "mtl_make_buffer_with_bytes_c"
      (Types.MtlDevice.t @-> ptr void @-> ulong @-> returning Types.MtlBuffer.t)

  let mtl_get_buffer_contents =
    foreign "mtl_get_buffer_contents_c"
      (Types.MtlBuffer.t @-> ptr void @-> ulong @-> returning void)

  let mps_alloc_matrix_multiplication =
    foreign "mps_alloc_matrix_multiplication_c"
      (Types.MtlDevice.t @-> bool @-> bool @-> ulong @-> ulong @-> ulong
     @-> float @-> float
      @-> returning Types.MpsMatrixMultiplication.t)

  let mps_encode_matrix_multiplication =
    foreign "mps_encode_matrix_multiplication_to_command_buffer_c"
      (Types.MpsMatrixMultiplication.t @-> Types.MtlCommandBuffer.t
     @-> Types.MpsMatrix.t @-> Types.MpsMatrix.t @-> Types.MpsMatrix.t
     @-> returning void)

  let mps_create_matrix_descriptor =
    foreign "mps_create_matrix_descriptor_c"
      (ulong @-> ulong @-> ulong @-> returning Types.MpsMatrixDescriptor.t)

  let mps_create_matrix =
    foreign "mps_create_matrix_c"
      (Types.MtlBuffer.t @-> Types.MpsMatrixDescriptor.t
      @-> returning Types.MpsMatrix.t)
end
