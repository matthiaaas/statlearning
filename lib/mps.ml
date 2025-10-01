open! Ctypes
open Mtl

module MatrixDescriptor = struct
  type t = { ptr : C.Types.MpsMatrixDescriptor.t }

  let create ~(rows : int) ~(columns : int) ~(row_bytes : int) =
    let ptr =
      C.Functions.mps_create_matrix_descriptor
        (Unsigned.ULong.of_int rows)
        (Unsigned.ULong.of_int columns)
        (Unsigned.ULong.of_int row_bytes)
    in
    { ptr }
end

module Matrix = struct
  type t = { ptr : C.Types.MpsMatrix.t }

  let of_buffer ~(descriptor : MatrixDescriptor.t) ~(buffer : Buffer.t) =
    let ptr = C.Functions.mps_create_matrix buffer.ptr descriptor.ptr in
    { ptr }
end

module Kernel = struct
  module MatrixMultiplication = struct
    type t = { ptr : C.Types.MpsMatrixMultiplication.t }

    let alloc ~(transpose_left : bool) ~(transpose_right : bool) ~(rows : int)
        ~(columns : int) ~(inner_dim : int) ~(alpha : float) ~(beta : float)
        (device : Device.t) =
      let ptr =
        C.Functions.mps_alloc_matrix_multiplication device.ptr transpose_left
          transpose_right
          (Unsigned.ULong.of_int rows)
          (Unsigned.ULong.of_int columns)
          (Unsigned.ULong.of_int inner_dim)
          alpha beta
      in
      { ptr }

    let encode (kernel : t) ~(left_matrix : Matrix.t) ~(right_matrix : Matrix.t)
        ~(result_matrix : Matrix.t) (command_buffer : CommandBuffer.t) =
      C.Functions.mps_encode_matrix_multiplication kernel.ptr command_buffer.ptr
        left_matrix.ptr right_matrix.ptr result_matrix.ptr;
      ()
  end
end
