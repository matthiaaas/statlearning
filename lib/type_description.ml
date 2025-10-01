open! Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  module MtlDevice = struct
    let mtl_device_struct : [ `mtl_device ] structure typ =
      structure "mtl_device"

    type t = [ `mtl_device ] structure ptr

    let t : t typ = ptr mtl_device_struct
  end

  module MtlCommandQueue = struct
    let mtl_command_queue_struct : [ `mtl_command_queue ] structure typ =
      structure "mtl_command_queue"

    type t = [ `mtl_command_queue ] structure ptr

    let t : t typ = ptr mtl_command_queue_struct
  end

  module MtlCommandBuffer = struct
    let mtl_command_buffer_struct : [ `mtl_command_buffer ] structure typ =
      structure "mtl_command_buffer"

    type t = [ `mtl_command_buffer ] structure ptr

    let t : t typ = ptr mtl_command_buffer_struct
  end

  module MtlBuffer = struct
    let mtl_buffer_struct : [ `mtl_buffer ] structure typ =
      structure "mtl_buffer"

    type t = [ `mtl_buffer ] structure ptr

    let t : t typ = ptr mtl_buffer_struct
  end

  module MpsMatrixMultiplication = struct
    let mps_matrix_multiplication_struct :
        [ `mps_matrix_multiplication ] structure typ =
      structure "mps_matrix_multiplication"

    type t = [ `mps_matrix_multiplication ] structure ptr

    let t : t typ = ptr mps_matrix_multiplication_struct
  end

  module MpsMatrixDescriptor = struct
    let mps_matrix_descriptor_struct : [ `mps_matrix_descriptor ] structure typ
        =
      structure "mps_matrix_descriptor"

    type t = [ `mps_matrix_descriptor ] structure ptr

    let t : t typ = ptr mps_matrix_descriptor_struct
  end

  module MpsMatrix = struct
    let mps_matrix_struct : [ `mps_matrix ] structure typ =
      structure "mps_matrix"

    type t = [ `mps_matrix ] structure ptr

    let t : t typ = ptr mps_matrix_struct
  end
end
