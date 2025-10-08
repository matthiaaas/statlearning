open! Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  module Graph_handle = struct
    type mps_graph_handle

    let mps_graph_handle : mps_graph_handle structure typ =
      structure "MPSGraphHandle"

    type t = mps_graph_handle structure ptr

    let t : t typ = ptr mps_graph_handle
  end

  module Tensor_handle = struct
    type mps_tensor_handle

    let mps_tensor_handle : mps_tensor_handle structure typ =
      structure "MPSTensorHandle"

    type t = mps_tensor_handle structure ptr

    let t : t typ = ptr mps_tensor_handle
  end

  module Tensor_data_handle = struct
    type mps_tensor_data_handle

    let mps_tensor_data_handle : mps_tensor_data_handle structure typ =
      structure "MPSTensorDataHandle"

    type t = mps_tensor_data_handle structure ptr

    let t : t typ = ptr mps_tensor_data_handle
  end
end
