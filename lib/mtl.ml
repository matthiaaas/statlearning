open! Ctypes

module Device = struct
  type t = { ptr : C.Types.MtlDevice.t; name : string }

  let create_system_default =
    let ptr = C.Functions.mtl_create_system_default_device () in
    let name = C.Functions.mtl_get_device_name ptr in
    { ptr; name }

  let release device = C.Functions.mtl_release_device device.ptr
end

module CommandQueue = struct
  type t = { ptr : C.Types.MtlCommandQueue.t }

  let make (device : Device.t) =
    { ptr = C.Functions.mtl_make_command_queue device.ptr }
end

module CommandBuffer = struct
  type t = { ptr : C.Types.MtlCommandBuffer.t }

  let make (command_queue : CommandQueue.t) =
    { ptr = C.Functions.mtl_make_command_buffer command_queue.ptr }

  let commit (command_buffer : t) =
    C.Functions.mtl_commit_command_buffer command_buffer.ptr

  let wait_until_completed (command_buffer : t) =
    C.Functions.mtl_wait_until_completed command_buffer.ptr
end

module Buffer = struct
  type t = { ptr : C.Types.MtlBuffer.t; byte_size : int }

  let of_length (device : Device.t) (length : int) =
    let arr = Bigarray.(Array1.create float32 c_layout length) in
    let byte_size = Bigarray.Array1.dim arr * 4 in
    {
      ptr =
        C.Functions.mtl_make_buffer_with_length device.ptr
          (Unsigned.ULong.of_int byte_size);
      byte_size;
    }

  let of_data (device : Device.t) (data : float array) =
    let arr = Bigarray.(Array1.of_array float32 c_layout data) in
    let byte_size = Bigarray.Array1.dim arr * 4 in
    let ptr = Ctypes.(to_voidp (bigarray_start array1 arr)) in
    {
      ptr =
        C.Functions.mtl_make_buffer_with_bytes device.ptr ptr
          (Unsigned.ULong.of_int byte_size);
      byte_size;
    }

  let to_float_array (buffer : t) : float array =
    let len = buffer.byte_size / 4 in
    let arr = Bigarray.(Array1.create float32 c_layout len) in
    let dst_ptr = Ctypes.(to_voidp (bigarray_start array1 arr)) in
    C.Functions.mtl_get_buffer_contents buffer.ptr dst_ptr
      (Unsigned.ULong.of_int buffer.byte_size);
    Array.init len (fun i -> Bigarray.Array1.get arr i)
end
