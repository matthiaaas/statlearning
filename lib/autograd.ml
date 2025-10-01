module Device = struct
  type t = Cpu | Mps

  let to_string = function Cpu -> "cpu" | Mps -> "mps"
end
