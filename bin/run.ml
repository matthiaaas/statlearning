open Statlearning

let () =
  let graph = Mps_graph.create () in
  let tensor_a = Mps_graph.placeholder graph [ 4 ] in
  let tensor_b = Mps_graph.placeholder graph [ 4 ] in
  let tensor_c = Mps_graph.add graph tensor_a tensor_b in
  let tensor_d = Mps_graph.add graph tensor_c tensor_c in
  let data_a = Mps_graph.Tensor_data.create [ 1.; 2.; 3.; 4. ] [ 4 ] in
  let data_b = Mps_graph.Tensor_data.create [ 10.; 20.; 30.; 40. ] [ 4 ] in
  let data_d = Mps_graph.Tensor_data.zeroes [ 4 ] in
  let _ =
    Mps_graph.run_forward_backward_with_feeds graph [ tensor_a; tensor_b ]
      [ data_a; data_b ] tensor_d data_d
  in
  Printf.printf "Done\n"
