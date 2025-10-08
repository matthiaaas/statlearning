open Statlearning

let () =
  let graph = Mps_graph.create () in
  let tensor_a = Mps_graph.placeholder graph [ 4 ] in
  let tensor_b = Mps_graph.placeholder graph [ 4 ] in
  let tensor_c = Mps_graph.add graph tensor_a tensor_b in
  let tensor_d = Mps_graph.mul graph tensor_a tensor_b in
  let tensor_e = Mps_graph.add graph tensor_c tensor_d in
  let data_a = Mps_graph.Tensor_data.create [ 1.; 2.; 3.; 4. ] [ 4 ] in
  let data_b = Mps_graph.Tensor_data.create [ 10.; 20.; 30.; 40. ] [ 4 ] in
  let data_e = Mps_graph.Tensor_data.zeroes [ 4 ] in
  let _ =
    Mps_graph.run_forward_backward_with_feeds graph [ tensor_a; tensor_b ]
      [ data_a; data_b ] tensor_e data_e
  in
  Printf.printf "Done\n";
  let result = Mps_graph.Tensor_data.to_list data_e in
  List.iter (Printf.printf "%f ") result;
  Printf.printf "\n";
  ()
