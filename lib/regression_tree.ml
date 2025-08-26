type ('feature, 'value) t =
  | Node of {
      feature_index : int;
      threshold : 'feature;
      left : ('feature, 'value) t;
      right : ('feature, 'value) t;
    }
  | Leaf of 'value

type ('x, 'y) dataset = ('x list * 'y) list

let sum (ys : float list) = List.fold_left ( +. ) 0.0 ys

let mean (ys : float list) : float =
  match ys with [] -> 0.0 | _ -> sum ys /. float_of_int (List.length ys)

let mse (ys : float list) : float =
  let avg = mean ys in
  let squared_errors =
    List.map
      (fun y ->
        let diff = y -. avg in
        diff *. diff)
      ys
  in
  mean squared_errors

(* [(a1, a2, ...), (b1, b2, ...), ...] -> [[a1, b1, ...], [a2, b2, ...], ...] *)
let rec transpose = function
  | [] | [] :: _ -> []
  | rows -> List.map List.hd rows :: transpose (List.map List.tl rows)

let thresholds (xs : float list) : float list =
  let uniquely_sorted_xs = List.sort_uniq compare xs in
  let rec build_thresholds acc lst =
    match lst with
    | a :: (b :: _ as rest) ->
        let m = (a +. b) /. 2.0 in
        build_thresholds (m :: acc) rest
    | _ -> List.rev acc
  in
  build_thresholds [] uniquely_sorted_xs

let split threshold pairs =
  let left_pairs, right_pairs =
    List.partition (fun (x, _) -> x <= threshold) pairs
  in
  (List.map snd left_pairs, List.map snd right_pairs)

let calculate_gain_for_split left_ys right_ys =
  let lc = float_of_int (List.length left_ys) in
  let rc = float_of_int (List.length right_ys) in
  let n = lc +. rc in
  if n = 0.0 then infinity
  else ((lc *. mse left_ys) +. (rc *. mse right_ys)) /. n

let find_best_split_of_feature xs ys =
  let possible_thresholds = List.map thresholds xs in
  if List.is_empty possible_thresholds then None
  else
    let gains_by_split =
      List.map
        (fun threshold ->
          let left_ys, right_ys = split threshold (List.combine xs ys) in
          (threshold, calculate_gain_for_split left_ys right_ys))
        possible_thresholds
    in
    let best =
      List.fold_left
        (fun (t1, g1) (t2, g2) -> if g1 < g2 then (t1, g1) else (t2, g2))
        (List.hd gains_by_split) (List.tl gains_by_split)
    in
    Some best

let find_best_split features ys =
  let xs_by_feature = transpose features in
  List.map find_best_split_of_feature xs_by_feature

let rec fit ~(data : ('x, 'y) dataset) =
  if List.length data <= 1 then Leaf (mean (List.map snd data))
  else
    let features, ys = List.split data in
    match find_best_split features ys with
    | None -> Leaf (mean ys)
    | Some (best_threshold, best_feature) -> Leaf 1.0

let () =
  let f a b = (a *. 4.0) +. (b *. 3.0) +. 1.0 in
  print_endline
    (String.concat " "
       (List.map
          (fun m -> Printf.sprintf "%f" m)
          (thresholds [ 2.0; 3.0; 4.0 ])))
(* print_endline (Printf.sprintf "%f" (mean [])) *)
(* print_endline
    (match fit ~data:[ ([ (2.0, 3.0) ], f 2.0 3.0) ] with
    | Leaf v -> Printf.sprintf "%f" v
    | Node _ -> "Node") *)

(* let () =
  let f (a : float) (b : float) = (a *. 4.0) +. (b *. 3.0) +. 1.0 in
  let data = [ ([ (2.0, 4.0) ], f 2.0 4.0); ([ (3.0, 5.0) ], f 3.0 5.0) ] in
  let data_str : string =
    data
    |> List.map (fun (xs, y) ->
           let xs_str =
             xs
             |> List.map (fun (a, b) -> Printf.sprintf "(%f, %f)" a b)
             |> String.concat ", "
           in
           Printf.sprintf "(%s, %f)" xs_str y)
    |> String.concat "; "
  in
  print_endline ("Data: [" ^ data_str ^ "]") *)
