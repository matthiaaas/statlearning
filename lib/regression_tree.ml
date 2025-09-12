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

let split (threshold : float) (pairs : (float * float) list) =
  let left_pairs, right_pairs =
    List.partition (fun (x, _) -> x <= threshold) pairs
  in
  (List.map snd left_pairs, List.map snd right_pairs)

let calculate_loss_for_split left_ys right_ys =
  let n_left = float_of_int (List.length left_ys) in
  let n_right = float_of_int (List.length right_ys) in
  let n = n_left +. n_right in
  if n = 0.0 then infinity
  else ((n_left *. mse left_ys) +. (n_right *. mse right_ys)) /. n

let find_best_split_of_feature (xs : float list) (ys : float list) =
  let possible_thresholds = thresholds xs in
  if List.is_empty possible_thresholds then None
  else
    let loss_by_split =
      List.map
        (fun threshold ->
          let left_ys, right_ys = split threshold (List.combine xs ys) in
          (threshold, calculate_loss_for_split left_ys right_ys))
        possible_thresholds
    in
    let best =
      List.fold_left
        (fun (t1, l1) (t2, l2) -> if l1 < l2 then (t1, l1) else (t2, l2))
        (List.hd loss_by_split) (List.tl loss_by_split)
    in
    Some best

let find_best_split (features : float list list) (ys : float list) =
  let xs_by_feature = transpose features in
  let best_splits_by_feature =
    List.map (fun xs -> find_best_split_of_feature xs ys) xs_by_feature
  in
  let best_splits_by_feature_and_index =
    best_splits_by_feature
    |> List.mapi (fun i opt ->
           match opt with Some (t, l) -> Some (i, t, l) | None -> None)
    |> List.filter_map Fun.id
  in
  let best =
    List.fold_left
      (fun (i1, t1, l1) (i2, t2, l2) ->
        if l2 < l1 then (i2, t2, l2) else (i1, t1, l1))
      (-1, 0.0, infinity) best_splits_by_feature_and_index
  in
  Some best

let rec fit ~(data : ('x, 'y) dataset) =
  if List.length data <= 1 then Leaf (mean (List.map snd data))
  else
    let features, ys = List.split data in
    match find_best_split features ys with
    | None -> Leaf (mean ys)
    | Some (best_feature, best_threshold, _) ->
        let left_data, right_data =
          List.partition
            (fun (x, _) -> List.nth x best_feature <= best_threshold)
            data
        in
        Node
          {
            feature_index = best_feature;
            threshold = best_threshold;
            left = fit ~data:left_data;
            right = fit ~data:right_data;
          }

