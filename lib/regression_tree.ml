type ('feature, 'value) t =
  | Node of {
      feature_index : int;
      threshold : 'feature;
      left : ('feature, 'value) t;
      right : ('feature, 'value) t;
    }
  | Leaf of 'value

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
  else
    ((n_left *. Common.svar left_ys) +. (n_right *. Common.svar right_ys)) /. n

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
  let xs_by_feature = Common.transpose features in
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

let fit ?(max_depth = max_int) (data : ('x, 'y) Common.dataset) =
  let rec aux max_depth data =
    if List.length data <= 1 || max_depth <= 0 then
      Leaf (Common.mean (List.map snd data))
    else
      let features, ys = List.split data in
      match find_best_split features ys with
      | None -> Leaf (Common.mean ys)
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
              left = aux (max_depth - 1) left_data;
              right = aux (max_depth - 1) right_data;
            }
  in
  aux max_depth data

let predict (tree : (float, float) t) (inputs : float list) : float =
  let rec aux node =
    match node with
    | Leaf value -> value
    | Node { feature_index; threshold; left; right } ->
        if List.nth inputs feature_index <= threshold then aux left
        else aux right
  in
  aux tree

let rec to_string ?(indent = 0) ?(level = 0) (tree : (float, float) t) =
  let spaces = if indent > 0 then String.make (level * indent) ' ' else "" in
  match tree with
  | Leaf value -> Printf.sprintf "%sLeaf (%f)" spaces value
  | Node { feature_index; threshold; left; right } ->
      let next_level = level + 1 in
      let left_str = to_string ~indent ~level:next_level left in
      let right_str = to_string ~indent ~level:next_level right in
      Printf.sprintf "%sNode (feature: %d, threshold: %f,\n%s,\n%s)" spaces
        feature_index threshold left_str right_str
