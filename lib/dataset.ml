type ('x, 'y, 'c) t = ('x list * 'c) list
type ('x, 'y) single = ('x, 'y, 'y) t
type ('x, 'y) multi = ('x, 'y, 'y list) t

let single_of_csv (feature_indices : int list) (target_index : int)
    (csv : Csv.t) : (float, float) single =
  let parse_float_at_index row index = List.nth row index |> float_of_string in
  List.map
    (fun row ->
      let features = List.map (parse_float_at_index row) feature_indices in
      let target = parse_float_at_index row target_index in
      (features, target))
    csv

let split_train_test ?(train_ratio = 0.8) (data : ('x, 'y, 'c) t) :
    ('x, 'y, 'c) t * ('x, 'y, 'c) t =
  assert (train_ratio > 0.0 && train_ratio < 1.0);
  let train_size =
    int_of_float (train_ratio *. float_of_int (List.length data))
  in
  let rec aux n acc rest =
    if n <= 0 then (List.rev acc, rest)
    else
      match rest with
      | [] -> (List.rev acc, [])
      | x :: xs -> aux (n - 1) (x :: acc) xs
  in
  aux train_size [] data
