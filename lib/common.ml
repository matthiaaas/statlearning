let sum = List.fold_left ( +. ) 0.0

let mean (ys : float list) : float =
  let n = float_of_int (List.length ys) in
  match ys with [] -> 0.0 | _ -> sum ys /. n

let residuals (targets : float list) (ys : float list) : float list =
  List.map2 ( -. ) targets ys

let mse (targets : float list) (ys : float list) : float =
  assert (List.length ys = List.length targets);
  let n = float_of_int (List.length targets) in
  match ys with
  | [] -> 0.0
  | _ ->
      let squared_errors =
        List.map2
          (fun y t ->
            let diff = y -. t in
            diff *. diff)
          ys targets
      in
      sum squared_errors /. n

let svar (ys : float list) : float =
  let n = float_of_int (List.length ys) in
  match ys with
  | [] -> 0.0
  | _ ->
      let y_mean = mean ys in
      let squared_diffs =
        List.map
          (fun y ->
            let diff = y -. y_mean in
            diff *. diff)
          ys
      in
      sum squared_diffs /. n

let rec transpose = function
  | [] | [] :: _ -> []
  | rows -> List.map List.hd rows :: transpose (List.map List.tl rows)

let rec apply_n_times ~n f x =
  if n <= 0 then x else apply_n_times ~n:(n - 1) f (f x)
