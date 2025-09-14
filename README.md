# `statlearning`

# Statistics

## Regression Tree

```ocaml
(* Fit *)
let data = [ ([ 1.0; 2.0 ], 1.0); ... ] in
let tree = Regression_tree.fit data ~max_depth:10 in
print_endline (Regression_tree.to_string tree ~indent:4);

(* Predict *)
let x = (List.hd data |> fst) in
print_endline (Regression_tree.predict tree x |> string_of_float)
```

## Gradient Boosting Regressor

```ocaml
(* Fit *)
let data = [ ([ 1.0; 2.0 ], 1.0); ... ] in
let regressor = Gradient_boosting_regressor.fit data ~n_estimators:8 in

let improved_regressor = Gradient_boosting_regressor.boost regressor [...]

(* Predict *)
let x = (List.hd data |> fst) in
print_endline (Gradient_boosting_regressor.predict regressor x |> string_of_float)
```
