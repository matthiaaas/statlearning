type ('feature, 'value) t = {
  trees : ('feature, 'value) Regression_tree.t list;
  bias : 'value;
  learning_rate : float;
}

let predict (regressor : ('feature, 'value) t) (x : 'feature list) =
  List.fold_left
    (fun acc tree ->
      let tree_pred = Regression_tree.predict tree x in
      acc +. (tree_pred *. regressor.learning_rate))
    regressor.bias (List.rev regressor.trees)

let boost ?(max_depth = 3) (regressor : ('feature, 'value) t)
    (data : ('feature, 'value) Dataset.single) : ('feature, 'value) t =
  let xs, ys = List.split data in
  let residuals =
    xs |> List.map (predict regressor) |> Common.residuals ys |> List.combine xs
  in
  let tree = Regression_tree.fit residuals ~max_depth in
  {
    trees = tree :: regressor.trees;
    bias = regressor.bias;
    learning_rate = regressor.learning_rate;
  }

let fit ?(n_estimators = 100) ?(learning_rate = 0.1) ?(max_depth = 3)
    (data : ('x, 'y) Dataset.single) =
  Common.apply_n_times
    (fun regressor -> boost regressor ~max_depth data)
    { trees = []; bias = Common.mean (List.map snd data); learning_rate }
    ~n:n_estimators
