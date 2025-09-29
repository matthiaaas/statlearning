open Statlearning

let load_csv =
  let csv = Csv.load "data/Auto.csv" in
  match csv with _ :: rows -> rows | [] -> []

let load_dataset =
  let csv = load_csv in
  Dataset.single_of_csv [ 3; 4 ] 5 csv

let eval_performance (dataset : (float, float) Dataset.single) loss predict =
  let preds, targets =
    dataset
    |> List.map (fun (features, y) -> (predict features, y))
    |> List.split
  in
  loss targets preds

let () =
  let data = load_dataset in
  let train_data, test_data = Dataset.split_train_test ~train_ratio:0.9 data in
  let regressor =
    Gradient_boosting_regressor.fit ~n_estimators:65 ~learning_rate:0.1
      ~max_depth:3 train_data
  in
  Printf.printf "Train MSE: %f\n"
    (eval_performance train_data Common.mse
       (Gradient_boosting_regressor.predict regressor));
  Printf.printf "Test MSE: %f\n"
    (eval_performance test_data Common.mse
       (Gradient_boosting_regressor.predict regressor));
  let print_prediction_comparison count =
    let rec loop i data =
      if i = 0 then ()
      else
        match data with
        | (features, target) :: rest ->
            let prediction =
              Gradient_boosting_regressor.predict regressor features
            in
            Printf.printf "Example %d - Expected: %.2f, Actual: %.2f\n" i target
              prediction;
            loop (i - 1) rest
        | [] -> ()
    in
    loop count test_data
  in
  print_prediction_comparison 5
