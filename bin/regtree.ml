open Statlearning

let () =
  (* House price prediction dataset:
     Features:
     - square_footage: Size of house in sq ft (100s)
     - num_bedrooms: Number of bedrooms
     - Target: House price in $10,000s
     
     Data follows roughly: price = 10 + 5*square_footage + 2*num_bedrooms + noise
     Noise is randomly distributed between -1 and 1
     
     Expected tree structure:
     - Root split on square_footage (most significant feature)
     - Second level splits on num_bedrooms if max_depth=2
  *)
  let data =
    [
      (* [square_footage, num_bedrooms], price *)
      ([ 10.0; 2.0 ], 62.5);
      (* 1000 sq ft, 2 bedrooms: ~$625,000 *)
      ([ 12.0; 3.0 ], 75.8);
      (* 1200 sq ft, 3 bedrooms: ~$758,000 *)
      ([ 14.0; 3.0 ], 84.2);
      (* 1400 sq ft, 3 bedrooms: ~$842,000 *)
      ([ 8.0; 2.0 ], 52.3);
      (* 800 sq ft, 2 bedrooms: ~$523,000 *)
      ([ 9.0; 1.0 ], 54.7);
      (* 900 sq ft, 1 bedroom: ~$547,000 *)
      ([ 15.0; 4.0 ], 94.6);
      (* 1500 sq ft, 4 bedrooms: ~$946,000 *)
      ([ 11.0; 2.0 ], 67.1);
      (* 1100 sq ft, 2 bedrooms: ~$671,000 *)
      ([ 13.0; 2.0 ], 76.9);
      (* 1300 sq ft, 2 bedrooms: ~$769,000 *)
      ([ 7.0; 1.0 ], 45.2);
      (* 700 sq ft, 1 bedroom: ~$452,000 *)
      ([ 16.0; 3.0 ], 93.8);
      (* 1600 sq ft, 3 bedrooms: ~$938,000 *)
    ]
  in
  let tree = Regression_tree.fit data ~max_depth:4 in
  print_endline (Regression_tree.to_string tree ~indent:4);
  let preds, targets =
    List.map
      (fun (features, y) -> (Regression_tree.predict tree features, y))
      data
    |> List.split
  in
  print_endline (Common.mse targets preds |> string_of_float)
