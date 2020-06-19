// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/generalized_linear_model.py

import Foundation
import XGBoost

let train = try DMatrix(
    name: "train",
    from: "Examples/Data/agaricus.txt.train",
    format: .libsvm
)
let test = try DMatrix(
    name: "test",
    from: "Examples/Data/agaricus.txt.test",
    format: .libsvm
)

let parameters = [
    Parameter("seed", 0),
    Parameter("lambda", 1),
    Parameter("alpha", 0.0001),
    Parameter("booster", "gblinear"),
    Parameter("objective", "binary:logistic"),
]

let booster = try Booster(
    with: [train, test],
    parameters: parameters
)

try booster.train(
    iterations: 4,
    trainingData: train
)

let preds = try booster.predict(from: test)
let labels = try test.get(field: .label)

let error = zip(preds, labels).map {
    ($0 > 0.5) == ($1 == 1) ? 0 : 1
}.sum() / Float(preds.count)

print("Test error: \(error).")
