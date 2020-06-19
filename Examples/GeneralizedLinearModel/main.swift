// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/generalized_linear_model.py

import Foundation
import XGBoost

// This script demonstrate how to fit generalized linear model in xgboost
// basically, we are using linear model, instead of tree for our boosters

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


// change booster to gblinear, so that we are fitting a linear model
// alpha is the L1 regularizer
// lambda is the L2 regularizer
// you can also set lambda_bias which is L2 regularizer on the bias term
var parameters = [
    Parameter("seed", 0),
    Parameter("lambda", 1),
    Parameter("alpha", 0.0001),
    Parameter("booster", "gblinear"),
    Parameter("objective", "binary:logistic"),
]

// normally, you do not need to set eta (step_size)
// XGBoost uses a parallel coordinate descent algorithm (shotgun),
// there could be affection on convergence with parallelization on certain cases
// setting eta to be smaller value, e.g 0.5 can make the optimization more stable
// parameters.append(Parameter("eta", 1))

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
