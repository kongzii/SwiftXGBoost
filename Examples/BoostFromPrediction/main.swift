// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/boost_from_prediction.py

import Foundation
import PythonKit
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
    Parameter("max_depth", 2),
    Parameter("eta", 1),
    Parameter("objective", "binary:logistic"),
]

let booster = try Booster(
    with: [train],
    parameters: parameters
)

try booster.train(
    iterations: 1,
    trainingData: train
)

// Note
// We need the margin value instead of transformed prediction in set_base_margin.
// Do predict with `outputMargin: true`, will always give you margin values before logistic transformation.
let predictedTrain = try booster.predict(
    from: train, outputMargin: true
)
let predictedTest = try booster.predict(
    from: test, outputMargin: true
)

try train.set(field: .baseMargin, values: predictedTrain)
try test.set(field: .baseMargin, values: predictedTest)

let booster2 = try Booster(
    with: [train],
    parameters: parameters
)

try booster2.train(
    iterations: 1,
    trainingData: train
)

let predictedTrain2 = try booster2.predict(
    from: train, outputMargin: true
)
let predictedTest2 = try booster2.predict(
    from: test, outputMargin: true
)

print(predictedTest2)
