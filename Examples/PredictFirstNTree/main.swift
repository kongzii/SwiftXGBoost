// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/predict_first_ntree.py

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
    Parameter("eta", 1),
    Parameter("max_depth", 2),
    Parameter("objective", "binary:logistic"),
]

let booster = try Booster(
    with: [train, test],
    parameters: parameters
)

try booster.train(
    iterations: 3,
    trainingData: train
)

let labels = try test.get(field: .label)

// Predict using first 1 tree
let yPred1 = try booster.predict(from: test, treeLimit: 1)
let error1 = zip(yPred1, labels).map {
    ($0 > 0.5) != ($1 == 1) ? 1 : 0
}.sum() / Float(labels.count)

// By default, we predict using all the trees
let yPred2 = try booster.predict(from: test)
let error2 = zip(yPred2, labels).map {
    ($0 > 0.5) != ($1 == 1) ? 1 : 0
}.sum() / Float(labels.count)

print(
    """
    Error yPred1: \(error1)
    Error yPred2: \(error2)
    """
)
