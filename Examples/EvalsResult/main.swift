// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_rmsle.py

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
    Parameter("max_depth", 2),
    Parameter("objective", "binary:logistic"),
    Parameter("eval_metric", "logloss"),
    Parameter("eval_metric", "error"),
]

let booster = try Booster(
    with: [train, test],
    parameters: parameters
)

try booster.train(
    iterations: 2,
    trainingData: train,
    evaluationData: [train, test]
) { _, _, evaluation, _ in
    // Note: `evaluation` will be `nil`, if `evaluationData` are not provided.
    print("Training logloss:", evaluation![train.name]!["logloss"]!)
    print("Training error:", evaluation![train.name]!["error"]!)
    print("Testing logloss:", evaluation![test.name]!["logloss"]!)
    print("Testing error:", evaluation![test.name]!["error"]!)
    print("---")

    // Note: This callback can return `.stop` to break training loop.
    return .next
}
