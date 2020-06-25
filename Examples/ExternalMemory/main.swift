// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py

import Foundation
import XGBoost

let train = try DMatrix(
    name: "train",
    from: "Examples/Data/agaricus.txt.train",
    format: .libsvm,
    useCache: true
)
let test = try DMatrix(
    name: "test",
    from: "Examples/Data/agaricus.txt.test",
    format: .libsvm,
    useCache: true
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
    iterations: 2,
    trainingData: train,
    evaluationData: [train, test]
) { _, iteration, evaluation, _ in
    print(iteration, ":", evaluation!)
    return .next
}
