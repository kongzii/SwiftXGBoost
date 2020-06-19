// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/predict_leaf_indices.py

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

let leafIndex1 = try booster.predict(
    from: test,
    treeLimit: 2,
    predictionLeaf: true
).makeNumpyArray()
let leafIndex2 = try booster.predict(
    from: test,
    predictionLeaf: true
).makeNumpyArray()

print(leafIndex1)
print(leafIndex2)
