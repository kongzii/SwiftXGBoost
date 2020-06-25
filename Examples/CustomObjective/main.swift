// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

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
]

// user define objective function, given prediction, return gradient and second
// order gradient this is log likelihood loss
let logRegObjective: ObjectiveFunction = { preds, data in
    let labels = try data.get(field: .label)
    let preds = preds.map { 1.0 / (1.0 + exp(-$0)) }
    let grad = (0 ..< labels.count).map { preds[$0] - labels[$0] }
    let hess = (0 ..< labels.count).map { preds[$0] * (1.0 - preds[$0]) }
    return (grad, hess)
}

// user defined evaluation function, return a pair metric_name, result
//
// NOTE: when you do customized loss function, the default prediction value is
// margin. this may make builtin evaluation metric not function properly for
// example, we are doing logistic loss, the prediction is score before logistic
// transformation the builtin evaluation error assumes input is after logistic
// transformation Take this in mind when you use the customization, and maybe
// you need write customized evaluation function
let evaluationFunction: EvaluationFunction = { preds, data in
    let labels = try data.get(field: .label)
    let error = (0 ..< labels.count).map {
        (labels[$0] > 0) != (preds[$0] > 0) ? 1.0 : 0.0
    }.sum() / Float(labels.count)

    return ("my-error", "\(error)")
}

let booster = try Booster(
    with: [train, test],
    parameters: parameters
)

// training with customized objective, we can also do step by step training
// simply look at Train.swift's implementation of train
try booster.train(
    iterations: 2,
    trainingData: train,
    objectiveFunction: logRegObjective,
    evaluationFunction: evaluationFunction
)

let testPreds = try booster.predict(from: test)
