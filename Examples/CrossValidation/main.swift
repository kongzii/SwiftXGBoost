// This example is based on https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py

import Foundation
import PythonKit
import XGBoost

let pandas = Python.import("pandas")

let data = try DMatrix(
    name: "data",
    from: "Examples/Data/agaricus.txt.train",
    format: .libsvm
)

let parameters = [
    Parameter("max_depth", "2"),
    Parameter("eta", "1"),
    Parameter("seed", "0"),
    Parameter("eval_metric", "error"),
    Parameter("objective", "binary:logistic"),
]

let cv1 = try crossValidationTraining(
    data: data,
    splits: 5,
    iterations: 2,
    parameters: parameters,
    shuffle: false
)
print(pandas.DataFrame.from_dict(cv1.results))

let cv2 = try crossValidationTraining(
    data: data,
    splits: 5,
    iterations: 10,
    parameters: parameters,
    earlyStopping: EarlyStopping(
        dataName: "data",
        metricName: "error",
        stoppingRounds: 3

    ),
    shuffle: false
)
print(pandas.DataFrame.from_dict(cv2.results))

let objectiveFunction: ObjectiveFunction = { preds, data in
    let labels = try data.get(field: .label)
    let preds = preds.map { 1.0 / (1.0 + exp(-$0)) }
    let grad = (0 ..< labels.count).map { preds[$0] - labels[$0] }
    let hess = (0 ..< labels.count).map { preds[$0] * (1.0 - preds[$0]) }
    return (grad, hess)
}

let evaluationFunction: EvaluationFunction = { preds, data in
    let labels = try data.get(field: .label)
    let error = (0 ..< labels.count).map {
        (labels[$0] > 0) != (preds[$0] > 0) ? 1.0 : 0.0
    }.sum() / Float(labels.count)

    return ("c-error", "\(error)")
}

let cv3 = try crossValidationTraining(
    data: data,
    splits: 5,
    iterations: 2,
    parameters: parameters,
    objectiveFunction: objectiveFunction,
    evaluationFunction: evaluationFunction,
    shuffle: false
)
print(pandas.DataFrame.from_dict(cv3.results))
