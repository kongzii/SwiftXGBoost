/// This example is based on https://github.com/dmlc/xgboost/blob/2c1a439869506532f48d387e7d39beeab358c76b/demo/aft_survival/aft_survival_demo.py.

import Foundation
import PythonKit
import XGBoost

let skLearnModelSelection = try Python.attemptImport("sklearn.model_selection")
let pandas = try Python.attemptImport("pandas")

// Load data from CSV using Pandas
let dataFrame = pandas.read_csv("Examples/Data/veterans_lung_cancer.csv")

// Split features and labels
let yLowerBound = dataFrame["Survival_label_lower_bound"]
let yUpperBound = dataFrame["Survival_label_upper_bound"]

// TODO: Replace when following bug is fixed (Padnas drop function is in collision with Swift function)
// Incorrect argument label in call (have 'columns:', expected 'while:')
// Swift.Collection:5:40: note: 'drop(while:)' declared here
// @inlinable public __consuming func drop(while predicate: (Self.Element) throws -> Bool) rethrows -> Self.SubSequence

// ! let X = dataFrame.drop(columns: ["Survival_label_lower_bound", "Survival_label_upper_bound"])
let X = Python.getattr(dataFrame, "drop")(columns: ["Survival_label_lower_bound", "Survival_label_upper_bound"])

// Split data into training and validation sets
let rs = skLearnModelSelection.ShuffleSplit(n_splits: 2, test_size: 0.7, random_state: 0)

// ! let splitted = Python.next(rs.split(X))
let splitted = Python.next(Python.getattr(rs, "split")(X))

let (trainIndex, validIndex) = (splitted[0], splitted[1])

let trainingData = try DMatrix(
    name: "training",
    from: X.values[trainIndex]
)

try trainingData.set(field: .labelLowerBound, values: [Float](yLowerBound.values[trainIndex])!)
try trainingData.set(field: .labelUpperBound, values: [Float](yUpperBound.values[trainIndex])!)

let validationData = try DMatrix(
    name: "validation",
    from: X.values[validIndex]
)

try validationData.set(field: .labelLowerBound, values: [Float](yLowerBound.values[validIndex])!)
try validationData.set(field: .labelUpperBound, values: [Float](yUpperBound.values[validIndex])!)

// Train gradient boosted trees using AFT loss and metric
let parameters: [Parameter] = [
    ("verbosity", "0"),
    ("objective", "survival:aft"),
    ("eval_metric", "aft-nloglik"),
    ("tree_method", "hist"),
    ("learning_rate", "0.05"),
    ("aft_loss_distribution", "normal"),
    ("aft_loss_distribution_scale", "1.20"),
    ("max_depth", "6"),
    ("lambda", "0.01"),
    ("alpha", "0.02"),
]

let booster = try Booster(
    with: [trainingData, validationData],
    parameters: parameters
)

try booster.train(
    iterations: 10000,
    trainingData: trainingData,
    evaluationData: [trainingData, validationData]
) { _, iteration, evaluation in
    print("Traning stats at \(iteration): \(evaluation!["training"]!)")
    print("Validation stats at \(iteration): \(evaluation!["validation"]!)")
    return .next
}

let predicted = try booster.predict(
    from: validationData
)

let compare = pandas.DataFrame([
    "Label lower bound": yLowerBound[validIndex],
    "Label upper bound": yUpperBound[validIndex],
    "Prediced": predicted.makeNumpyArray(),
])

print(compare)

try booster.save(to: "aft_model.xgboost")
