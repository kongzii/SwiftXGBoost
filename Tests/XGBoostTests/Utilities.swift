import PythonKit
import XCTest

@testable import XGBoost

func assertEqual(
    _ a: [String: Float],
    _ b: [String: Float],
    accuracy: Float
) {
    for (key, value) in a {
        XCTAssertNotNil(b[key])
        XCTAssertEqual(value, b[key]!, accuracy: accuracy)
    }
}

func assertEqual(
    _ a: [String: [Float]],
    _ b: [String: [Float]],
    accuracy: Float
) {
    for (key, values) in a {
        XCTAssertNotNil(b[key])
        for (valueA, valueB) in zip(values, b[key]!) {
            XCTAssertEqual(valueA, valueB, accuracy: accuracy)
        }
    }
}

func temporaryFile(name: String = "xgboost") -> String {
    FileManager.default.temporaryDirectory.appendingPathComponent(
        "\(name).\(Int.random(in: 0 ..< Int.max))", isDirectory: false
    ).path
}

func pythonXGBoost() throws -> PythonObject {
    try Python.attemptImport("xgboost")
}

func python(
    booster: Booster
) throws -> PythonObject {
    let temporaryModelFile = temporaryFile()
    try booster.save(to: temporaryModelFile)
    let pyXgboost = try Python.attemptImport("xgboost")

    return pyXgboost.Booster(model_file: temporaryModelFile)
}

func python(
    dmatrix: DMatrix
) throws -> PythonObject {
    let temporaryDataFile = temporaryFile()
    try dmatrix.save(to: temporaryDataFile)
    let pyXgboost = try Python.attemptImport("xgboost")

    return pyXgboost.DMatrix(data: temporaryDataFile)
}
