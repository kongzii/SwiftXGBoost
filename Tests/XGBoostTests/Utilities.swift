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

func randomDMatrix(
    rows: Int = 10,
    columns: Int = 5,
    threads: Int = 1
) throws -> DMatrix {
    let randomArray = (0 ..< rows * columns).map { _ in Float.random(in: 0 ..< 2) }
    let label = (0 ..< rows).map { _ in Float([0, 1].randomElement()!) }
    let data = try DMatrix(
        name: "data",
        from: randomArray,
        shape: Shape(rows, columns),
        label: label,
        threads: threads
    )
    return data
}

func randomTrainedBooster(
    iterations: Int = 1
) throws -> Booster {
    let data = try randomDMatrix()
    let booster = try Booster(
        with: [data],
        parameters: [Parameter("tree_method", "exact")]
    )
    try booster.train(iterations: iterations, trainingData: data)

    return booster
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

    return pyXgboost.Booster(
        model_file: temporaryModelFile,
        params: ["tree_method": "exact"]
    )
}

func python(
    dmatrix: DMatrix
) throws -> PythonObject {
    let temporaryDataFile = temporaryFile()
    try dmatrix.save(to: temporaryDataFile)
    let pyXgboost = try Python.attemptImport("xgboost")

    return pyXgboost.DMatrix(data: temporaryDataFile)
}
