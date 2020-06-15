import PythonKit
import XCTest

@testable import XGBoost

private let PXGBOOST = Python.import("xgboost")
private let PJSON = Python.import("json")

final class CrossValidationTests: XCTestCase {
    func testBasicCrossValidation() throws {
        let randomArray = (0 ..< 10000).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 1000).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(1000, 10),
            label: label,
            threads: 1
        )

        let temporaryDataFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testBasicCrossValidation.data", isDirectory: false
        ).path

        try data.save(to: temporaryDataFile)
        let pyData = PXGBOOST.DMatrix(data: temporaryDataFile)

        let (results, folds) = try crossValidationTraining(
            data: data,
            splits: 5,
            iterations: 10,
            parameters: [
                Parameter("seed", 0),
            ],
            shuffle: false
        )

        let pyJsonResults = PXGBOOST.cv(
            params: ["seed": 0],
            dtrain: pyData,
            nfold: 5,
            num_boost_round: 10,
            as_pandas: false,
            seed: 0,
            shuffle: false
        )

        var parsedPyJsonResults = [String: [Float]]()
        for name in pyJsonResults {
            parsedPyJsonResults["data-\(name)"] = [Float](pyJsonResults[name])!
        }

        assertEqual(results, parsedPyJsonResults, accuracy: 1e-6)
    }

    func testEarlyStoppingCrossValidation() throws {
        let randomArray = (0 ..< 10000).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 1000).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(1000, 10),
            label: label,
            threads: 1
        )

        let temporaryDataFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testBasicCrossValidation.data", isDirectory: false
        ).path

        try data.save(to: temporaryDataFile)
        let pyData = PXGBOOST.DMatrix(data: temporaryDataFile)

        let (results, folds) = try crossValidationTraining(
            data: data,
            splits: 5,
            iterations: 10,
            parameters: [
                Parameter("seed", 0),
            ],
            earlyStopping: EarlyStopping(
                dataName: "data",
                metricName: "rmse",
                stoppingRounds: 3
            ),
            shuffle: false
        )

        let pyJsonResults = PXGBOOST.cv(
            params: ["seed": 0],
            dtrain: pyData,
            nfold: 5,
            num_boost_round: 10,
            as_pandas: false,
            seed: 0,
            callbacks: [PXGBOOST.callback.early_stop(3)],
            shuffle: false
        )

        var parsedPyJsonResults = [String: [Float]]()
        for name in pyJsonResults {
            parsedPyJsonResults["data-\(name)"] = [Float](pyJsonResults[name])!
        }

        assertEqual(results, parsedPyJsonResults, accuracy: 1e-7)
    }

    static var allTests = [
        ("testBasicCrossValidation", testBasicCrossValidation),
        ("testEarlyStoppingCrossValidation", testEarlyStoppingCrossValidation),
    ]
}
