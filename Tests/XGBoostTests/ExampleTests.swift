import XCTest

@testable import XGBoost

final class ExampleTests: XCTestCase {
    func testReadmeExample1() throws {
        // import XGBoost

        // Register your own callback function for log(info) messages
        try XGBoost.registerLogCallback {
            print("Swifty log:", String(cString: $0!))
        }

        // Create some random features and labels
        let randomArray = (0 ..< 1000).map { _ in Float.random(in: 0 ..< 2) }
        let labels = (0 ..< 100).map { _ in Float([0, 1].randomElement()!) }

        // Initialize data, DMatrixHandle in the background
        let data = try DMatrix(
            name: "data",
            values: randomArray,
            shape: Shape(100, 10),
            label: labels,
            threads: 1
        )

        // Slice array into train and test
        let train = try data.slice(indexes: 0 ..< 90, newName: "train")
        let test = try data.slice(indexes: 90 ..< 100, newName: "test")

        // Parameters for XGBoost, check https://xgboost.readthedocs.io/en/latest/parameter.html
        let parameters: [Parameter] = [
            ("verbosity", "2"),
            ("seed", "0"),
        ]

        // Create XGBoost model, `with` data will be cached
        let xgboost = try XGBoost(
            with: [train, test],
            parameters: parameters
        )

        // Train xgboost, optionally provide callback functions called before and after each iteration
        try xgboost.train(
            iterations: 10,
            trainingData: train,
            evaluationData: [train, test]
        )

        // Predict from test data
        let predictions = try xgboost.predict(from: test)

        // Save
        try xgboost.save(to: "model.xgboost")

        // Assert outputs
        XCTAssertEqual(try data.label(), labels)
        XCTAssertEqual(try data.rowCount(), 100)
        XCTAssertEqual(try data.columnCount(), 10)
        XCTAssertEqual(try train.rowCount(), 90)
        XCTAssertEqual(try train.columnCount(), 10)
        XCTAssertEqual(try test.rowCount(), 10)
        XCTAssertEqual(try test.columnCount(), 10)
    }

    static var allTests = [
        ("testReadmeExample1", testReadmeExample1),
    ]
}
