import XCTest

@testable import XGBoost

final class ExampleTests: XCTestCase {
    func readmeExample1() throws {
        // import XGBoost

        // Create some random features and labels
        let randomArray = (0 ..< 1000).map { _ in Float.random(in: 0 ..< 2) }
        let labels = (0 ..< 100).map { _ in Float([0, 1].randomElement()!) }

        // Initialize data, DMatrixHandle in the background
        let data = try Data(
            name: "data",
            values: randomArray,
            rowCount: 100,
            columnCount: 10,
            threads: 1
        )

        // Set labels
        try data.setFloatInfo(field: .label, values: labels)

        // Slice array into train and test
        let train = try data.slice(indexes: 0 ..< 90, newName: "train")
        let test = try data.slice(indexes: 90 ..< 100, newName: "test")

        // Parameters for XGBoost, check https://xgboost.readthedocs.io/en/latest/parameter.html
        let parameters: [Parameter] = [
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

        // Assert outputs
        XCTAssertEqual(try data.getFloatInfo(field: .label), labels)
        XCTAssertEqual(try data.getRowCount(), 100)
        XCTAssertEqual(try data.getColumnCount(), 10)
        XCTAssertEqual(try train.getRowCount(), 90)
        XCTAssertEqual(try train.getColumnCount(), 10)
        XCTAssertEqual(try test.getRowCount(), 10)
        XCTAssertEqual(try test.getColumnCount(), 10)
    }

    static var allTests = [
        ("readmeExample1", readmeExample1),
    ]
}
