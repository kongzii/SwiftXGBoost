import XCTest

@testable import XGBoost

final class TrainTests: XCTestCase {
    func testEarlyStoppingMinimize() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 1),
            label: label,
            threads: 1
        )
        let booster = try Booster(
            with: [data],
            parameters: [Parameter(name: "seed", value: "0")]
        )

        var scores = [String]()

        try booster.train(
            iterations: 5000,
            trainingData: data,
            evaluationData: [data],
            earlyStopping: .init(
                dataName: "data",
                metricName: "rmse",
                stoppingRounds: 10,
                verbose: true
            )
        ) { _, _, evaluation in
            scores.append(evaluation!["data"]!["rmse"]!)
            return .next
        }

        XCTAssertTrue(scores.count >= 10)
        XCTAssertTrue(scores.first! >= scores.last!)
        XCTAssertTrue(scores[scores.count - 10 ..< scores.count].allSatisfy { $0 == scores.last! })
        XCTAssertEqual(Double(try booster.attribute(name: "best_score")!)!, Double(scores.last!)!)
    }

    func testEarlyStoppingMaximize() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 1),
            label: label,
            threads: 1
        )
        let booster = try Booster(
            with: [data],
            parameters: [
                Parameter(name: "seed", value: "0"),
                Parameter(name: "eval_metric", value: "rmse"),
                Parameter(name: "eval_metric", value: "auc"),
            ]
        )

        var scores = [String]()

        try booster.train(
            iterations: 5000,
            trainingData: data,
            evaluationData: [data],
            earlyStopping: .init(
                dataName: "data",
                metricName: "auc",
                stoppingRounds: 10,
                verbose: true
            )
        ) { _, _, evaluation in
            scores.append(evaluation!["data"]!["auc"]!)
            return .next
        }

        XCTAssertTrue(scores.count >= 10)
        XCTAssertTrue(scores.first! <= scores.last!)
        XCTAssertTrue(scores[scores.count - 10 ..< scores.count].allSatisfy { $0 == scores.last! })
        XCTAssertEqual(Double(try booster.attribute(name: "best_score")!)!, Double(scores.last!)!)
    }

    static var allTests = [
        ("testEarlyStoppingMinimize", testEarlyStoppingMinimize),
        ("testEarlyStoppingMaximize", testEarlyStoppingMaximize),
    ]
}
