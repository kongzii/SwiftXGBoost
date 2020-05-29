import XCTest

@testable import XGBoost

final class PlotTests: XCTestCase {
    func testSaveImportanceGraph() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let features = (0 ..< 5).map { Feature(name: "Name-\($0)", type: .quantitative) }
        let data = try Data(
            name: "data",
            values: randomArray,
            shape: (10, 5),
            label: label,
            features: features,
            threads: 1
        )
        let xgboost = try XGBoost(with: [data])
        try xgboost.train(
            iterations: 10,
            trainingData: data
        )

        let graphFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testSaveImportanceGraph", isDirectory: false
        ).path
        let featureMapFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testSaveImportanceGraph.featureMap", isDirectory: false
        ).path

        try data.saveFeatureMap(to: featureMapFile)
        try xgboost.saveImportanceGraph(fileName: graphFile, featureMap: featureMapFile)

        XCTAssertTrue(FileManager.default.fileExists(atPath: featureMapFile))
        XCTAssertTrue(FileManager.default.fileExists(atPath: graphFile + ".png"))
    }

    static var allTests = [
        ("testSaveImportanceGraph", testSaveImportanceGraph),
    ]
}
