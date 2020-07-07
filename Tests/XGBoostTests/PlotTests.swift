import XCTest

@testable import XGBoost

final class PlotTests: XCTestCase {
    func testSaveImportanceGraph() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let features = (0 ..< 5).map { Feature(name: "Name-\($0)", type: .quantitative) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 5),
            features: features,
            label: label,
            threads: 1
        )
        let booster = try Booster(with: [data])
        try booster.train(
            iterations: 10,
            trainingData: data
        )

        let graphFile = temporaryFile()
        let featureMapFile = temporaryFile()

        try data.saveFeatureMap(to: featureMapFile)
        try booster.saveImportanceGraph(to: graphFile, featureMap: featureMapFile)

        XCTAssertTrue(FileManager.default.fileExists(atPath: featureMapFile))
        XCTAssertTrue(FileManager.default.fileExists(atPath: graphFile + ".svg"))
    }

    static var allTests = [
        ("testSaveImportanceGraph", testSaveImportanceGraph),
    ]
}
