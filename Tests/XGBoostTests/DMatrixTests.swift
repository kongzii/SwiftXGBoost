import XCTest

@testable import XGBoost

final class DMatrixTests: XCTestCase {
    func testCounts() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(5, 2),
            threads: 1
        )
        XCTAssertEqual(try data.rowCount(), 5)
        XCTAssertEqual(try data.columnCount(), 2)
        XCTAssertEqual(try data.shape().row, 5)
        XCTAssertEqual(try data.shape().column, 2)
    }

    func testFromCSVFile() throws {
        let csvFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testFromCSVFile.csv", isDirectory: false
        ).path

        let fileContent = """
        1.0,2.0,3.0
        0.0,4.0,5.0
        1.0,6.0,7.0
        """

        try fileContent.write(
            toFile: csvFile,
            atomically: true,
            encoding: .utf8
        )

        let data = try DMatrix(
            name: "test",
            from: csvFile,
            labelColumn: 0
        )

        XCTAssertEqual(try data.label(), [1.0, 0.0, 1.0])
        XCTAssertEqual(try data.rowCount(), 3)
        XCTAssertEqual(try data.columnCount(), 2)
    }

    func testSaveAndLoadBinary() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            label: [1, 0, 1]
        )

        let path = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testSaveAndLoadBinary.dmatrix", isDirectory: false
        ).path

        try data.save(to: path)

        let loadedData = try DMatrix(
            name: "loaded",
            from: path,
            format: .binary
        )

        XCTAssertEqual(try data.label(), try loadedData.label())
        XCTAssertEqual(try data.rowCount(), try loadedData.rowCount())
        XCTAssertEqual(try data.columnCount(), try loadedData.columnCount())
    }

    func testSaveAndLoadFeatureMap() throws {
        let features = [
            Feature("x", .quantitative),
            Feature("y", .quantitative),
        ]

        let path = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testSaveAndLoadFeatureMap.featuremap", isDirectory: false
        ).path

        try features.saveFeatureMap(to: path)

        let loadedFeatures = try [Feature](fromFeatureMap: path)

        XCTAssertEqual(features, loadedFeatures)
    }

    static var allTests = [
        ("testCounts", testCounts),
        ("testFromCSVFile", testFromCSVFile),
        ("testSaveAndLoadBinary", testSaveAndLoadBinary),
        ("testSaveAndLoadFeatureMap", testSaveAndLoadFeatureMap),
    ]
}
