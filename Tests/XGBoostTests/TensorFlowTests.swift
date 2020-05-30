import XCTest

#if canImport(TensorFlow)
    import TensorFlow
#endif

@testable import XGBoost

final class TensorFlowTests: XCTestCase {
    #if canImport(TensorFlow)
        func testDataFromTensor() throws {
            let x = Tensor<Float>(shape: TensorShape([2, 3]), scalars: [1, 2, 3, 4, 5, 6])
            let data = try Data(name: "test", from: x)
            XCTAssertEqual(try data.rowCount(), 2)
            XCTAssertEqual(try data.columnCount(), 3)
        }

        static var allTests = [
            ("testDataFromTensor", testDataFromTensor),
        ]
    #else
        static var allTests = [(String, (XCTestCase) -> () -> Void)]()
    #endif
}
