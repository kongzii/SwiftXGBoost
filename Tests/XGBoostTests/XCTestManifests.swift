import XCTest

#if !os(macOS)
    public func XGBoostAllTests() -> [XCTestCaseEntry] {
        [
            testCase(ExampleTests.allTests),
            testCase(DMatrixTests.allTests),
            testCase(XGBoostTests.allTests),
            testCase(PlotTests.allTests),
            testCase(TensorFlowTests.allTests),
        ]
    }
#endif
