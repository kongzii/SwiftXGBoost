import XCTest

#if !os(macOS)
    public func XGBoostAllTests() -> [XCTestCaseEntry] {
        [
            testCase(ExampleTests.allTests),
            testCase(DMatrixTests.allTests),
            testCase(BoosterTests.allTests),
            testCase(PlotTests.allTests),
            testCase(TensorFlowTests.allTests),
            testCase(XGBoostDocsTests.allTests),
            testCase(TrainTests.allTests),
            testCase(CrossValidationTests.allTests),
            testCase(ArrayTests.allTests),
            testCase(PythonTests.allTests),
        ]
    }
#endif
