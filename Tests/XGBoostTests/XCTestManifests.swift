import XCTest

#if !os(macOS)
    public func XGBoostAllTests() -> [XCTestCaseEntry] {
        [
            testCase(ExampleTests.allTests),
            testCase(DataTests.allTests),
            testCase(XGBoostTests.allTests),
            testCase(PlotTests.allTests),
        ]
    }
#endif
