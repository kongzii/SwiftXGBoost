import CXGBoost
import PythonKit
import XCTest

#if canImport(TensorFlow)
    import TensorFlow
#endif

@testable import XGBoost

final class XGBoostDocsTests: XCTestCase {
    func testXGBoostIntroDoc() throws {
        let numpy = Python.import("numpy")
        let pandas = Python.import("pandas")

        let dataFrame = pandas.read_csv("Examples/Data/veterans_lung_cancer.csv")

        #if canImport(TensorFlow)
            let tensor = Tensor<Float>(shape: TensorShape([2, 3]), scalars: [1, 2, 3, 4, 5, 6])
            let tensorData = try DMatrix(name: "tensorData", from: tensor)
        #endif

        let svmData = try DMatrix(name: "train", from: "Examples/Data/data.svm.txt", format: .libsvm)

        let csvData = try DMatrix(name: "train", from: "Examples/Data/data.csv", format: .csv, labelColumn: 0)

        let numpyData = try DMatrix(name: "train", from: numpy.random.rand(5, 10), label: numpy.random.randint(2, size: 5))

        let pandasDataFrame = pandas.DataFrame(numpy.arange(12).reshape([4, 3]), columns: ["a", "b", "c"])
        let pandasLabel = numpy.random.randint(2, size: 4)
        let pandasData = try DMatrix(name: "data", from: pandasDataFrame.values, label: pandasLabel)

        try pandasData.save(to: "train.buffer")

        let dataWithMissingValues = try DMatrix(name: "data", from: pandasDataFrame.values, missingValue: 999.0)

        try dataWithMissingValues.set(field: .weight, values: [Float](repeating: 1, count: try dataWithMissingValues.rowCount()))

        let labelsFromData = try pandasData.get(field: .label)

        let firstBooster = try Booster()
        try firstBooster.set(parameter: "tree_method", value: "hist")

        let parameters = [Parameter(name: "tree_method", value: "hist")]
        let secondBooster = try Booster(parameters: parameters)

        let trainingData = try DMatrix(name: "train", from: "Examples/Data/data.csv", format: .csv, labelColumn: 0)
        let boosterWithCachedData = try Booster(with: [trainingData])
        try boosterWithCachedData.train(iterations: 5, trainingData: trainingData)

        try boosterWithCachedData.save(to: "0001.xgboost")

        let textModel = try boosterWithCachedData.dumped(format: .text)

        let loadedBooster = try Booster(from: "0001.xgboost")

        let testDataNumpy = try DMatrix(name: "test", from: numpy.random.rand(7, 12))
        let predictionNumpy = try loadedBooster.predict(from: testDataNumpy)

        let testData = try DMatrix(name: "test", from: [69.0, 60.0, 7.0, 0, 0, 0, 1, 1, 0, 1, 0, 0], shape: Shape(1, 12))
        let prediction = try loadedBooster.predict(from: testData)

        try boosterWithCachedData.saveImportanceGraph(to: "importance") // .svg extension will be added

        try safe { XGBoosterSaveModel(boosterWithCachedData.booster, "0002.xgboost") }
    }

    static var allTests = [
        ("testXGBoostIntroDoc", testXGBoostIntroDoc),
    ]
}
