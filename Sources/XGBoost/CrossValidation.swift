import CXGBoost
import Foundation

/// Auxiliary class to hold one fold of CrossValidation.
public class CVPack {
    public let id: String
    public let train, test: DMatrix
    public let booster: Booster

    /// Initialization of CVPack. Booster will get attribute `cvpack_id` with value of `id`.
    ///
    /// - Parameter id: Identifier for this fold.
    /// - Parameter train: Training DMatrix.
    /// - Parameter test: Testing DMatrix.
    /// - Parameter parameters: Parameters pased to Booster.
    public init(
        id: String,
        train: DMatrix,
        test: DMatrix,
        parameters: [Parameter]
    ) throws {
        self.id = id
        self.train = train
        self.test = test
        booster = try Booster(
            with: [train, test],
            parameters: parameters
        )

        try booster.set(attribute: "cvpack_id", value: self.id)
    }
}

/// Given group row boundaries, convert ground indexes to row indexes
///
/// - Parameter groups: Array of groups.
/// - Parameter boundaries: Rows index limits of each group.
/// - Returns: Row in group.
func groupsToRows(
    groups: [Int],
    boundaries: [UInt32]
) -> [UInt32] {
    groups.map { g in
        boundaries[g] ..< boundaries[g + 1]
    }.flatMap { $0 }
}

/// Make folds for cross-validation maintaining groups.
///
/// - Parameter data: DMatrix data.
/// - Parameter splits: Number of folds to create.
/// - Parameter parameters: Parameters passed to Booster in CVPack.
/// - Parameter shuffle: If true, rows in folds will be shuffled.
/// - Returns: Array of CVPacks.
func makeGroupNFold(
    data: Data,
    splits: Int,
    parameters: [Parameter],
    shuffle: Bool
) throws -> [CVPack] {
    let groupBoundaries = try data.get(field: .groupPtr)
    let groupSizes = groupBoundaries.diff()

    var idx = Array(0 ..< groupSizes.count)

    if shuffle {
        idx.shuffle()
    }

    let outGroupIdSet = idx.chunked(into: splits)
    let inGroupIdSet = (0 ..< splits).map { k in
        (0 ..< splits).filter { i in i != k }.flatMap { i in
            outGroupIdSet[i]
        }
    }

    let outIdSet = outGroupIdSet.map {
        groupsToRows(groups: $0, boundaries: groupBoundaries).map { Int32($0) }
    }
    let inIdSet = inGroupIdSet.map {
        groupsToRows(groups: $0, boundaries: groupBoundaries).map { Int32($0) }
    }

    var packs = [CVPack]()

    for k in 0 ..< splits {
        let train = try data.slice(
            indexes: inIdSet[k],
            allowGroups: true
        )
        let trainGroupSizes = inGroupIdSet[k].map { groupSizes[Int($0)] }
        try train.set(
            field: .group,
            values: trainGroupSizes
        )
        let test = try data.slice(
            indexes: outIdSet[k],
            allowGroups: true
        )
        let testGroupSizes = outGroupIdSet[k].map { groupSizes[Int($0)] }
        try test.set(
            field: .group,
            values: testGroupSizes
        )
        packs.append(
            try CVPack(
                id: "\(k)",
                train: train,
                test: test,
                parameters: parameters
            )
        )
    }

    return packs
}

/// Make folds for cross-validation.
///
/// - Parameter data: DMatrix data.
/// - Parameter splits: Number of folds to create.
/// - Parameter parameters: Parameters passed to Booster in CVPack.
/// - Parameter shuffle: If true, rows in folds will be shuffled.
/// - Returns: Array of CVPacks.
func makeNFold(
    data: Data,
    splits: Int,
    parameters: [Parameter],
    shuffle: Bool
) throws -> [CVPack] {
    if try data.get(field: .groupPtr).count > 1 {
        return try makeGroupNFold(
            data: data,
            splits: splits,
            parameters: parameters,
            shuffle: shuffle
        )
    }

    var idx = try Array(0 ..< Int32(data.rowCount()))

    if shuffle {
        idx.shuffle()
    }

    let outIdSet = idx.chunked(into: splits)
    let inIdSet = (0 ..< splits).map { k in
        (0 ..< splits).filter { i in i != k }.flatMap { i in
            outIdSet[i]
        }
    }

    var packs = [CVPack]()

    for k in 0 ..< splits {
        let train = try data.slice(
            indexes: inIdSet[k],
            newName: "\(data.name)-train"
        )
        let test = try data.slice(
            indexes: outIdSet[k],
            newName: "\(data.name)-test"
        )

        packs.append(
            try CVPack(
                id: "\(k)",
                train: train,
                test: test,
                parameters: parameters
            )
        )
    }

    return packs
}

/// - Parameter results: Array of Evaluation.
/// - Returns: Metrics for cross-validation.
func aggregateCrossValidationResults(
    results: [Evaluation]
) throws -> [(metricName: String, mean: Float, std: Float)] {
    var crossDict = CVEvaluation()

    for result in results {
        for (dataName, dataResults) in result {
            for (metricName, metricValue) in dataResults {
                crossDict["\(dataName)-\(metricName)", or: []].append(Float(metricValue)!)
            }
        }
    }

    var results = [(metricName: String, mean: Float, std: Float)]()

    for (metricName, values) in crossDict {
        let mean = values.mean()
        results.append((
            metricName: metricName,
            mean: mean,
            std: values.std(mean: mean)
        ))
    }

    return results
}

/// Typealias for dictionary of cross-validation evaluations.
public typealias CVEvaluation = [String: [Float]]

/// Typealias for function called after CV iteration.
public typealias AfterCVIteration = (Int, CVEvaluation, Bool) throws -> AfterIterationOutput

/// Default function called after CV iteration.
public let DefaultAfterCVIteration: AfterCVIteration = { _, _, _ in .next }

/// Training with cross-validation.
///
/// - Parameter folds: Cross-validation folds to train.
/// - Parameter iterations: Number training of iterations.
/// - Parameter earlyStopping: Early stopping used for cross-validation results.
/// - Parameter objectiveFunction: Custom objective function passed to the booster.train.
/// - Parameter evaluationFunction: Custom evaluation function passed to the booster.train.
/// - Parameter beforeIteration: Before iteration function passed to the booster.train.
/// - Parameter callbacks: Callbacks passed to the booster.train.
/// - Parameter afterIteration: After iteration function passed to the booster.train.
/// - Parameter afterCVIteration: Function called after one cv iteration, e.g. after one training iteration of every fold.
public func crossValidationTraining(
    folds: [CVPack],
    iterations: Int,
    earlyStopping: EarlyStopping? = nil,
    objectiveFunction: ObjectiveFunction? = nil,
    evaluationFunction: EvaluationFunction? = nil,
    beforeIteration: BeforeIteration = DefaultBeforeIteration,
    callbacks: [Callback] = [],
    afterIteration: AfterIteration = DefaultAfterIteration,
    afterCVIteration: AfterCVIteration = DefaultAfterCVIteration
) throws -> (results: CVEvaluation, folds: [CVPack]) {
    var cvResults = CVEvaluation()

    training: for iteration in 0 ..< iterations {
        var stopped = 0
        var iterationEvaluations = [Evaluation]()

        for (index, fold) in folds.enumerated() {
            try fold.booster.train(
                iterations: iteration + 1,
                startIteration: iteration,
                trainingData: fold.train,
                objectiveFunction: objectiveFunction,
                evaluationData: [fold.train, fold.test],
                evaluationFunction: evaluationFunction,
                beforeIteration: beforeIteration,
                callbacks: callbacks
            ) { booster, iteration, evaluation, outputs in
                if outputs.willStop {
                    stopped += 1
                }

                iterationEvaluations.append(evaluation!)
                return try afterIteration(booster, iteration, evaluation, outputs)
            }
        }

        let aggregatedEvaluation = try aggregateCrossValidationResults(
            results: iterationEvaluations
        )
        for (metricName, mean, std) in aggregatedEvaluation {
            cvResults[metricName + "-mean", or: []].append(mean)
            cvResults[metricName + "-std", or: []].append(std)
        }

        let earlyStop: Bool = try {
            if let earlyStopping = earlyStopping {
                return try earlyStopping.call(
                    iteration: iteration,
                    evaluation: cvResults
                ) == .stop
            }

            return false
        }()

        let willStop = earlyStop || stopped == folds.count

        switch try afterCVIteration(
            iteration,
            cvResults,
            willStop
        ) {
        case .stop:
            break training
        case .next:
            break
        }

        if willStop {
            break training
        }
    }

    if let earlyStopping = earlyStopping {
        for (key, values) in cvResults {
            cvResults[key] = Array(values[0 ... earlyStopping.state.bestIteration])
        }
    }

    return (cvResults, folds)
}

/// Training with cross-validation.
///
/// - Parameter data: Data to split.
/// - Parameter splits: Number of cvfoolds that will be created.
/// - Parameter iterations: Number training of iterations.
/// - Parameter parameters: Parameters passed to the Booster initialized in CVPack.
/// - Parameter earlyStopping: Early stopping used for cross-validation results.
/// - Parameter objectiveFunction: Custom objective function passed to the booster.train.
/// - Parameter evaluationFunction: Custom evaluation function passed to the booster.train.
/// - Parameter beforeIteration: Before iteration function passed to the booster.train.
/// - Parameter callbacks: Callbacks passed to the booster.train.
/// - Parameter afterIteration: After iteration function passed to the booster.train.
/// - Parameter afterCVIteration: Function called after one cv iteration, e.g. after one training iteration of every fold.
public func crossValidationTraining(
    data: Data,
    splits: Int,
    iterations: Int,
    parameters: [Parameter],
    earlyStopping: EarlyStopping? = nil,
    objectiveFunction: ObjectiveFunction? = nil,
    evaluationFunction: EvaluationFunction? = nil,
    shuffle: Bool = true,
    beforeIteration: BeforeIteration = DefaultBeforeIteration,
    callbacks: [Callback] = [],
    afterIteration: AfterIteration = DefaultAfterIteration,
    afterCVIteration: AfterCVIteration = DefaultAfterCVIteration
) throws -> (results: CVEvaluation, folds: [CVPack]) {
    try crossValidationTraining(
        folds: try makeNFold(
            data: data,
            splits: splits,
            parameters: parameters,
            shuffle: shuffle
        ),
        iterations: iterations,
        earlyStopping: earlyStopping,
        objectiveFunction: objectiveFunction,
        evaluationFunction: evaluationFunction,
        beforeIteration: beforeIteration,
        callbacks: callbacks,
        afterIteration: afterIteration,
        afterCVIteration: afterCVIteration
    )
}
