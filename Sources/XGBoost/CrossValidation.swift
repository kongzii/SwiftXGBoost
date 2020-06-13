import CXGBoost
import Foundation

public class CVPack {
    public let train, test: DMatrix
    public let booster: Booster

    public init(
        train: DMatrix,
        test: DMatrix,
        parameters: [Parameter]
    ) throws {
        self.train = train
        self.test = test
        booster = try Booster(
            with: [train, test],
            parameters: parameters
        )
    }
}

func groupsToRows(
    groups: [Int],
    boundaries: [UInt32]
) -> [UInt32] {
    groups.map { g in
        boundaries[g] ..< boundaries[g + 1]
    }.flatMap { $0 }
}

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
                train: train,
                test: test,
                parameters: parameters
            )
        )
    }

    return packs
}

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
            newName: "train"
        )
        let test = try data.slice(
            indexes: outIdSet[k],
            newName: "test"
        )

        packs.append(
            try CVPack(
                train: train,
                test: test,
                parameters: parameters
            )
        )
    }

    return packs
}

func aggregateCrossValidationResults(
    results: [Evaluation]
) throws -> [(metricName: String, mean: Float, std: Float)] {
    var crossDict = [String: [Float]]()

    for result in results {
        for (dataName, dataResults) in result {
            for (metricName, metricValue) in dataResults {
                crossDict["\(dataName)-\(metricName)", or: []].append(Float(metricValue)!)
            }
        }
    }

    var results = [(metricName: String, mean: Float, std: Float)]()
    let crossTuples = crossDict.map { $0 }

    for (metricName, values) in crossTuples {
        let mean = values.mean()
        results.append((
            metricName: metricName,
            mean: mean,
            std: values.std(mean: mean)
        ))
    }

    return results
}

public func crossValidationTraining(
    folds: [CVPack],
    iterations: Int,
    objectiveFunction: ObjectiveFunction? = nil,
    evaluationFunction: EvaluationFunction? = nil,
    beforeIteration: (Booster, Int) throws -> AfterIterationOutput = { _, _ in .next },
    callbacks: [Callback] = [],
    afterIteration: (Booster, Int, Evaluation?, [AfterIterationOutput]) throws -> AfterIterationOutput = { _, _, _, _ in .next }
) throws -> (results: [String: [Float]], folds: [CVPack]) {
    var iterationsEvaluations = [Int: [Evaluation]]()

    for fold in folds {
        let callbacks = callbacks.map { $0.copy() }
        try fold.booster.train(
            iterations: iterations,
            trainingData: fold.train,
            objectiveFunction: objectiveFunction,
            evaluationData: [fold.train, fold.test],
            evaluationFunction: evaluationFunction,
            beforeIteration: beforeIteration,
            callbacks: callbacks
        ) { booster, iteration, evaluation, outputs in
            iterationsEvaluations[iteration, or: []].append(evaluation!)
            return try afterIteration(booster, iteration, evaluation, outputs)
        }
    }

    var cvResults = [String: [Float]]()

    for iteration in 0 ..< iterations {
        guard let evaluations = iterationsEvaluations[iteration] else {
            break
        }

        let aggregatedEvaluation = try aggregateCrossValidationResults(
            results: evaluations
        )

        for (metricName, mean, std) in aggregatedEvaluation {
            cvResults[metricName + "-mean", or: []].append(mean)
            cvResults[metricName + "-std", or: []].append(std)
        }
    }

    return (cvResults, folds)
}

public func crossValidationTraining(
    data: Data,
    splits: Int,
    iterations: Int,
    parameters: [Parameter],
    objectiveFunction: ObjectiveFunction? = nil,
    evaluationFunction: EvaluationFunction? = nil,
    shuffle: Bool = true,
    beforeIteration: (Booster, Int) throws -> AfterIterationOutput = { _, _ in .next },
    callbacks: [Callback] = [],
    afterIteration: (Booster, Int, Evaluation?, [AfterIterationOutput]) throws -> AfterIterationOutput = { _, _, _, _ in .next }
) throws -> (results: [String: [Float]], folds: [CVPack]) {
    try crossValidationTraining(
        folds: try makeNFold(
            data: data,
            splits: splits,
            parameters: parameters,
            shuffle: shuffle
        ),
        iterations: iterations,
        objectiveFunction: objectiveFunction,
        evaluationFunction: evaluationFunction,
        beforeIteration: beforeIteration,
        callbacks: callbacks,
        afterIteration: afterIteration
    )
}
