import CXGBoost
import Foundation

/// Dictionary for evaluation in form [data_name: [metric_name: value]]
public typealias Evaluation = [String: [String: String]]

/// Specify if should be executed at beginning of iteration or at the end
public enum LoopPosition {
    case before, after
}

/// Protocol for classes and structs that can be passed to traning in callbacks array.
public protocol Callback {
    // Name of the callback
    var name: String { get }

    /// Time when to execute the callback in training loop.
    var execute: [LoopPosition] { get }

    /// - Parameter booster: Optional booster.
    /// - Parameter iteration: Current iteration.
    /// - Parameter evaluation: Dictionary with evaluations.
    func call(
        booster: Booster?,
        iteration: Int,
        evaluation: Evaluation?
    ) throws -> AfterIterationOutput

    /// - Returns: Copy of itself
    func copy() -> Self
}

public class EarlyStopping: Callback {
    public var name: String = "EarlyStopping"
    public var execute: [LoopPosition] = [.after]

    public typealias State = (
        maximizeScore: Bool,
        bestIteration: Int,
        bestScore: Float,
        bestMsg: String
    )

    public var state: State
    public var dataName: String
    public var metricName: String
    public var stoppingRounds: Int
    public var verbose: Bool

    static func formatMessage(
        bestIteration: Int,
        bestEvaluation: [String: [String: String]]
    ) -> String {
        "Best iteration: \(bestIteration), best evaluation: \(bestEvaluation)."
    }

    static func shouldMaximize(
        maximize: Bool,
        metricName: String
    ) -> Bool {
        var maximizeScore = maximize
        let maximizeMetrics = ["auc", "aucpr", "map", "ndcg"]
        let maximizeAtNMetrics = ["auc@", "aucpr@", "map@", "ndcg@"]

        let metric = metricName.split(separator: "-", maxSplits: 1).last!

        if maximizeAtNMetrics.contains(where: { metric.hasPrefix($0) }) {
            maximizeScore = true
        }

        if maximizeMetrics.contains(where: { $0 == metric.split(separator: ":").first! }) {
            maximizeScore = true
        }

        return maximizeScore
    }

    /// - Parameter dataName: Name of data used for early stopping.
    /// - Parameter metricName: Metric to look for.
    /// - Parameter stoppingRounds: Number of rounds to check improvence for.
    /// - Parameter state: Initial state.
    /// - Parameter verbose: Print on new best or stopping.
    public required init(
        dataName: String,
        metricName: String,
        stoppingRounds: Int,
        state: State,
        verbose: Bool = false
    ) {
        self.dataName = dataName
        self.metricName = metricName
        self.stoppingRounds = stoppingRounds
        self.verbose = verbose
        self.state = state
    }

    /// - Parameter dataName: Name of data used for early stopping.
    /// - Parameter metricName: Metric to look for.
    /// - Parameter stoppingRounds: Number of rounds to check improvence for.
    /// - Parameter maximize: If metric should be maximized, minimzed otherwise.
    /// - Parameter verbose: Print on new best or stopping.
    public convenience init(
        dataName: String,
        metricName: String,
        stoppingRounds: Int,
        maximize: Bool = false,
        verbose: Bool = false
    ) {
        let maximizeScore = EarlyStopping.shouldMaximize(maximize: maximize, metricName: metricName)
        self.init(
            dataName: dataName,
            metricName: metricName,
            stoppingRounds: stoppingRounds,
            state: (
                maximizeScore: maximizeScore,
                bestIteration: 0,
                bestScore: maximizeScore ? -Float.greatestFiniteMagnitude : Float.greatestFiniteMagnitude,
                bestMsg: EarlyStopping.formatMessage(bestIteration: 0, bestEvaluation: [:])
            ),
            verbose: verbose
        )
    }

    /// - Parameter dataName: Name of data used for early stopping.
    /// - Parameter metricName: Metric to look for.
    /// - Parameter stoppingRounds: Number of rounds to check improvence for.
    /// - Parameter maximize: If metric should be maximized, minimzed otherwise.
    /// - Parameter booster: Booster to load state from.
    /// - Parameter verbose: Print on new best or stopping.
    public convenience init(
        dataName: String,
        metricName: String,
        stoppingRounds: Int,
        maximize: Bool,
        booster: Booster,
        verbose: Bool = false
    ) throws {
        let maximizeScore = EarlyStopping.shouldMaximize(maximize: maximize, metricName: metricName)

        var state = (
            maximizeScore: maximizeScore,
            bestIteration: 0,
            bestScore: maximizeScore ? -Float.greatestFiniteMagnitude : Float.greatestFiniteMagnitude,
            bestMsg: EarlyStopping.formatMessage(bestIteration: 0, bestEvaluation: [:])
        )

        if let bestIteration = try booster.attribute(name: "best_iteration") {
            state.bestIteration = Int(bestIteration)!
        }

        if let bestScore = try booster.attribute(name: "best_score") {
            state.bestScore = Float(bestScore)!
        }

        if let bestMsg = try booster.attribute(name: "best_msg") {
            state.bestMsg = bestMsg
        }

        self.init(
            dataName: dataName,
            metricName: metricName,
            stoppingRounds: stoppingRounds,
            state: state,
            verbose: verbose
        )
    }

    /// - Parameter booster: Booster.
    /// - Parameter iteration: Current iteration.
    /// - Parameter evaluation: Dictionary with evaluations.
    public func call(
        booster: Booster? = nil,
        iteration: Int,
        evaluation: Evaluation?
    ) throws -> AfterIterationOutput {
        guard let evaluation = evaluation else {
            throw ValueError.runtimeError("Evaluation data can not be nil.")
        }

        guard let data = evaluation[dataName] else {
            throw ValueError.runtimeError("Name of DMatrix \(dataName) not found in evaluation.")
        }

        guard let stringScore = data[metricName] else {
            throw ValueError.runtimeError("Name of metric \(metricName) not found in evaluation.")
        }

        let score = Float(stringScore)!

        if (state.maximizeScore && score > state.bestScore) || (!state.maximizeScore && score < state.bestScore) {
            state.bestScore = score
            state.bestIteration = iteration
            state.bestMsg = EarlyStopping.formatMessage(bestIteration: iteration, bestEvaluation: evaluation)

            if let booster = booster {
                try booster.set(attribute: "best_score", value: String(state.bestScore))
                try booster.set(attribute: "best_iteration", value: String(state.bestIteration))
                try booster.set(attribute: "best_msg", value: state.bestMsg)
            }

            if verbose {
                log(state.bestMsg)
            }
        } else if iteration - state.bestIteration >= stoppingRounds {
            if verbose {
                log("Stopping at iteration \(iteration): " + state.bestMsg)
            }

            return .stop
        }

        return .next
    }

    public func copy() -> Self {
        .init(
            dataName: dataName,
            metricName: metricName,
            stoppingRounds: stoppingRounds,
            state: state,
            verbose: verbose
        )
    }
}

public class VariableLearningRate: Callback {
    public var name: String = "VariableLearningRate"
    public var execute: [LoopPosition] = [.before]

    public typealias Function = (Int, Int) -> String

    var iterations: Int
    var learningRates: [String]?
    var learningRateFunction: Function?

    required init(
        learningRateFunction: Function?,
        learningRates: [String]?,
        iterations: Int
    ) {
        precondition(
            iterations == (learningRates?.count ?? iterations),
            "Learning rates count must be equal to iterations."
        )

        self.learningRateFunction = learningRateFunction
        self.learningRates = learningRates
        self.iterations = iterations
    }

    public init(
        learningRates: [String],
        iterations: Int
    ) {
        precondition(
            iterations == learningRates.count,
            "Learning rates count must be equal to iterations."
        )

        self.learningRates = learningRates
        self.iterations = iterations
    }

    public init(
        learningRate: @escaping Function,
        iterations: Int
    ) {
        learningRateFunction = learningRate
        self.iterations = iterations
    }

    /// - Parameter booster: Booster.
    /// - Parameter iteration: Current iteration.
    /// - Parameter evaluation: Dictionary with evaluations.
    public func call(
        booster: Booster?,
        iteration: Int,
        evaluation _: Evaluation?
    ) throws -> AfterIterationOutput {
        let newLearningRate: String = {
            if let rates = learningRates {
                return rates[iteration]
            } else {
                return learningRateFunction!(iteration, iterations)
            }
        }()

        guard let booster = booster else {
            throw ValueError.runtimeError("Booster is required to set learning rate.")
        }

        try booster.set(parameter: "learning_rate", value: newLearningRate)

        return .next
    }

    public func copy() -> Self {
        .init(
            learningRateFunction: learningRateFunction,
            learningRates: learningRates,
            iterations: iterations
        )
    }
}

extension Booster {
    /// Train booster.
    ///
    /// - Parameter iterations: Number of training iterations, but training can be stopped early.
    /// - Parameter startIteration: N. of starting iteration.
    /// - Parameter trainingData: Data to train on.
    /// - Parameter evaluationData: Data to evaluate on, if provided.
    /// - Parameter earlyStopping: Early stopping.
    /// - Parameter beforeIteration: Callback called before each iteration.
    /// - Parameter afterIteration: Callback called after each iteration.
    public func train(
        iterations: Int,
        startIteration: Int? = nil,
        trainingData: Data,
        objectiveFunction: ObjectiveFunction? = nil,
        evaluationData: [Data] = [],
        evaluationFunction: EvaluationFunction? = nil,
        beforeIteration: (Booster, Int) throws -> AfterIterationOutput = { _, _ in .next },
        callbacks: [Callback] = [],
        afterIteration: (Booster, Int, Evaluation?, [AfterIterationOutput]) throws -> AfterIterationOutput = { _, _, _, _ in .next }
    ) throws {
        var version = try loadRabitCheckpoint()
        let startIteration = startIteration ?? Int(version / 2)

        training: for iteration in startIteration ..< iterations {
            var outputs = [AfterIterationOutput]()

            switch try beforeIteration(
                self,
                iteration
            ) {
            case .stop:
                break training
            case .next:
                break
            }

            outputs.append(contentsOf: try callbacks.filter { $0.execute.contains(.before) }.map {
                try $0.call(booster: self, iteration: iteration, evaluation: nil)
            })

            if outputs.contains(where: { $0 == .stop }) {
                break training
            }

            if let objectiveFunction = objectiveFunction {
                try update(
                    data: trainingData,
                    objective: objectiveFunction
                )
            } else {
                try update(
                    iteration: iteration,
                    data: trainingData
                )
            }

            let evaluation =
                evaluationData.isEmpty ? nil : try evaluate(
                    iteration: iteration, data: evaluationData, function: evaluationFunction
                )

            outputs.append(contentsOf: try callbacks.filter { $0.execute.contains(.after) }.map {
                try $0.call(booster: self, iteration: iteration, evaluation: evaluation)
            })

            switch try afterIteration(
                self,
                iteration,
                evaluation,
                outputs
            ) {
            case .stop:
                break training
            case .next:
                break
            }

            try saveRabitCheckpoint()
            version += 1

            if outputs.contains(where: { $0 == .stop }) {
                break training
            }
        }
    }
}
