import CXGBoost
import Foundation

/// Dictionary for evaluation in form [data_name: [metric_name: value]]
public typealias Evaluation = [String: [String: String]]

/// Specify if should be executed at beginning of iteration or at the end.
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
    /// - Parameter evaluation: Optional dictionary with evaluations.
    func call(
        booster: Booster?,
        iteration: Int,
        evaluation: Evaluation?
    ) throws -> AfterIterationOutput
}

/// Class used for early stopping feature.
public class EarlyStopping: Callback {
    /// Name of the callback.
    public var name: String = "EarlyStopping"

    /// Position of the execution.
    public var execute: [LoopPosition] = [.after]

    /// Typealais for tuple holding current state.
    public typealias State = (
        maximizeScore: Bool,
        bestIteration: Int,
        bestScore: Float,
        bestMsg: String
    )

    /// Current state of training.
    public var state: State

    /// Name of watched DMatrix.
    public var dataName: String

    /// Name of watched metric.
    public var metricName: String

    /// Number of stopping rounds.
    public var stoppingRounds: Int

    /// If true, statistics will be printed.
    public var verbose: Bool

    /// - Parameter bestIteration: Number of booster best iteration.
    /// - Parameter bestEvaluation: Booster best evaluation.
    static func formatMessage(
        bestIteration: Int,
        bestEvaluation: Any
    ) -> String {
        "Best iteration: \(bestIteration), best evaluation: \(bestEvaluation)."
    }

    /// Decides whenever metric should be maximized or minimized.
    ///
    /// - Parameter maximize: Force maximization.
    /// - Parameter metricName: Metric name to check, if it should be maximized.
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

    /// Call used in Booster training.
    ///
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

    /// Call used in cross-validation training.
    ///
    /// - Parameter iteration: Current iteration.
    /// - Parameter evaluation: Cross-validation evaluation.
    public func call(
        iteration: Int,
        evaluation: CVEvaluation
    ) throws -> AfterIterationOutput {
        let name = dataName + "-test-" + metricName + "-mean"

        guard let scores = evaluation[name] else {
            throw ValueError.runtimeError("Name \(name) not found in evaluation \(evaluation).")
        }

        guard let score = scores.last else {
            throw ValueError.runtimeError("Scores for \(name) are empty.")
        }

        if (state.maximizeScore && score > state.bestScore) || (!state.maximizeScore && score < state.bestScore) {
            state.bestScore = score
            state.bestIteration = iteration
            state.bestMsg = EarlyStopping.formatMessage(bestIteration: iteration, bestEvaluation: evaluation)

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
}

/// Class used for variable learning rate feature.
public class VariableLearningRate: Callback {
    /// Name of the callback.
    public var name: String = "VariableLearningRate"

    /// Position of the execution.
    public var execute: [LoopPosition] = [.before]

    /// Typealias for learningRateFunction.
    public typealias Function = (Int, Int) -> String

    /// Number of iterations in training.
    var iterations: Int

    /// List with learning rates.
    var learningRates: [String]?

    /// Function returning learning rate as string based on current iteration and maximum number of iterations.
    var learningRateFunction: Function?

    /// Initialize VariableLearningRate by array of learning rates.
    ///
    /// - Precondition: iterations == learningRates.count.
    /// - Parameter learningRates: Array of learning rates that will be accessed at every iteration.
    /// - Parameter iterations: Number of iteration in training.
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

    /// Initialize VariableLearningRate with function generating learning rate.
    ///
    /// - Parameter learningRate: Function that will return learning rate at each iteration.
    /// - Parameter iterations: Number of iteration in training.
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
}

/// Typealias for function called before each iteration at training.
public typealias BeforeIteration = (Booster, Int) throws -> AfterIterationOutput

/// Typealias for function called after each iteration at training.
public typealias AfterIteration = (Booster, Int, Evaluation?, [AfterIterationOutput]) throws -> AfterIterationOutput

/// Default before iteration function, basically does nothing.
public let DefaultBeforeIteration: BeforeIteration = { _, _ in .next }

/// Default after iteration function, basically does nothing.
public let DefaultAfterIteration: AfterIteration = { _, _, _, _ in .next }

extension Booster {
    /// Train booster.
    ///
    /// - Parameter iterations: Number of training iterations, but training can be stopped early.
    /// - Parameter startIteration: N. of starting iteration.
    /// - Parameter trainingData: Data to train on.
    /// - Parameter evaluationData: Data to evaluate on, if provided.
    /// - Parameter evaluationFunction: Custom evaluation function.
    /// - Parameter beforeIteration: Callback called before each iteration.
    /// - Parameter callbacks: Array of callbacks called at each iteration.
    /// - Parameter afterIteration: Callback called after each iteration.
    public func train(
        iterations: Int,
        startIteration: Int? = nil,
        trainingData: Data,
        objectiveFunction: ObjectiveFunction? = nil,
        evaluationData: [Data] = [],
        evaluationFunction: EvaluationFunction? = nil,
        beforeIteration: BeforeIteration = DefaultBeforeIteration,
        callbacks: [Callback] = [],
        afterIteration: AfterIteration = DefaultAfterIteration
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

            try saveRabitCheckpoint()
            version += 1

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

            if outputs.contains(where: { $0 == .stop }) {
                break training
            }
        }
    }
}
