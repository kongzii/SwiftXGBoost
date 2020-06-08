import CXGBoost
import Foundation

/// Dictionary for evaluation in form [data_name: [metric_name: value]]
public typealias Evaluation = [String: [String: String]]

/// Protocol for classes and structs that can be called in training
protocol Callback {
    /// Paremeter booster: Booster.
    /// Paremeter iteration: Current iteration.
    /// Paremeter evaluation: Dictionary with evaluations.
    func call(
        booster: Booster,
        iteration: Int,
        evaluation: Evaluation?
    ) throws -> AfterIterationCallbackOutput
}

public class EarlyStopping: Callback {
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

    /// Paremeter dataName: Name of data used for early stopping.
    /// Paremeter metricName: Metric to look for.
    /// Paremeter stoppingRounds: Number of rounds to check improvence for.
    /// Paremeter state: Initial state.
    /// Paremeter verbose: Print on new best or stopping.
    public init(
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

    /// Paremeter dataName: Name of data used for early stopping.
    /// Paremeter metricName: Metric to look for.
    /// Paremeter stoppingRounds: Number of rounds to check improvence for.
    /// Paremeter maximize: If metric should be maximized, minimzed otherwise.
    /// Paremeter verbose: Print on new best or stopping.
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

    /// Paremeter dataName: Name of data used for early stopping.
    /// Paremeter metricName: Metric to look for.
    /// Paremeter stoppingRounds: Number of rounds to check improvence for.
    /// Paremeter maximize: If metric should be maximized, minimzed otherwise.
    /// Paremeter booster: Booster to load state from.
    /// Paremeter verbose: Print on new best or stopping.
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

    /// Paremeter booster: Booster.
    /// Paremeter iteration: Current iteration.
    /// Paremeter evaluation: Dictionary with evaluations.
    public func call(
        booster: Booster,
        iteration: Int,
        evaluation: Evaluation?
    ) throws -> AfterIterationCallbackOutput {
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

            try booster.set(attribute: "best_score", value: String(state.bestScore))
            try booster.set(attribute: "best_iteration", value: String(state.bestIteration))
            try booster.set(attribute: "best_msg", value: state.bestMsg)

            if verbose {
                print(state.bestMsg)
            }
        } else if iteration - state.bestIteration >= stoppingRounds {
            if verbose {
                print("Stopping at iteration \(iteration): " + state.bestMsg)
            }

            return .stop
        }

        return .next
    }
}

extension Booster {
    /// Train booster.
    ///
    /// - Parameter iterations: Number of training iterations, but training can be stopped early.
    /// - Parameter trainingData: Data to train on.
    /// - Parameter evaluationData: Data to evaluate on, if provided.
    /// - Parameter earlyStopping: Early stopping.
    /// - Parameter beforeIteration: Callback called before each iteration.
    /// - Parameter afterIteration: Callback called after each iteration.
    public func train(
        iterations: Int,
        trainingData: Data,
        evaluationData: [Data] = [],
        earlyStopping: EarlyStopping? = nil,
        beforeIterationCallback: (Booster, Int) throws -> Void = { _, _ in },
        afterIterationCallback: (Booster, Int, Evaluation?) throws -> AfterIterationCallbackOutput = { _, _, _ in .next }
    ) throws {
        if earlyStopping != nil, evaluationData.count == 0 {
            throw ValueError.runtimeError("Evaluation data needs to be set for early stopping.")
        }

        var callbacks = [Callback]()
        var version = try loadRabitCheckpoint()
        let startIteration = Int(version / 2)

        if let earlyStopping = earlyStopping {
            callbacks.append(earlyStopping)
        }

        training: for iteration in startIteration ..< iterations {
            try beforeIterationCallback(
                self,
                iteration
            )

            try update(
                iteration: iteration,
                data: trainingData
            )

            let evaluation =
                evaluationData.isEmpty ? nil : try evaluate(iteration: iteration, data: evaluationData)

            let outputs = try callbacks.map {
                try $0.call(booster: self, iteration: iteration, evaluation: evaluation)
            }

            switch try afterIterationCallback(
                self,
                iteration,
                evaluation
            ) {
            case .stop:
                break training
            case .next:
                break
            }

            if outputs.contains(where: { $0 == .stop }) {
                break training
            }

            try saveRabitCheckpoint()
            version += 1
        }
    }
}
