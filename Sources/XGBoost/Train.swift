import CXGBoost
import Foundation

public typealias State = (
    maximizeScore: Bool,
    bestIteration: Int,
    bestScore: Float,
    bestMsg: String
)

public class EarlyStopping {
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

    public func call(
        booster: Booster,
        iteration: Int,
        evaluation: [String: [String: String]]
    ) throws -> AfterIterationCallbackOutput {
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
        beforeIteration: (Booster, Int) throws -> Void = { _, _ in },
        afterIteration: (Booster, Int, [String: [String: String]]?) throws -> AfterIterationCallbackOutput = { _, _, _ in .next }
    ) throws {
        var version = try loadRabitCheckpoint()
        let startIteration = Int(version / 2)

        training: for iteration in startIteration ..< iterations {
            try beforeIteration(
                self,
                iteration
            )

            try update(
                iteration: iteration,
                data: trainingData
            )

            let evaluation =
                evaluationData.isEmpty && earlyStopping == nil ? nil : try evaluate(iteration: iteration, data: evaluationData)

            if earlyStopping != nil {
                switch try earlyStopping!.call(
                    booster: self,
                    iteration: iteration,
                    evaluation: evaluation!
                ) {
                case .stop:
                    break training
                case .next:
                    break
                }
            }

            switch try afterIteration(
                self,
                iteration,
                evaluation
            ) {
            case .stop:
                break training
            case .next:
                break
            }

            try saveRabitCheckpoint()
            version += 1
        }
    }
}
