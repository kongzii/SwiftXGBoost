import AGGRenderer
import Plotly
import SwiftPlot

extension XGBoost {
    public func saveImportanceGraph(
        fileName: String,
        featureMap: String = "",
        importance: Importance = .weight,
        label: String = "Feature importance",
        title: String = "Feature importance",
        xAxisLabel: String = "Score",
        yAxisLabel: String = "Features",
        maxNumberOfFeatures: Int? = nil,
        graphOrientation: BarGraph<String, Float>.GraphOrientation = .horizontal,
        enableGrid: Bool = true,
        size: Size = Size(width: 1000, height: 660),
        renderer: Renderer = AGGRenderer()
    ) throws {
        let (features, gains) = try score(featureMap: featureMap, importance: importance)

        if features.count == 0 {
            throw ValueError.runtimeError(
                "Score from booster is empty. This maybe caused by having all trees as decision dumps."
            )
        }

        var importance: [(name: String, value: Float)] = {
            switch importance {
            case .weight:
                return features.map { ($0, Float($1)) }
            case .gain, .cover, .totalGain, .totalCover:
                return gains!.map { ($0, $1) }
            }
        }()

        importance = importance.sorted(by: { $0.value < $1.value })

        if let maxNumberOfFeatures = maxNumberOfFeatures {
            importance = Array(importance[importance.count - maxNumberOfFeatures ..< importance.count])
        }

        let x = importance.map { $0.name }
        let y = importance.map { $0.value }

        var graph = BarGraph<String, Float>(enableGrid: enableGrid)
        graph.addSeries(x, y, label: label, graphOrientation: graphOrientation)
        graph.plotTitle.title = title
        graph.plotLabel.xLabel = xAxisLabel
        graph.plotLabel.yLabel = yAxisLabel
        try graph.drawGraphAndOutput(size: size, fileName: fileName, renderer: renderer)
    }
}
