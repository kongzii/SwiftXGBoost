import SVGRenderer
import SwiftPlot

extension Booster {
    /// Saves plot with importance based on fitted trees.
    ///
    /// - Parameter to: File where graph will be saved, .svg extension will be added if rendered remains SVGRenderer.
    /// - Parameter featureMap: Path to the feature map, if provided, replaces default f0, f1, ... feature names.
    /// - Parameter importance: Type of importance to plot.
    /// - Parameter label: Label of graph.
    /// - Parameter title: Title of graph.
    /// - Parameter xAxisLabel: Label of X-axis.
    /// - Parameter yAxisLabel: Label of Y-axis.
    /// - Parameter maxNumberOfFeatures: Maximum number of top features displayed on plot. If None, all features will be displayed.
    /// - Parameter graphOrientation: Orientaton of ploted graph.
    /// - Parameter enableGrid: urn the axes grids on or off.
    /// - Parameter size: Size of ploted graph.
    /// - Parameter renderer: Renderer to use.
    public func saveImportanceGraph(
        to fileName: String,
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
        renderer: Renderer = SVGRenderer()
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

        let x = importance.map(\.name)
        let y = importance.map(\.value)

        var graph = BarGraph<String, Float>(enableGrid: enableGrid)
        graph.addSeries(x, y, label: label, graphOrientation: graphOrientation)
        graph.plotTitle.title = title
        graph.plotLabel.xLabel = xAxisLabel
        graph.plotLabel.yLabel = yAxisLabel
        try graph.drawGraphAndOutput(size: size, fileName: fileName, renderer: renderer)
    }
}
