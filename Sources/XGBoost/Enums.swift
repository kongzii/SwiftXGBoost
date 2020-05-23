import Foundation

public
typealias Parameter = (name: String, value: String)

public
typealias Version = (major: Int, minor: Int, patch: Int)

public
typealias Feature = (name: String, type: FeatureType)

public enum AfterIterationCallbackOutput {
    case stop, next
}

public enum FeatureType: String {
    case quantitative = "q"
    case indicator = "i"
}

public enum ModelFormat: String {
    case text, json, dot
}

public enum DataFormat: String {
    case binary, csv
}

public enum Field: String {
    case label
    case weight
    case baseMargin = "base_margin"
}

public enum PredictOutputMask: Int32 {
    case normal = 0 // normal prediction
    case margin = 1 // output margin instead of transformed value
    case leaf = 2 // output leaf index of trees instead of leaf value, note leaf index is unique per tree
    case feature = 4 // output feature contributions to individual predictions
}

public enum Stage: Int32 {
    case inference = 0
    case training = 1
}
