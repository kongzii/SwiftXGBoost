import CXGBoost
import Foundation

enum XGBoostError: Error {
    /// Runtime error caused when output of C-API functions != 0.
    case runtimeError(String)
}

enum ValueError: Error {
    /// Runtime error caused on incompatible inputs.
    case runtimeError(String)
}

/// Helper function to handle C-API calls.
///
/// - Parameter call: Closure returning output of XGBoost C-API function.
func safe(call: () -> Int32) throws {
    if call() != 0 {
        throw XGBoostError.runtimeError(String(cString: XGBGetLastError()))
    }
}
