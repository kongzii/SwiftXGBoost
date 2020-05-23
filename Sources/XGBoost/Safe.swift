import CXGBoost
import Foundation

enum XGBoostError: Error {
    case runtimeError(String)
}

func safe(call: () -> Int32) throws {
    if call() != 0 {
        throw XGBoostError.runtimeError(String(cString: XGBGetLastError()))
    }
}
