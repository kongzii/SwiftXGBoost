import Foundation
import PythonKit

/// Shape of Data.
public struct Shape: ExpressibleByArrayLiteral, Equatable {
    public let row: Int
    public let column: Int
    public typealias ArrayLiteralElement = Int

    public init(_ row: Int, _ column: Int) {
        self.row = row
        self.column = column
    }

    public init(row: Int, column: Int) {
        self.row = row
        self.column = column
    }

    public init(arrayLiteral elements: Int...) {
        precondition(elements.count == 2, "Invalid shape of input.")
        row = elements[0]
        column = elements[1]
    }

    public init(_ row: PythonObject, _ column: PythonObject) throws {
        guard let row = Int(row), let column = Int(column) else {
            throw ValueError.runtimeError("Invalid type of python input.")
        }

        self.row = row
        self.column = column
    }
}
