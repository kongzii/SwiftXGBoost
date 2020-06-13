public extension Dictionary {
    subscript(key: Key, or def: Value) -> Value {
        mutating get {
            self[key] ?? {
                self[key] = def
                return def
            }()
        }
        set { self[key] = newValue }
    }
}
