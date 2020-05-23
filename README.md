# XGBoost for Swift

Bindings for XGBoost system library.

## Installation

### System library dependency

#### Linux

Install XGBoost from sources

```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git checkout release_1.1.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_LIBDIR=/usr/lib ..
make -j$(nproc)
make install
```

#### OSX

Intall XGBoost from brew

```
brew install xgboost
```

### Package

Add dependency in your your `Package.swift`

```swift
.package(url: "https://github.com/kongzii/SwiftXGBoost.git", from: "0.0.0"),
```

Import Swifty XGBoost 

```swift
import XGBoost
```

or directly C library 

```swift
import CXGBoost
```

both `XGBoost` and `Data` classes are exposing `pointee` variable to the underlying C pointer,
so you can utilize C-API directly for more advanced usage.

## Example usage

```swift
import XGBoost

// Create some random features and labels
let randomArray = (0 ..< 1000).map { _ in Float.random(in: 0 ..< 2) }
let labels = (0 ..< 100).map { _ in Float([0, 1].randomElement()!) }

// Initialize data, DMatrixHandle in the background
let data = try Data(
    name: "data",
    values: randomArray,
    rowCount: 100,
    columnCount: 10,
    threads: 1
)

// Set labels
try data.setFloatInfo(field: .label, values: labels)

// Slice array into train and test
let train = try data.slice(indexes: 0 ..< 90, newName: "train")
let test = try data.slice(indexes: 90 ..< 100, newName: "test")

// Parameters for XGBoost, check https://xgboost.readthedocs.io/en/latest/parameter.html
let parameters: [Parameter] = [
    ("seed", "0"),
]

// Create XGBoost model, `with` data will be cached
let xgboost = try XGBoost(
    with: [train, test],
    parameters: parameters
)

// Train xgboost, optionally provide callback functions called before and after each iteration
try xgboost.train(
    iterations: 10,
    trainingData: train,
    evaluationData: [train, test]
)

// Predict from test data
let predictions = try xgboost.predict(from: test)
```
