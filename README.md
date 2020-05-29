# XGBoost for Swift

Bindings for [the XGBoost system library](https://en.wikipedia.org/wiki/XGBoost). 
The aim of this package is to mimic [XGBoost Python bindings](https://xgboost.readthedocs.io/en/latest/python/python_intro.html) but, at the same time, utilize the power of Swift and C compatibility. Some things thus behave differently but should provide you maximum flexibility over XGBoost.

Documentation is available at [Wiki](https://github.com/kongzii/SwiftXGBoost/wiki).

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
cmake ..
make -j$(nproc)
make install
ldconfig
```

Or you can use provided installation script

```
./install.sh
```

#### OSX

You can build and install similarly as on Linux, or just use brew

```
brew install xgboost
```

### Package

Add a dependency in your your `Package.swift`

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

As library is still evolving, there can be incompatible changes, 
use exact version if you do not want to worry about updating your packages.

```swift
.package(url: "https://github.com/kongzii/SwiftXGBoost.git", .exact("0.1.0")),
```

## Example usage

```swift
import XGBoost

// Register your own callback function for log(info) messages
try XGBoost.registerLogCallback {
    print("Swifty log:", String(cString: $0!))
}

// Create some random features and labels
let randomArray = (0 ..< 1000).map { _ in Float.random(in: 0 ..< 2) }
let labels = (0 ..< 100).map { _ in Float([0, 1].randomElement()!) }

// Initialize data, DMatrixHandle in the background
let data = try Data(
    name: "data",
    values: randomArray,
    shape: (100, 10),
    label: labels,
    threads: 1
)

// Slice array into train and test
let train = try data.slice(indexes: 0 ..< 90, newName: "train")
let test = try data.slice(indexes: 90 ..< 100, newName: "test")

// Parameters for XGBoost, check https://xgboost.readthedocs.io/en/latest/parameter.html
let parameters: [Parameter] = [
    ("verbosity", "2"),
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

// Save
try xgboost.save(to: "model.xgboost")
```

## Development

### Documentation

[Swiftdoc](https://github.com/SwiftDocOrg/swift-doc) is used for generation of documenation for module `XGBoost`.

You can generate documentation locally using 

```
docker-compose run documentation
```

Github wiki will be updated automatically, when merged into master.

### Tests

Where possible, Swift implementation is tested against reference implementation in Python via PythonKit. For example, test of `score` method in `scoreEmptyFeatureMapTest`

```swift
let pyFMap = [String: Int](pyXgboost.get_score(
    fmap: "", importance_type: "weight"))!
let (fMap, _) = try xgboost.score(featureMap: "", importance: .weight)

XCTAssertEqual(fMap, pyFMap)
```

#### Run locally

```
docker-compose run test 
```
