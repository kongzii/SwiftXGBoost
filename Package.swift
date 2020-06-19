// swift-tools-version:5.1

import PackageDescription

let package = Package(
    name: "SwiftXGBoost",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(
            name: "XGBoost",
            targets: [
                "XGBoost",
            ]
        ),
        .library(
            name: "CXGBoost",
            targets: [
                "CXGBoost",
            ]
        ),
    ],
    dependencies: [
        // TODO: Switch to official PythonKit when its versioned.
        .package(url: "https://github.com/kongzii/PythonKit", .exact("0.0.1")),
        // .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
        .package(url: "https://github.com/KarthikRIyer/swiftplot.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "XGBoost",
            dependencies: [
                "CXGBoost",
                "Logging",
                "PythonKit",
                "SwiftPlot",
                "AGGRenderer",
            ]
        ),
        .systemLibrary(
            name: "CXGBoost",
            pkgConfig: "xgboost",
            providers: [
                .brew(["xgboost"]),
            ]
        ),
        .testTarget(
            name: "XGBoostTests",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ]
        ),
        .target(
            name: "AftSurvival",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ],
            path: "Examples/AftSurvival"
        ),
        .target(
            name: "CrossValidation",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ],
            path: "Examples/CrossValidation"
        ),
        .target(
            name: "BoostFromPrediction",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ],
            path: "Examples/BoostFromPrediction"
        ),
        .target(
            name: "CustomObjective",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ],
            path: "Examples/CustomObjective"
        ),
        .target(
            name: "EvalsResult",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ],
            path: "Examples/EvalsResult"
        ),
        .target(
            name: "ExternalMemory",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ],
            path: "Examples/ExternalMemory"
        ),
    ]
)
