// swift-tools-version:5.1

import PackageDescription

let package = Package(
    name: "SwiftXGBoost",
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
        .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
    ],
    targets: [
        .target(
            name: "XGBoost",
            dependencies: [
                "CXGBoost",
            ]
        ),
        .systemLibrary(
            name: "CXGBoost",
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
    ]
)
