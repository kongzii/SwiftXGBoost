.PHONY: documentation

test:
	docker-compose run test

format:
	docker-compose run swiftxgboost swiftformat . --exclude Sources/XGBoost/Python.swift,Tests/XGBoostTests/ArrayTests.swift,.build,.history

documentation:
	@sourcekitten doc \
		--spm \
		--module-name XGBoost \
		> SK_XGBoost.json
	@jazzy \
		--output Documentation \
		--github_url https://github.com/kongzii/SwiftXGBoost \
		--min-acl public \
		--sourcekitten-sourcefile SK_XGBoost.json
	@rm SK_*.json
