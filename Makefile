.PHONY: documentation

documentation:
	@jazzy \
		--output Documentation \
		--github_url https://github.com/kongzii/SwiftXGBoost \
		--min-acl internal
