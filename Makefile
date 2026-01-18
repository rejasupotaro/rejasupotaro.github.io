.PHONY: dev build sync clean deploy

# Serve the site locally
dev:
	uv run mkdocs serve

# Build the static site
build:
	uv run mkdocs build

# Sync dependencies
sync:
	uv sync

# Clean build artifacts
clean:
	rm -rf site

# Deploy to GitHub Pages
deploy:
	uv run mkdocs gh-deploy
