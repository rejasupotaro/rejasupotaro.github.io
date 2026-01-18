# Experiments Blog

Personal blog and experiments site powered by Material for MkDocs.

🌐 **Live Site**: https://rejasupotaro.github.io/

## Setup

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer

### Installation

1. Install uv (if not already installed):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Quick Start (Makefile)

The project includes a `Makefile` for common operations:

```bash
# Install dependencies
make sync

# Start development server
make dev

# Build the site
make build

# Clean build artifacts
make clean
```

### Manual Commands

If you prefer to run commands manually or need specific flags:

1. Install dependencies:

```bash
uv sync
```

2. Start the development server:
```bash
uv run mkdocs serve
```

3. Manage dependencies:

```bash
# Add a package
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a package
uv remove package-name
```

## Writing Blog Posts

### Create a New Post

1. Create a new markdown file in `docs/blog/posts/`:

```bash
docs/blog/posts/my-new-post.md
```

2. Add frontmatter with metadata:

```markdown
---
date: 2024-10-18
authors:
  - rejasupotaro
tags:
  - python
  - web
  - tutorial
---

# My Post Title

Brief excerpt that appears in the blog listing.

<!-- more -->

Full post content goes here...
```

### Frontmatter Options

- `date`: Publication date (YYYY-MM-DD)
- `authors`: List of author IDs (defined in `docs/blog/.authors.yml`)
- `tags`: Post tags (for organization and filtering)
- `draft: true`: Mark post as draft (won't be published)

## Deployment

### GitHub Actions (Automatic)

The site automatically deploys to GitHub Pages when you push to the `main` branch.

**First-time setup:**

1. Go to your repository Settings → Pages
2. Under "Build and deployment", set Source to "GitHub Actions"
3. Push your changes to trigger the workflow

### Manual Deployment

You can also deploy manually:
```bash
make deploy
# OR
uv run mkdocs gh-deploy
```

## Project Structure

```
.
├── docs/                      # Documentation source
│   ├── index.md              # Home page
│   ├── blog/                 # Blog section
│   │   ├── .authors.yml      # Author information
│   │   └── posts/            # Blog posts
│   ├── experiments/          # Experiments pages
│   └── tags.md              # Tags index
├── mkdocs.yml               # MkDocs configuration
├── pyproject.toml           # Python dependencies (uv)
├── uv.lock                  # Lockfile (generated)
└── .github/
    └── workflows/
        └── deploy.yml       # GitHub Actions workflow
```

## Features

- ✅ Blog with posts and tags
- ✅ RSS feed
- ✅ Full-text search
- ✅ Dark/light mode
- ✅ Code syntax highlighting
- ✅ Responsive design
- ✅ Reading time estimates
- ✅ Archive by date
- ✅ Social sharing cards

## Customization

### Changing Theme Colors

Edit `mkdocs.yml`:

```yaml
theme:
  palette:
    - scheme: default
      primary: indigo  # Change this
      accent: indigo   # And this
```

Available colors: red, pink, purple, deep purple, indigo, blue, light blue, cyan, teal, green, light green, lime, yellow, amber, orange, deep orange

### Adding Social Links

Edit `mkdocs.yml`:

```yaml
extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/yourusername
    - icon: fontawesome/brands/twitter
      link: https://twitter.com/yourusername
```

## Learn More

- [Material for MkDocs Documentation](https://squidfunk.github.io/mkdocs-material/)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Blog Plugin](https://squidfunk.github.io/mkdocs-material/plugins/blog/)
