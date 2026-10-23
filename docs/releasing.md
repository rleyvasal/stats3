# Releasing stats3

A release is one version bump and one tag. GitHub Actions does the rest:
it runs the tests with R on Python 3.12 and 3.14, builds the wheel and
sdist, uploads them to PyPI, and creates a GitHub release with the files
attached.

## Each release

1. Set the new version in `src/stats3/__version__.py` (the only place it lives)
   and date its section in `CHANGELOG.md`.
2. Commit, tag, and push:

   ```bash
   git commit -am "Release 0.1.1"
   git tag v0.1.1
   git push origin main v0.1.1
   ```

The tag must equal `v` + `__version__`, or the workflow stops before
uploading. PyPI never accepts the same version twice, so a mistake means a
new version number, not a re-upload.

Watch the run under the repository's **Actions** tab, or with
`gh run watch`.

## One-time setup

PyPI trusted publishing lets the workflow upload without any token or
password stored in GitHub or on your computer.

1. Sign in at <https://pypi.org> (with two-factor authentication on).
2. Go to **Your projects → Publishing → Add a new pending publisher** and enter:
   - PyPI project name: `stats3`
   - Owner: `rleyvasal`
   - Repository name: `stats3`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Optional: GitHub creates the `pypi` environment on the first run. To make
   each upload wait for one click of approval, open **Settings →
   Environments → pypi** and add yourself as a required reviewer.

The first tagged release creates the project on PyPI; after that the
pending publisher becomes a normal one.

## Supported versions

Python 3.12+, Polars 1.39+, SciPy 1.18+ (the first whose inverse t and
Shapiro-Wilk match R to full precision); pandas 2.2.2+ and pyarrow 16+ for
pandas data. The `minimum` CI job tests exactly these.

## Checking a build locally

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```
