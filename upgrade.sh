# Build and upload a release to PyPI (maintainer use only).
rm -rf dist
python -m build
twine upload dist/*
