# Release checklist

CELLSTINE releases are cut from `main`. Use `dev` for integration work, then
fast-forward `main` only after the release candidate passes the gates below.

## Versioning

- Use semantic versions.
- Use a major release for hard CLI/API breaks, such as the Gram-form moire
  replacement and simplified CLI in the 3.x line.
- Do not move a pushed release tag silently. If a blocker is found after a tag
  has been pushed, prefer a patch release unless the tag was clearly never
  consumed and the maintainer explicitly decides to replace it.

## Required gates

Run these before publishing package artifacts:

```powershell
python -m pytest tests -q
python -m build --sdist --wheel
python -m twine check dist/*
```

Then smoke-test both artifacts in clean environments:

```powershell
python -m venv .release-core-venv
.release-core-venv\Scripts\python.exe -m pip install dist\cellstine-<version>-py3-none-any.whl
.release-core-venv\Scripts\cellstine.exe --version

python -m venv .release-cli-venv
.release-cli-venv\Scripts\python.exe -m pip install "dist\cellstine-<version>-py3-none-any.whl[cli]"
.release-cli-venv\Scripts\cellstine.exe --help

python -m venv .release-sdist-venv
.release-sdist-venv\Scripts\python.exe -m pip install dist\cellstine-<version>.tar.gz
.release-sdist-venv\Scripts\cellstine.exe --version
```

For supported Python versions, run the full test suite on Python 3.10, 3.11,
and 3.12. The GitHub Actions workflow mirrors that matrix on Windows because
the guided CLI has Windows console-encoding behavior that should remain tested.

## Publication sequence

1. Ensure `pyproject.toml`, `src/cellstine/__init__.py`, and the README badge
   agree on the release version.
2. Run the full test matrix and package checks.
3. Commit any release-only fixes.
4. Push `main` and `dev`.
5. Create an annotated tag, for example `v3.0.1`.
6. Push the tag.
7. Upload the checked files from `dist/` to PyPI.

`dist/`, local release virtual environments, pytest basetemps, and Python cache
directories are generated local artifacts and should not be committed.
