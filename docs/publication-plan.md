# Package publication plan

Status: planning only. No upload workflow, credential change, tag, release, or package publication is authorized by the systematic-review goal.

## Candidate readiness

1. Finish the tracked-function inventory audit. Every function must have review evidence or an investigated, documented deferral; no untouched pending entries may be described as reviewed. Resolve correctness blockers and describe remaining limitations.
2. Select a frozen candidate from `dev` after its exact commit passes Python 3.10–3.14 tests, ledger drift checks, coverage reporting, and packaging CI. Review coverage gaps with scientific assertions; do not treat a percentage as a correctness proof.
3. Compare public APIs, CLI/help, result schemas, artifact names, ordering and scientific conventions with the preceding release. Keep N-layer functionality labelled experimental unless a separate acceptance audit proves otherwise.
4. Verify the actual existing PyPI/TestPyPI versions and maintainer access. A patch version is appropriate only if the final changes remain backward compatible; `3.0.3` is a proposal, not an assigned or reserved version. Defer the final version until the full review's scope is known.
5. Update `pyproject.toml`, `cellstine.__version__`, README version badge and release notes together. Re-run all gates after version/metadata changes.

Read-only public metadata check on 2026-10-30: `https://pypi.org/pypi/cellstine/json` returned HTTP 404. No public version history was read back. This does not establish project-name availability or maintainer access, and no account or publisher configuration was changed. If a new project is required, PyPI documents a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/); configuring one does not reserve the name, so availability must still be checked at the later authorized publication step.

## Artifact validation

Build fresh wheel and source distribution from the frozen candidate; keep release artifacts separate from prior local builds:

```powershell
python -m pytest tests -q
python tools/review_inventory.py --check
python -m build --sdist --wheel --outdir .tmpdist/release-candidate
python -m twine check .tmpdist/release-candidate/*
```

Record artifact SHA256 values and compare wheel contents with package source. Reject generated runs, local environments, credentials and unrelated files. Install the exact wheel core-only, wheel with CLI extras, and source distribution into separate fresh environments. Check version/help, import all public workflows, exercise representative numerical/I/O workflows and run dependency checks. Repeat core-only/minimum-dependency and optional-extras coverage for supported interpreters through CI where practical. Verify distribution licenses and source links, and ensure the README renders correctly through Twine's metadata checks.

## Future delivery sequence, requiring an explicit publication instruction

1. Present the frozen commit, proposed version, release notes, CI links, artifact checksums and unresolved limitations for maintainer review.
2. Once instructed to publish, integrate the approved candidate into `main` under repository policy and confirm the same commit's gates. Create a new annotated version tag; do not move an already published tag.
3. Prefer a separately configured trusted publisher scoped to repository owner `sarwin89`, repository `cellstine`, an approved publishing workflow filename, and a protected publication environment. PyPI's [trusted publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) binds these identities and can use environment restrictions. The publishing account/environment configuration is future maintainer work, not performed here.
4. If requested, publish a rehearsal to TestPyPI and install the resulting artifacts in clean environments. Do not assume PyPI and TestPyPI identities/versions are interchangeable. Then publish only the previously checked production artifacts to PyPI when authorized.
5. Read back the published package version and checksums, install from PyPI in a clean environment, and check version/help and a representative workflow. Attach the approved artifacts/release notes to a GitHub Release only when that publication is also requested.
6. If a published blocker is discovered, document its impact and prepare a new patch; consider yanking only under an explicit maintainer decision. Preserve artifact/checksum provenance and never silently replace consumed versions/tags.

This plan deliberately does not create a functioning upload action. The current work expands validation and prepares a reviewable release decision while leaving the actual publication action for later authorization.
