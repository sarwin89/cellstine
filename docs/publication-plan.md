# Package publication plan

Status: planning only. No upload workflow, credential change, tag, release, or package publication is authorized by the systematic-review goal.

## Candidate readiness

1. Finish the tracked-function inventory audit. Every function must have review evidence or an investigated, documented deferral; no untouched pending entries may be described as reviewed. Resolve correctness blockers and describe remaining limitations.
2. Select a frozen candidate from `dev` after its exact commit passes Python 3.10–3.14 tests, ledger drift checks, coverage reporting, packaging CI the NumPy1.26.4 helper lane, and all nine Windows/Linux/macOS portable-result lanes. Review coverage gaps with scientific assertions; do not treat a percentage as a correctness proof.
3. Compare public APIs, CLI/help, result schemas, artifact names, ordering and scientific conventions with the preceding release. Keep N-layer functionality labelled experimental unless a separate acceptance audit proves otherwise.
4. Verify the actual existing PyPI/TestPyPI versions and maintainer access. A patch version is appropriate only if the final changes remain backward compatible; `3.0.3` is a proposal, not an assigned or reserved version. Defer the final version until the full review's scope is known.
5. Update `pyproject.toml`, `cellstine.__version__`, README version badge and release notes together. Re-run all gates after version/metadata changes.

Part03b identified release follow-ups in [the shared-infrastructure report](optimization/part-03b-shared-infrastructure.md). [Part03c](optimization/part-03c-safe-default-outputs.md) protects the default-output consumers through exclusive destination reservation while preserving free preferred names and explicit file behavior; its working-directory and failure-artifact limits remain explicit. [Part04](optimization/part-04-io-integrity.md) addresses molecular fallback temporary-file ownership and incomplete XYZ frames; its optional-dependency ownership probe uses a fake adapter and does not establish every external format. Extreme-scale validation still needs a supported-domain decision or end-to-end repair with unchanged scientific thresholds. These remaining limits may not disappear from candidate-readiness review merely because CI is green.

The user-authorized [portable I/O follow-up](optimization/portable-io-options.md) plans optional versioned no-pickle NPZ while retaining JSON compatibility. Before including it in a candidate, verify archive validation, exact values/order, legacy JSON consumers, manifest routing, generation/plot readbacks and Windows/Linux/macOS tests with supported Python/NumPy combinations. Part05d commit8bfcf17 passed local full/package/installed checks and all16 exact-head CI41777781487 jobs. All nine Windows/Linux/macOS format lanes passed36cases each (2optional-render skips), with NumPy1.26.4/2.2.6/2.5.4 tested across their designated Python versions. Coverage artifact13298477050 has verified head and ZIPdigest; supported-platform format execution is established for this checkpoint, not every future candidate or optional dependency. Reassess the version proposal if the final candidate adds this new format capability.

Read-only public metadata check on 2026-10-30: `https://pypi.org/pypi/cellstine/json` returned HTTP 404. No public version history was read back. This does not establish project-name availability or maintainer access, and no account or publisher configuration was changed. If a new project is required, PyPI documents a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/); configuring one does not reserve the name, so availability must still be checked at the later authorized publication step.

The2026-11-05 user-authorized NPZ default changes the default bilayer and
assembly filename/artifact to `results.npz`/`results_npz`. Preserve explicit JSON
exports and old JSON readers, document this migration, and reassess release
versioning against scripts that hardcode the former default. Do not label this
as an unchanged default or publish without the later explicit instruction.

## Artifact validation

Build fresh wheel and source distribution from the frozen candidate; keep release artifacts separate from prior local builds:

```powershell
python -m pytest tests -q
python tools/review_inventory.py --check
python -m build --sdist --wheel --outdir .tmpdist/release-candidate
python -m twine check .tmpdist/release-candidate/*
```

On Windows restricted environments, use writable verification directories for TEMP/TMP and PIP_CACHE_DIR. Part05e directly diagnosed an indefinite temporary-directory retry in the default pip sandbox cache; the offline isolated sdist install passed with a workspace cache. This is an environment setup requirement, not a runtime dependency change.

Record artifact SHA256 values and compare wheel contents with package source. Reject generated runs, local environments, credentials and unrelated files. Install the exact wheel core-only, wheel with CLI extras, and source distribution into separate fresh environments. Check version/help, import all public workflows, exercise representative numerical/I/O workflows and run dependency checks. Repeat core-only/minimum-dependency and optional-extras coverage for supported interpreters through CI where practical. Verify distribution licenses and source links, and ensure the README renders correctly through Twine's metadata checks.

## Future delivery sequence, requiring an explicit publication instruction

1. Present the frozen commit, proposed version, release notes, CI links, artifact checksums and unresolved limitations for maintainer review.
2. Once instructed to publish, integrate the approved candidate into `main` under repository policy and confirm the same commit's gates. Create a new annotated version tag; do not move an already published tag.
3. Prefer a separately configured trusted publisher scoped to repository owner `sarwin89`, repository `cellstine`, an approved publishing workflow filename, and a protected publication environment. PyPI's [trusted publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) binds these identities and can use environment restrictions. The publishing account/environment configuration is future maintainer work, not performed here.
4. If requested, publish a rehearsal to TestPyPI and install the resulting artifacts in clean environments. Do not assume PyPI and TestPyPI identities/versions are interchangeable. Then publish only the previously checked production artifacts to PyPI when authorized.
5. Read back the published package version and checksums, install from PyPI in a clean environment, and check version/help and a representative workflow. Attach the approved artifacts/release notes to a GitHub Release only when that publication is also requested.
6. If a published blocker is discovered, document its impact and prepare a new patch; consider yanking only under an explicit maintainer decision. Preserve artifact/checksum provenance and never silently replace consumed versions/tags.

This plan deliberately does not create a functioning upload action. The current work expands validation and prepares a reviewable release decision while leaving the actual publication action for later authorization.
