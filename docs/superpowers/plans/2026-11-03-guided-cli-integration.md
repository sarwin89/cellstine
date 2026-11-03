# Guided CLI integration and resource review

The user authorized continued autonomous integration, optimization and pushes to dev. Preserve the shared root/Claude exports and use only the isolated dev clone. Prior persistence checkpoint837be42 has all seven exact-head CI jobs and downloaded coverage verified.

## Scope and design

Integrate original01 CLI and incremental05 exports through the prepared three-way merge, preserving current dev output reservations, public parsers and scientific policies. Review all119 original scoped functions plus the additional history/replay/preview dependencies individually; clean text merges are not semantic proof. History records validated answers and caches in-memory previews; execution belongs outside replay so Back/cancel cannot repeat commands or persist unconfirmed previews. Confirmed commands retain error/retry behavior. Preserve exact multiline/help/text escaping where meaningful and both plain/Rich operation.

The prepared merge needed explicit missing imports, the optional prefilled slab argument and an in-memory defect-analysis return type: dev's public preview persists a run and returns CommandResult, which cannot substitute for DefectAnalysis. Keep the public preview command intact; the guided helper reads/analyzes without persisting and generated commands use the original source.

Replace05's header regex with the **actual candidate count from the already validated preview read**. A64KB substring is not a safe index bound: nested keys, split numeric tokens and duplicate top-level keys can mislead it. A producer's metadata can also disagree with actual rows under the current validator. The core private preview helper returns only text/count, preserving the public string API and avoiding retained documents. Multi-result manifests have no single bound. Building reuses its preview read/cache; selected visualization performs one validated read when needed. Empty selections get a clear error. This deliberately supersedes the earlier proposed bounded-header parser, whose inability to exclude later duplicate keys made its guarantee too weak.

Investigate bounded recent-file discovery in root-priority order. Preserve path resolution/dedup, first-seen equal-mtime ordering and exact selected files; avoid descending into lower-priority roots once enough higher-priority matches exist. Use controlled fixtures/owned trees and a baseline comparison; do not scan scientific runs for benchmarks. Leave sound helpers unchanged.

## Execution checkpoints

1. Prepared base/ours/theirs hashes and export reproduction are in `.tmp/claude-cli-prep-20261103`;29 conflict blocks are resolved in scratch, including explicit dependency repairs. F821 passes there; this is not yet production CLI verification.
2. Core preview helper has four red-before/green-after regressions for empty/one/101 rows, misleading declared count, one-read behavior, public text compatibility and multi-result ambiguity. Add actual valid-file/nested/truncation/duplicate-key guided tests when applying the scratch CLI. Update exported tests to assert real validated inputs instead of fabricated incomplete headers.
3. Apply only owned prepared files after asserting the dev baseline snapshots still match. Inspect the complete diff and callers, then run affected CLI/workflow tests. Fix integration failures rather than dropping existing assertions. Validate every built command with the public parser.
4. Benchmark changed discovery/preview paths against837be42 and inspect all scoped functions with individual ledger reasons. Request independent read-only review, freeze tracked Python hashes, then run one full suite/coverage and relevant build/Twine/installed checks.
5. Commit scoped verified changes, push dev and read exact-head supported-version CI/coverage before the next production batch. Keep function-level review/global audit open.

## Additional user request

Portable I/O comparison/design is recorded in `docs/optimization/portable-io-options.md`. Optional no-pickle versioned NPZ is the selected follow-up direction; current JSON readers/writers/defaults remain compatible. Its production implementation and Windows/Linux/macOS checks form a separate batch after CLI integration. No release/package publication.
