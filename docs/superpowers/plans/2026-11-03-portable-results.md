# Optional NPZ implementation plan

Parent checkpoint86adf7ae is pushed; all seven CI41679502002 jobs and its exact-head coverage artifact are verified. Implementation is now permitted. The user authorized investigating/switching faster portable I/O; agreed direction is optional NPZ with current JSON compatibility/defaults retained.

## Intended behavior

- Add an explicit bilayer result format choice through public search API, run_find, CLI parser/dispatch and guided advanced questions. Default JSON output, public API shortlist defaults, existing JSON bytes/schema and result ordering remain unchanged. N-layer retains its own experimental JSON path.
- Select `.npz` only when requested. Use a versioned archive with primitive numerical/bool NPY arrays and UTF-8 JSON metadata represented without pickled objects. Fixed-width dtype/shape/byte-order declarations are explicit; matrix row orientation and float64 precision remain those of JSON v2.
- `read_results` recognizes supported archives and returns the same detached, validated document contract as JSON. `write_results` preflights archive suitability before opening/truncating the output. Do not silently narrow large integers, cast mixed fields lossily or discard extension fields; preserve them losslessly or fail clearly for this opt-in format while JSON remains available.
- Start with uncompressed NPZ for predictable write speed. Compression is an explicit lossless option only if its API/help and representative benchmark justify the added choice. Both write paths must work with NumPy1.26; avoid assuming newer `savez(allow_pickle=...)` keyword support. Individual `numpy.save(..., allow_pickle=False)` members via standard ZIP are a compatible building block.
- Archive reading validates required schema/version/member names, duplicates, dtype kinds/object rejection, shapes, row counts and full scientific invariants. NPY headers/declared payload sizes are checked before allocating; invalid inputs produce useful errors. Never enable pickle.
- Manifest consumers explicitly recognize the chosen format: Moire.make, visualization resolution, previews/counts and source-path readers. Existing interface matches and N-layer `results_json` fields do not change silently. Search progress/help names the actual file rather than hardcoding results.json.

## Meaningful acceptance evidence

1. Red-before regressions for complete generated-document roundtrip: empty, one/many candidates, folded/unfolded square/hexagonal and asymmetric symmetry cases; original index/rank order, matrix orientation, counts, numerical bytes and metadata.
2. Explicit tests of unsupported/lossy candidate fields, large integer values, malformed/truncated archives, wrong shapes/dtypes/counts, unknown versions, duplicate members and object-array rejection; preserve existing destination on preflight failure.
3. Direct results and manifest-driven build/preview/plot input readbacks. Retain current one-read preview/count behavior and output reservations. Public Python/CLI/guided format routing must reach the actual writer.
4. Benchmark actual current JSON versus NPZ, including conversion, reconstruction and validation. Report compressed data repetition and allocation limits; arrays-only loading is a separate diagnostic, never a substitute for complete read timings.
5. Portable-format focused CI on Windows/Linux/macOS and compatible Python3.10--3.14 / NumPy1.26/latest combinations. Existing Windows full-suite/package/coverage gates remain. Cross-host source POSCAR paths are preserved as existing JSON paths, not implicitly bundled/relocated.
6. Complete individual function/caller review, focused tests and independent review before freezing one full/package checkpoint. Promote ledger only after evidence, commit/push dev and inspect exact-head CI/coverage. No releases/uploads.

## Existing evidence

Committed consumer map and local measurements: docs/optimization/portable-io-options.md. Prototype script and final machine-readable results are under the shared root `.tmp/portable_io_probe.py` and `.tmp/portable-io-probe.json`, including source hash and equal garbage-collection policy. Actual365-row and labelled20000-row workloads show material NPZ size/write/read benefits; only Windows has been executed so far. At preparation time no archive implementation or new dependency was present. The uncommitted implementation and pending gates are tracked in docs/optimization/part-05d-portable-results.md.


## Lossless extension fallback spike

Chosen refinement: keep numerical NPY columns for ordinary generated rows; use JSON encoded as uint8 NPY for metadata/irregular extension values when numerical conversion would lose types, keys, ragged structure or exact large integers. If per-row key layouts differ, fall back for the candidate list so original ordering/presence survives. No object dtype or pickle is needed. An ordinary NumPy reader can inspect every member with allow_pickle=False. This avoids silently dropping accepted JSON extension data or claiming all inputs fit fixed-width numerical arrays.

A two-pass plain-JSON iterator can count exact ASCII/UTF8 payload bytes before writing the NPY1.0 header, then stream encoding directly to a ZIP member. Place all suitability/serialization planning before opening the production destination. Large atomic strings can still require their own encoder temporary; this is not a universal RSS cap. Reading a JSON member can validate the uint8 header/declared byte size, then decode from the remaining member stream without materializing both a NumPy byte array and another decoded copy. Preserve existing JSON schema/scientific validation after reconstruction.

Owned probe .tmp/npz_json_member_probe.py passed exact JSON-representation roundtrips for2**160+1, mixed2**60+1/float values, ragged lists/None, Unicode, signed zero and dictionary ordering. A6000-record844902-byte JSON member wrote at0.135MiB Python-tracked temporary peak; this is a scratch writer measurement, not production end-to-end performance. np.load(...,allow_pickle=False) read both archives as ordinary uint8 NPY members. Member dtype/shape/header offset/exact payload length were verified. NumPy1.26 documents the public write_array_header_1_0 API; supported-version CI remains an implementation gate. Probe JSON and source hash are retained at .tmp/npz-json-member-probe.json. No production changes.

Portable-format focused matrix: Windows/Linux/macOS, Python3.10/latest-compatibleNumPy, Python3.14/latest-compatibleNumPy, andPython3.12/NumPy1.26.4. Keep existing Windows3.10--3.14 full jobs and add the format subset in the minimum lane. Actual plotting/generation integration remains covered in full/installed checks; a focused archive matrix does not prove every optional rendering backend on every OS.
