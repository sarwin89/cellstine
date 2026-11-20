# Final pending source review: visualization, CLI and reference tools

Base dev7466c93b76c9a63560664b95c578320d1ae4508c passed all16 jobs of
CI44530971712. Downloaded artifact14535286392 digest
5511716a4d3af9dd58d6bcae56685722a9a7907491ac0832c04f355e98e3cd68
matches its head/job/module/statement sets. Coverage14595/16183 statements,
4232/5384 branches; only known prompts.py501–502 differed from local coverage.
This gate closed before any Part09 runtime edits.

## Reviewed scope and retained contracts

All96 callable bodies in16 modules were independently reviewed, including
nested functions and lambdas. Individual LF-normalized module hashes, body AST
hashes, prior dispositions, current rationales and evidence support the ledger.

| Source area | Bodies | Scope |
| --- | ---: | --- |
| visualize/backends/matplotlib.py |16|Optional import, element radii, projection/depth ordering, cells, structure/summary rendering and selected rows|
| visualize/backends/plotly.py |2|Cell outline and structure HTML|
| visualize/results/plotly.py |6|Frame data, size, rich metadata, HTML and validated result routing|
| CLI plain/rich/spec and five workflow parser modules |12|Lazy imports, frontend routing, defaults/aliases, validation and help|
| benchmarks/benchmark_gram_search.py |9|Independent-reference comparison, signatures, tolerances and timings|
| benchmarks/reference_moire.py |12|Independent coefficient/pair/Gram/coincidence/angle enumeration and reference search|
| core/jsonio.py |3|Existing bounded indented writer and new private inline JSON encoder|
| defect/workflow.py |8|Re-review after supercell doc correction; executable AST unchanged|
| defect/supercell.py |28|Re-review after doc/example correction; executable AST unchanged|

This closes56 pending bodies and adds one helper; the other39 bodies are
re-reviewed because their modules changed. The original numerical and physical
conventions, cutoff/tie policies, validated JSON/NPZ data, source order and
public workflow signatures remain. The independent reference does not reuse
production search arithmetic. CLI builders remain unchanged except honest
supercell help; numerical twist limits are ultimately validated as finite and
in[0,180] by SearchConfig, so no new acceptance policy is invented in spec.py.

## Implemented changes

`_dumps_inline_json` uses the default JSON encoder and escapes literal less-than
signs. HTML interprets script boundaries before JavaScript interprets strings;
an ordinary JSON-quoted `</script>` could therefore escape generated viewer data.
Both structure and result viewers now keep those boundaries out of every
embedded JSON field. Decoded text, Unicode, coordinates, units and rich `<br>`
metadata retain their values. This is a raw-script boundary repair, not a claim
that arbitrary future HTML/Plotly extensions are universally sanitized.

Structure HTML computes radius-derived marker size once per distinct expanded
species label, preserving per-atom ordering, zero-count labels and exact sizes.
Static result summaries filter candidate indices before creating plotting-row
dictionaries, preserving source ordering, copied outer rows, selected values,
payload ownership and missing-selection errors. Complete HTML/frame/selected
output still requires proportional storage; no output cap is introduced.

Both static renderers now close their own figure in finally through plotting,
export and show. An existing summary construction leak was caught during peer
review: plt.subplots registered a figure before try. It is now created first,
with fig.subplots protected inside try. Save, show and construction failures
propagate and leave caller-owned figures open.

Review also confirmed a preexisting --show bug: unconditional forceAgg defeated
the documented GUI request. Default export still selects Agg; requested show
respects Matplotlib's configured backend. A headless/configured Agg stays Agg,
and no GUI restoration after an earlier Agg selection is promised. Branch and
forwarding tests verify the policy; no interactive GUI demonstration is claimed.

The independent reference returns empty short-vector arrays with shape(0,2),
allowing a valid small length bound with no admitted vectors to return no
candidates instead of IndexError. Nonempty coefficient arithmetic is unchanged.

The final archive audit also found the source distribution included the tests
but omitted tests/conftest.py, their fixture/bootstrap module. MANIFEST.in now
includes it explicitly. Final source-archive readback checks all299 frozen Python
files, including this bootstrap, rather than only selected test files. Runtime
Python source and dependencies are unchanged by this packaging correction.

Supercell docs/help now state image separation is ranked first, followed by a
heuristic shape tie-breaker. Global reduced-basis roundness and arbitrary-order
independence are not certified by the32-entry cheap shortlist. The FCC example
now correctly uses four primitive cells forming the conventional cubic cell
(separation a), versus a/sqrt(2) for four-cell diagonal repeats. Scientific
executable ASTs in both defect modules match the predecessor after doc removal.
The existing investigated _roundest disposition remains; it is not "fixed" by
correcting prose.

## Regression, review and compatibility evidence

The initial new batch failed28 tests and passed5 against the predecessor.
Construction refinement failed1 summary case while1 structure case passed.
Show-policy regression failed4 cases and passed2 before its fix. The final
affected batch passes201 tests in12.39s. There are41 new regression cases;
optional Matplotlib cases remain skippable in core/minimum-dependency lanes.
Both new files run in minimum-NumPy and9 portable OS/Python jobs.

The final independent96-body review records both P2 findings resolved and no
open finding. The concrete secondary-roundness deferral is preserved. Current
Python source/test hashes match its reviewed snapshot. Earlier195-test and
pre-show benchmark snapshots are not evidence for the final source freeze.

Actual predecessor modules were compiled separately. Checks passed:8 summary
views,2 ordinary HTML byte identities,3 decoded HTML values (including attack
text),2 byte-identical static PNGs,6 nonempty independent reference signatures
and2 scientific executable-AST identities. These are distinct from whole-suite
or supported-version evidence.

## Measurements

Python3.14.3/NumPy2.4.3; native env1; seven warm interleaved medians; separate
tracemalloc peaks, not RSS. No other local verification runtime was active.
Summary measurements isolate row preparation after the same already-validated
payload boundary: they do not include result decoding/disk I/O. HTML cases
include actual file writes and readbacks. Tiny-case costs remain visible.

| Case | Input | Selected | Old ms | New ms | Old traced bytes | New traced bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Summary rows |3|all|0.0036|0.0036|1856|1856|
| Summary rows |3|1|0.0041|0.0020|2016|1200|
| Summary rows |10000|all|11.9575|11.8937|4725552|4725552|
| Summary rows |10000|1|12.4028|0.8685|4725712|1200|
| Structure HTML |2 atoms|all|2.6950|2.9419|145294|145470|
| Structure HTML |10000 atoms|all|80.6949|70.2450|3575259|3455467|

One selected row is about14.3x faster in this isolated phase. Large HTML export
is about13% faster in this sample; the two-atom export costs about0.25ms and
176 traced bytes. This is not a universal renderer or complete-workflow claim.

## Checkpoint and completion boundary

The frozen full suite passed **4113 tests**, with 4 skips and
81 dependency warnings in 747.81s. All299 Python hashes
remain unchanged. Local package coverage: **14604/16192
statements**, **4237/5386 branches**.
Wheel/sdist/Twine and three installed core/CLI/sdist environments pass actual HTML
boundary/decoded-coordinate/native-NPZ summary checks, previous physical
workflows and dependency checks. Final documentation/bootstrap packaging changes
refresh the source archive; byte readback must match all299 frozen Python files,
all144 runtime modules in both archives and8 final support files before push.

The canonical ledger now records **1238 functions with inspection evidence**:
**865 reviewed unchanged**, **359 optimized and verified**,
**14 investigated deferrals**, **0 pending**. This batch reviews96 bodies and
adds57 newly inspected bodies, including one private helper. All-function
inspection is complete; it does not remove the14 documented limitations or
certify an unrestricted release. [The comprehensive audit](release-preparation-audit.md)
describes the initial/current boundaries and outstanding maintainer decisions.

For a candidate commit, all16 exact-head CI jobs and downloaded coverage remain
separate required evidence; their status must be read from that commit rather
than inferred from this source-tree report. The completion readback also checks
the objective requirement by requirement. No main merge, tag, upload, publisher
configuration or release is authorized by these local gates.
