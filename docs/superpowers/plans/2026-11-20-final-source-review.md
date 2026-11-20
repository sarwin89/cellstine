# Final pending visualization, CLI and reference review implementation plan

Execute inline with the existing writing/executing-plans workflow, regression
first, then one independent complete-batch source reviewer. Part08c closed before runtime edits: dev7466c93, all16CI44530971712 jobs,
artifact14535286392 SHA2565511716a4d3af9dd58d6bcae56685722a9a7907491ac0832c04f355e98e3cd68.
Head/job/digest/module/statement sets verified; only known prompts501-502 delta.

Goal: complete the56 pending callable reviews while fixing witnessed HTML
raw-script interpolation, bounded selected-row preparation, repeated marker
lookup, figure ownership on export failures and empty reference-vector shape.
No physical search/builder/tolerance or concurrency-default changes.

Scope: all24 bodies of three visualization backends/results modules (23 pending,
one previously inspected build_visualization);12 plain/rich/spec/parser bodies;
21 independent benchmark/reference bodies. Add `_dumps_inline_json` to existing
core/jsonio.py and re-review its2 existing bodies. Correct overstatements of the
already-investigated secondary roundness shortlist in supercell.py docs,
workflow.supercell docs and plain_defect help: re-review those28+8 bodies,
retaining all scientific code and the existing investigated deferral. Expected
whole-batch review96 bodies,1 new callable, final inventory1238. Refresh exact
AST inventory rather than treating this estimate as proof.

Architecture: JSON default encoding with escaped literal < in inline scripts,
shared by structure/result viewers; decoded strings/numbers/rich <br> subtitles
stay identical. Filter requested summary candidates before projecting plotting
rows; preserve source order, row ownership and empty-selection errors. Cache
marker sizes only per distinct expanded label. Own each Matplotlib figure in a
try/finally through its complete construction/export/show path; close exactly
that figure and propagate failures. Correct independent _short_vectors empty
shape to(0,2), keeping brute-force arithmetic independent of production search.

- [x] Confirm Part08c all16CI/head/digest/coverage closure; keep dev clone clean
  until then. Move this reviewed draft and test drafts into scoped tracked paths.
- [x] Run new meaningful cases against the predecessor first: raw HTML parser
  script boundaries, decoded field/numeric fidelity, real repeated radius calls,
 10000-candidate selected-row allocation, filter/source order/ownership, missing
  selections, figure save/show errors with caller-owned figure still open,
  empty-reference shape/search. Keep optional Matplotlib tests skippable.
- [x] Implement the five scoped changes; retain normal rendered geometry,
  formatting, UTF-8 and default JSON encoder semantics. Avoid new valid-empty
  structure policy; cleanup applies to existing failures as well.
- [x] Align roundness/FCC example documentation with actual image separation,
  bounded shape shortlist and observed order dependence. Verify stripped-doc
  executable AST equality in the two scientific modules. Keep the investigated
  _roundest disposition; doc correction does not certify global optimality.
- [x] Complete all remaining function/caller/test review. Plain CLI builders
  should remain unchanged except honest help copy; keep parser laziness,
  legacy errors/defaults, plain/rich routing and numerical validator contracts.
  Independent reference arithmetic remains deliberately simple and separate.
- [x] Run quiet actual predecessor/current measurements for selected/all rows
  and tiny/large structure HTML exports, with exact ordinary output/decoded
  metadata comparisons, separate traced peak and small-case costs. Do not
  claim bounded complete HTML/frame output or universal rendering acceleration.
- [x] One independent96-body review, resolve findings, freeze Python hashes;
  focused/full coverage/build/Twine/3installed/native/HTML data readbacks, then
  scoped ledger/report/publish-plan updates and final support-only sdist/raw
  archive comparison. Add portable/minimum NumPy coverage for new regressions.
- [x] Scoped commit and push dev;16 exact-head CI/downloaded coverage closure.
  Only afterward perform the final requirement-by-requirement audit, re-query
  GitHub scan alert state, inspect14 investigated limitations, source/package/
  documentation boundaries, publication plan and initial-versus-current report.
  All-function review is necessary, not sufficient, to mark the goal complete.

Preserve shared main/Claude edits and scientific runs. No main merge, tag,
release, PyPI/TestPyPI upload, alert dismissal or artifact publication.

Review confirmed --show was defeated by unconditional Agg. Keep default headless export; pass requested show policy to _pyplot so configured interactive backends can be used. Headless/previously configured Agg remains Agg; no promise of GUI backend restoration. Add backend-selection and renderer-forwarding regressions.

Local299-Python full/package/installed gates, runtime integration, exact-head16CI/artifact and source objective audit pass; final documentation/archive/latest-head readback is recorded separately.

## Final source checkpoint closure (2026-11-20)

Runtime source commit `37d2137dbe24216701f5b7c706c1ebd051175ef3` passed all16 jobs of
[CI44666671915](https://github.com/sarwin89/cellstine/actions/runs/44666671915).
Downloaded coverage artifact14599036843 SHA256
`114e2048d98dbafa2ab88ae5be103a67d8efa619f92e889c78fc63cf45346afc` matches the head/job/digest/module/statement sets.
CI coverage:14606/16192 statements,4237/5386 branches; only the known
prompts.py501-502 fallback differs from local coverage, with no branch delta.
The canonical ledger contains1238 inspected callables,0 pending and14 concrete
investigated deferrals. Full source4113tests/4skips/81warnings,201affected cases,
independent96-body review and package/installed/readback gates passed.

The documentation closure records those results and retains the same299
Python source hashes. Completion for the latest dev head also requires its
own CI/readback; neither this source checkpoint nor its coverage percentage
certifies the14 retained numerical/surface limitations. No main merge,
publication, publisher configuration, tag or release was performed.
