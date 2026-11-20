# CI token permissions follow-up

On 2026-11-19 the GitHub code-scanning API reported two open medium-security
warnings, [alert 1](https://github.com/sarwin89/cellstine/security/code-scanning/1)
and [alert 2](https://github.com/sarwin89/cellstine/security/code-scanning/2).
Both use `actions/missing-workflow-permissions` in `.github/workflows/ci.yml`.
Their only reported instances are on `main` at `bb90941471c96d6f93426107e06c1caba5208b56`.
The current dev workflow also omitted explicit permissions, so this was an
outstanding configuration defect rather than merely stale alert presentation.

The workflow now declares `permissions: contents: read` at its root. This applies
to all four jobs (including matrix expansions); no job requests additional token
permissions. The jobs read the checkout, install/test/build, and upload the
coverage artifact. They do not publish, push commits, or write issues/PRs.
The change follows GitHub's [least-permission guidance](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token).

This follow-up changes CI configuration and documentation only. Runtime source,
tests, dependency requirements, scientific behavior and function-review counts
are unchanged from dev `a18b2b8baaea4b874208a16f9a13aa618f5b40c5`, whose all16 CI
jobs passed in [run44477217790](https://github.com/sarwin89/cellstine/actions/runs/44477217790).
Its downloaded coverage artifact14507627405 has verified SHA256
`f782daea039639697806bd396bd73fd86bf9f4349be62b95739e534447e3a4c4`;
CI covers14451/16081 statements and4138/5310 branches, with only the known two
prompts.py fallback statements beyond local coverage and no branch delta.

The fix is prepared on `dev`. Alert closure on the default `main` branch still
requires integration and a successful subsequent code scan there. No alert is
dismissed manually, and a dev fix must not be described as GitHub-confirmed
default-branch closure. No merge to main, release, tag or package upload occurs.

## Verified dev CI closure

Fix dev5fd5be33f892004b7d6934029adf1a1eb6cf943b passed all16 jobs in
CI44479741997, including retained coverage upload with contents:read.
Artifact14507809942 SHA256
01155cfa9e6eb62b52b865cef9d537c29a05a5b470f157269651ba015c9ed4a6
was verified against exact head/job and unchanged coverage:14451/16081
statements,4138/5310 branches, zero delta from a18b2b8. All292 Python hashes
were unchanged. This closes the dev configuration check; default-main alert
closure still requires later integration/rescanning. No alert dismissal occurs.
