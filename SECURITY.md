# Security

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Report privately via GitHub's [private vulnerability reporting](https://github.com/TracineHQ/eval-kit/security/advisories/new).

Acknowledgement within 72 hours. Fixes land before public disclosure.

## Supported versions

Only the latest released version of each binding receives security updates.
From 1.0.0, the latest minor release is supported.

| Version | Supported |
|---|---|
| 0.x     | Yes (latest pre-release) |

## Scope

This is a statistical library with no network or authentication surface.
The realistic threat model is limited:

- **Libraries** (`js/lib/stats.mjs`, `eval_kit.stats`): pure computation on
  the numbers passed in. No `eval`, no I/O.
- **CLIs** (`js/cli.mjs`, `eval-kit`): read the JSON score files named on the
  command line (or stdin) and print to stdout. They never write files, run
  commands, or touch the network. Malformed or hostile input is rejected with
  exit code 2; a crash, hang, or traceback on any input is a bug worth
  reporting.
- **Claude Code skill** (`SKILL.md`): pre-approves exactly one command,
  `node ${CLAUDE_PLUGIN_ROOT}/js/cli.mjs *`. If you find a way for the skill
  to widen that, report it here.
- **Python binding:** Uses `numpy` / `scipy`. Report scipy vulnerabilities to
  the scipy maintainers; report our misuse of scipy here.
- **Cross-validation bridge** (`tests/cross-validation/bridge.mjs`): Spawns a
  Node subprocess reading JSON from stdin. Test-only; not shipped to consumers.

Reports outside this scope (e.g., "numpy has a CVE") will be acknowledged
but redirected upstream.
