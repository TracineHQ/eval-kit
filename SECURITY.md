# Security

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Report privately via GitHub's [private vulnerability reporting](https://github.com/TracineHQ/eval-kit/security/advisories/new)
or email the maintainer.

We will acknowledge within 72 hours and coordinate a fix before public
disclosure.

## Supported versions

Only the latest released version of each binding receives security updates.
Once 1.0.0 ships, we will maintain the most recent minor release.

| Version | Supported |
|---|---|
| 0.x     | Yes (latest pre-release) |

## Scope

This is a statistical library with no network, filesystem, or authentication
surface. The realistic threat model is limited:

- **JS binding:** Pure computation. No user-input escape hatches; no `eval`.
- **Python binding:** Uses `numpy` / `scipy`. Report scipy vulnerabilities to
  the scipy maintainers; report our misuse of scipy here.
- **Cross-validation bridge** (`tests/cross-validation/bridge.mjs`): Spawns a
  Node subprocess reading JSON from stdin. Test-only; not shipped to consumers.

Reports outside this scope (e.g., "numpy has a CVE") will be acknowledged
but redirected upstream.
