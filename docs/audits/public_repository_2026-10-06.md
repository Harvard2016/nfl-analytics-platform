# Public repository audit — 2026-10-06

Base reviewed: `b8e671348216860a88190685453d528908012818` (main after PR #2). This report concerns the public repository and local API code. It is not a penetration test, a data-license opinion or an audit of the owner's device.

## Evidence and findings

| Check | Result | Scope / limit |
|---|---|---|
| Visibility and base | Public; PR #2 merged | Checked through GitHub connection |
| Current tracked files | 650 at the reviewed base | No tracked raw/processed/features, snapshot, local-media, upload-job or model directories |
| Reachable Git history | 31 reachable commits; Gitleaks scanned 29 non-merge commits, about 29.4 MB | Full fetched history and refs available to this checkout; no deleted or unreachable commits |
| Secret candidates | 12 generic-key detections, all public `idempotency_key` hashes | Reviewed against SHA256 construction in `pregame/forecast_v3.py`; not job tokens or credentials |
| Public JSON | No credential-field names detected under public exports | Recursive field check; not a full secret detector |
| Private source media/data paths | No current tracked credential or private-source files | Git path inventory; includes previous history scan for secrets, not visual inspection of every binary |
| npm production lockfile audit | 0 advisories | `npm audit --omit=dev --package-lock-only --json` |
| npm full lockfile audit | 5 high entries from **one** underlying advisory | `braces` plus its lint-tool dependency chain; no published patch |
| Python locked service dependency audit | 72 distributions; 0 advisories | uv export with service extra, no dev; pip-audit with no dependency resolution. Optional `extract` extra not included |
| Local upload trust boundary | Foreign-origin simple POST and foreign Host were not rejected server-side | Fixed on this review branch, with synthetic regression tests |
| Actions | Floating action tags; implicit token permissions; persisted checkout credentials | Pinned commits, read-only permissions, credentials disabled on this review branch |
| Repository upkeep | No security policy, dependency-update config or publication check | Added on this review branch |

## What the secret scanner flagged

The 12 hits were in `reports/forecasts/v3/forecasts_20261004T223731Z.json` and its public forecast export. `idempotency_key` is the first 24 hex characters of SHA256 over game ID, model version, cutoff and source-content hash. It deduplicates append-only forecast records; it does not authorize access.

`.gitleaks.toml` retains default rules and allows only an entire JSON line with this exact field and 24-hex format in those forecast locations. It does not exclude whole directories or allow access-token fields. Scanner reports are redacted and are not committed.

## Remaining dependency advisory

[`GHSA-vfj7-8cjw-p6xm`](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm), CVE-2026-93687: stack exhaustion from deeply nested brace patterns. Published 2026-09-18, reviewed update 2026-10-02; affects `braces` through 3.0.3. The advisory and npm registry list no patched release at this audit date.

The chain is `eslint-config-next` → `@next/eslint-plugin-next` → `fast-glob` → `micromatch` → `braces`. This is a lint/development path in the lockfile, not a dependency in the production-only scan. Keep its inputs controlled and monitor the upstream patch. The suggested `npm audit fix --force` would downgrade the Next lint configuration; it was not applied or used to claim the issue resolved.

## Fixes prepared

- Reject non-loopback Host values and foreign Origin headers before local request parsing; require a loopback bind address.
- Keep allowed local frontend origins, non-browser clients, per-job tokens and disabled access logging.
- Add ignore rules for credentials, source media, archives, private databases, model files and browser output.
- Add a source catalog and publication checker; retain frozen path contracts and source rights.
- Add Gitleaks history scanning in CI, read-only workflow permissions, pinned Actions commits and weekly Dependabot updates.
- Correct stale documentation that still said local-only Git and hard-linked forecast snapshots.

## Account-level checks not verified

The public rulesets endpoint returned an empty list. The separate classic main-branch protection endpoint returned 403 (integration lacks access), so protection is not confirmed either way. The connection does not expose secret-scanning alert state, repository Actions policy or Vercel environment variables. We did not change those account settings. GitHub documentation says automatic secret scanning runs on public repos; that is not evidence that there are no alerts for this account.

The owner should inspect Security alerts and enable push protection, private vulnerability reporting and a main-branch rule requiring reviewed PRs/checks where available. Confirm Vercel has no unintended inference-service URL or secret in `NEXT_PUBLIC_*`. These are separate controls, not things a README or workflow file can enable.

Sources: [GitHub secret scanning](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning), [Actions secure use](https://docs.github.com/en/actions/reference/security/secure-use), [Gitleaks](https://github.com/gitleaks/gitleaks), and the advisory linked above.
