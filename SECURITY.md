# Security

Gridiron Lens is a public portfolio website with an optional **local-only** upload service. It is not a hosted multi-user inference API.

## Report a problem privately

If the repository's Security tab offers **Report a vulnerability**, use that private channel. Otherwise, contact the repository owner privately through a contact method they provide before sharing details. Private vulnerability reporting is an account setting and is not enabled by this file.

Do not put credentials, job access tokens, private clips, personal information or an exploit containing them in a public issue. A public issue may describe a general non-sensitive bug without those details.

## Trust boundaries

| Surface | Boundary |
|---|---|
| Vercel website | Reads committed public exports; no upload-service endpoint is configured |
| `apps/web/public/` | All contents are public and downloadable |
| Local API | Loopback bind, loopback Host checks and explicit allowed browser origins |
| Upload jobs | Per-job tokens; generated storage names; media assets checked against a job manifest |
| Model files | Trusted locally produced artifacts only; never accept uploaded pickle/joblib/model files |
| Forecast sources | Independent snapshot copies; file sizes and SHA256 verified before forecasting and publication |
| Dataset display | Source-specific rights review; raw inputs and private source media stay out of Git |

CORS alone does not prevent a browser from sending a simple POST. The local API rejects foreign `Origin` headers before parsing a request. Non-loopback Host values are rejected to reduce DNS-rebinding risk. Command-line clients with no Origin still work; this does not make the service suitable for public hosting.

Do not use public tunnels, `0.0.0.0`, forwarded ports or wildcard origins. Multi-user hosting would require separate authentication, quotas, isolation and a new security review. Keep ffmpeg and local ASR binaries updated; parsing an untrusted media file carries risk even when it stays local.

## Repository checks

- `python scripts/check_repository.py` checks tracked paths, the catalog, public JSON credential fields and local documentation links.
- Gitleaks scans reachable Git history in CI with redacted findings. Its narrow exception covers a public SHA256-derived forecast deduplication field, not access tokens.
- Workflows request read-only repository access, pin action commits and do not persist checkout credentials.
- Dependabot checks npm, uv and Actions dependencies weekly.
- Lockfiles are committed; dependency updates still need tests and review.

If a real secret is found, revoke or rotate it first. Removing a current file does not remove earlier commits or copies already made by other people. Do not rewrite public history without coordinating with the owner.

## Known issue and audit scope

The [2026-10-06 audit](docs/audits/public_repository_2026-10-06.md) found no actual credentials in the scanned reachable history, no tracked private source media or raw datasets, and no advisories in the scanned production JavaScript or locked Python service dependencies. It records one unresolved **development-tool** `braces` advisory with no published patch. This is not a promise that the whole system is free of security bugs.

Account settings, security alerts, branch protection, Vercel environment variables, old CI artifacts, deleted/unreachable commits and the owner's computer are outside this audit. Check those separately before opening any public upload service.
