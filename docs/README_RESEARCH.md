# How the repository presentation was chosen

Reviewed on 2026-10-06. These are structure references, not code dependencies or claims that this project has their scale. No text or artwork was copied.

| Public project / source | Useful pattern | Applied here |
|---|---|---|
| [Supabase](https://github.com/supabase/supabase) | Product and documentation links early; separate apps, packages and supporting tools | Website first, one frontend and separate Python module packages |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Clear visual identity, short working examples, deeper docs outside the README | Product screenshots, a short web-only start, linked model cards |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Visual task examples and quantitative comparisons beside useful commands | Real interface previews and charts generated from saved metrics |
| [Cal.diy](https://github.com/calcom/cal.diy) | Product screenshots, hosted/self-hosted distinction, detailed setup separated from overview | Public exploration separated from the local upload lab |
| [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) | Separate original inputs, transformations, features, models and reports | Data catalog, layer map and folder-level indexes; existing path contracts preserved |
| [GitHub Actions secure-use guide](https://docs.github.com/en/actions/reference/security/secure-use) | Minimum token access and action versions pinned to commit hashes | Read-only workflows, disabled persisted checkout credentials and pinned actions |
| [Gitleaks](https://github.com/gitleaks/gitleaks) | Scan Git history, redact findings, narrowly explain false positives | History scan in CI; a field-and-path exception for public forecast deduplication hashes |

The README supports two visits: a recruiter can open the website and see the product in under a minute; an engineer can then follow the evidence into code, datasets, model cards, runs and tests. Numerical graphics are drawn from committed reports. UI screenshots are captured from the deployed site. Artwork is documented separately in [ASSETS.md](ASSETS.md).
