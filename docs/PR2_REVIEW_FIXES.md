# PR #2 review fixes — 2026-10-06

This is an engineering fix pass on `feat/inference-and-model-v3`. It does not retrain a model, rescore a benchmark,
change the champion, overwrite a forecast record, or publish private data/media.

- **Forecast inputs:** new snapshots copy independent bytes instead of linking to mutable raw files. All file hashes,
  sizes and the manifest hash are checked before fitting, again after fitting, and before exporting forecasts.
  Export also verifies each older snapshot referenced by a forecast. Missing or changed snapshots stop export;
  recorded hashes and forecasts are never rewritten to make changed inputs pass. Keep the private snapshots needed
  to reproduce the forecast log. Earlier snapshots remain usable if their bytes still match their original hashes.
- **Clip timing:** the service returns exact audio-bin bounds in uploaded-file seconds. Delayed audio, shorter audio
  and the final partial bin are handled explicitly. Speech times and selected clips use the same origin. The graph,
  pointer seeking and playhead use video seconds, with unanalysed regions shaded instead of shown as silent audio.
- **Review exports:** candidates start unreviewed. A person must confirm each clip; editing bounds, replay status or
  removal clears that confirmation. Machine speech is labelled predicted rather than observed.
- **File switches:** changing files clears the old result and job binding. Polls, result fetches and deletions check
  their generation before updating state. Both upload pages reject stale responses; switching cancels the old job
  on a best-effort basis. The file picker is disabled while the upload request is being sent.
- **CI:** the service extra and ffmpeg are installed. API admission, cancellation, ownership, completion, deletion
  and media timing run without private weights. A separate configured browser suite mocks service responses and
  tests time seeking, exported review flags and late result responses. The public build still accepts no uploads.
- **Fresh coverage wording:** the existing 2024 numbers are unchanged. Its interval is now explicitly a play-level
  Wilson interval, which assumes independent plays and does not account for games shared by those plays.

Local validation: 80 Python tests passed, 8 skipped because required private data/models are absent; 9 web unit tests
passed; ruff, ESLint, TypeScript and production build passed. Local Chromium startup is blocked by the execution
environment's socket restrictions; browser validation is delegated to the PR's GitHub Actions workflow.

Trained highlight inference on arbitrary new video, event detection, public hosted inference and video-to-coverage
remain outside this fix pass and keep their existing limitations.
