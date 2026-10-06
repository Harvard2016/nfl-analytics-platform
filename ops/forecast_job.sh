#!/bin/sh
# One scheduled forecast run: fetch and preserve the sources, write forecast records, refresh the site export.
# Fails loudly (non-zero exit, message in the log) if the sources cannot be fetched; stale files are never used silently.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
echo "== forecast job $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bin/pregame-lens snapshot || { echo "SOURCE REFRESH FAILED: no forecast written"; exit 2; }
bin/pregame-lens forecast-v3 || { echo "FORECAST FAILED"; exit 3; }
echo "done. Review and commit reports/forecasts/v3 and apps/web/public/demo/pregame/forecasts.json to publish."
