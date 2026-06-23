#!/usr/bin/env bash
# Daily data update for ConcentratedInvestment (Phase 5 cron).
#
# Runs the full daily ETL — OHLCV + features + cross-asset, plus a dated
# sentiment_analyst snapshot so the live analyst signals accumulate history.
#
# Schedule ~22:00 Europe/Berlin (after the US close). Install with `crontab -e`:
#
#     CRON_TZ=Europe/Berlin
#     0 22 * * * /home/he/ai/dev/langgraph/ConcentratedInvestment/scripts/daily_update.sh
#
# (CRON_TZ is honoured by Vixie/cronie; on systems without it, convert 22:00 CET
# to the host timezone, or run the unit under a systemd timer with the TZ set.)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data  # gitignored runtime dir; holds the SQLite db and this log
exec >> data/daily_update.log 2>&1

echo "=== $(date -Is) daily update start ==="
uv run concinvest update --sentiment
echo "=== $(date -Is) daily update done ==="
