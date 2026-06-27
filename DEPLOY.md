# Deploying ConcentratedInvestment (Railway)

Beginner-friendly managed deploy: the Streamlit app + a persistent SQLite volume + a
daily cron that refreshes data and **emails you when your portfolio has a buy/sell
trigger**. ~$5/mo on Railway's Hobby plan.

The repo already ships everything needed: the [`Dockerfile`](Dockerfile),
[`railway.json`](railway.json) (app start command), [`scripts/daily_update.sh`](scripts/daily_update.sh)
(ETL + notify), the `concinvest notify` command, and [`.env.example`](.env.example).

---

## 1. Email provider (5 min, do this first)

Automated mail through Hotmail/Gmail SMTP gets throttled and spam-filed — use a
transactional API. We use **Resend** (free tier 3k emails/mo):

1. Sign up at <https://resend.com>, create an **API key**.
2. To send *to* your Hotmail you can use Resend's sandbox sender
   (`onboarding@resend.dev`) immediately; for a custom `from` address, verify your
   domain in Resend first.

## 2. Create the Railway project

1. Push to GitHub (done). At <https://railway.app> → **New Project → Deploy from GitHub
   repo** → pick this repo. Railway reads `railway.json` and builds the `Dockerfile`.
2. **Add a Volume** (service → *Variables/Settings → Volumes*): mount path **`/app/data`**.
   This is where the SQLite db + portfolio CSVs live; the volume makes them survive
   redeploys. (WAL mode is enabled in code so the app reads while the cron writes.)
   The image sets `CONCINVEST_DATA_DIR=/app/data` so the app **and** the cron service
   actually read/write there — without it the data path is derived from the package's
   install location (site-packages, not `/app`), and the volume would capture nothing.
   Attach the **same** volume to the cron service (§4) so both see one shared db.
3. **Variables** tab — set (values from `.env.example`):
   `RESEND_API_KEY`, `ALERT_EMAIL_TO`, optional `ALERT_EMAIL_FROM`, and
   `ALERT_PORTFOLIO` (the saved portfolio name to watch).
4. Deploy. Railway gives you a public `*.up.railway.app` URL (Settings → **Networking →
   Generate Domain**). Add a custom domain there later if you want.

## 3. Seed the data (one-off)

The volume starts empty. From the service's **Shell** (or locally against the volume),
run once to populate the SQLite db and create a portfolio:

```bash
uv run concinvest update --sentiment        # fills data/concinvest.sqlite
```

Then open the app, build your book in **Live: Sample Portfolio**, and **💾 Save** it —
that writes `data/portfolios/<name>.csv` on the volume. Set `ALERT_PORTFOLIO=<name>`.

## 4. The daily cron + email

Add a **second service** in the same project, pointing at the **same repo** (so it shares
the image) and the **same volume** (`/app/data`):

- **Start command:** `bash scripts/daily_update.sh`
- **Cron schedule:** `0 20 * * *`  — Railway cron is **UTC**. 22:00 Europe/Berlin is
  20:00 UTC in summer (CEST) / 21:00 UTC in winter (CET); pick one, or run an hour early.

Each run does `concinvest update --sentiment` then, if `ALERT_PORTFOLIO` is set,
`concinvest notify --portfolio "$ALERT_PORTFOLIO"`. The notify step is **non-fatal** and
**only emails when something fires** — a buy/sell in the ML forecast or a mandatory
risk-rule sell. Nothing fired → no email.

Test it on demand from the Shell:

```bash
uv run concinvest notify --portfolio <name>      # prints the alert if RESEND_* unset
```

---

## Notes

- **One image, two services** — the web app (long-running) and the cron (runs, exits).
  Don't run cron inside the web container.
- **SQLite is fine here** — single user; WAL handles app-read / cron-write. Add a nightly
  `sqlite3 data/concinvest.sqlite ".backup data/backup.sqlite"` if you want belt-and-braces.
- **Keep VADER (default) sentiment in prod** — the FinBERT `sentiment` extra pulls in
  torch and bloats the image/deploy.
- **Secrets** live only in Railway's Variables (or a local `.env`, gitignored) — never
  committed.
- **Self-hosting instead?** The same `scripts/daily_update.sh` + env vars work under a
  host cron / systemd timer on any VPS; only the scheduling UI differs.
