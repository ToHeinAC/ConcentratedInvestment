"""Email-on-trigger notification (Phase 5 deploy).

Turns a live recommendation (the 5-field ML forecast + the strategy's mandatory
risk-rule sells) into an email **only when something fires**. Split into a pure,
offline-testable ``build_alert`` and a thin ``send_email`` that posts to the Resend
transactional-email API (the one new outbound-network call outside ``data.fetch``;
kept here so the rest of the package stays pure).

Config comes from the environment (never committed): ``RESEND_API_KEY``,
``ALERT_EMAIL_TO``, optional ``ALERT_EMAIL_FROM``. See ``.env.example`` / ``DEPLOY.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

_TIER_LABEL = {1: "stock", 2: "2x", 3: "3x", None: ""}


@dataclass
class Alert:
    subject: str
    body: str


def build_alert(forecasts, actions, *, portfolio_name: str, strategy: str) -> Alert | None:
    """Render an :class:`Alert` from a recommendation, or ``None`` when nothing fired.

    ``forecasts`` is the 5-field ML forecast list (``ml.forecast.Forecast``); ``actions``
    is the strategy's mandatory sells (``portfolio.rules.Trade``). Pure — no network."""
    if not forecasts and not actions:
        return None
    lines = [f"Portfolio: {portfolio_name}   (strategy: {strategy})",
             f"Date: {date.today().isoformat()}", ""]
    if actions:
        lines.append("Strategy actions — mandatory risk-rule sells:")
        lines += [f"  - {t.action.upper()} {t.ticker} {_TIER_LABEL.get(t.tier, '')}"
                  f" — €{t.amount_eur:,.0f}" for t in actions]
        lines.append("")
    if forecasts:
        lines.append("ML + news/sentiment signals:")
        lines += [f"  - {f.action.upper()} {f.ticker} {f.leverage}"
                  f" — €{f.amount_eur:,.0f} (confidence {f.confidence:.0%})"
                  for f in forecasts]
        lines.append("")
    lines.append("Open the app to review and act. Automated alert — not advice.")
    n = len(actions) + len(forecasts)
    subject = f"[ConcInvest] {n} trigger(s) for {portfolio_name} ({strategy})"
    return Alert(subject=subject, body="\n".join(lines))


def email_config_from_env() -> dict | None:
    """Resend credentials from the env, or ``None`` if unconfigured (alert then prints
    instead of sending — so a missing key never breaks the cron)."""
    api_key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("ALERT_EMAIL_TO")
    if not api_key or not to:
        return None
    sender = os.environ.get("ALERT_EMAIL_FROM", "ConcInvest <onboarding@resend.dev>")
    return {"api_key": api_key, "to": to, "sender": sender}


def send_email(alert: Alert, *, to: str, api_key: str, sender: str) -> None:
    """POST the alert to the Resend API (raises on a non-2xx response)."""
    import requests

    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"from": sender, "to": [to], "subject": alert.subject, "text": alert.body},
        timeout=30,
    )
    resp.raise_for_status()
