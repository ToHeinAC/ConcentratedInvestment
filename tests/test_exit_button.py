"""Tests for the safe-exit helper (Phase 0)."""

import os
import signal

from concinvest.app import exit_button


def test_shutdown_signals_only_current_process(monkeypatch):
    # shutdown() must SIGTERM this very process — never an lsof/port kill.
    calls = []
    monkeypatch.setattr(exit_button.os, "kill",
                        lambda pid, sig: calls.append((pid, sig)))
    exit_button.shutdown()
    assert calls == [(os.getpid(), signal.SIGTERM)]
