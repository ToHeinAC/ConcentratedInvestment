"""Tests for the safe-exit helper (Phase 0)."""

import os

from concinvest.app import exit_button


def test_current_port_prefers_env(monkeypatch):
    monkeypatch.setitem(os.environ, "STREAMLIT_SERVER_PORT", "8523")
    assert exit_button.current_port() == 8523


def test_current_port_default_is_above_8510(monkeypatch):
    monkeypatch.delitem(os.environ, "STREAMLIT_SERVER_PORT", raising=False)
    # default fallback must respect the >8510 rule
    assert exit_button.current_port() > 8510


def test_is_ssh_for_current_process():
    # The test runner is not ssh, so this must be False.
    assert exit_button._is_ssh(str(os.getpid())) is False
