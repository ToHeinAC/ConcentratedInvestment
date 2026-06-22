"""Tests for the safe-exit helper (Phase 0)."""

import os

from concinvest import config
from concinvest.app import exit_button


def test_current_port_prefers_env(monkeypatch):
    monkeypatch.setitem(os.environ, "STREAMLIT_SERVER_PORT", "8523")
    assert exit_button.current_port() == 8523


def test_current_port_default_matches_config(monkeypatch):
    monkeypatch.delitem(os.environ, "STREAMLIT_SERVER_PORT", raising=False)
    # default fallback must be the project-configured port
    assert exit_button.current_port() == config.STREAMLIT_PORT
    assert config.STREAMLIT_PORT == 8505


def test_is_ssh_for_current_process():
    # The test runner is not ssh, so this must be False.
    assert exit_button._is_ssh(str(os.getpid())) is False
