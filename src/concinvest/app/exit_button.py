"""Safe-exit helper for the Streamlit app.

Dynamically finds the port the current Streamlit server is bound to (the app runs
on port 8505 by project convention) and kills only that process group. It never
touches an SSH connection.
"""

from __future__ import annotations

import os
import subprocess

from concinvest import config


def current_port(default: int = config.STREAMLIT_PORT) -> int:
    """Best-effort discovery of the running Streamlit server port."""
    # Streamlit exposes its port via this env var when launched with --server.port.
    env_port = os.environ.get("STREAMLIT_SERVER_PORT")
    if env_port and env_port.isdigit():
        return int(env_port)
    try:
        from streamlit import config as st_config  # local import: optional dep

        port = int(st_config.get_option("server.port"))
        if port > 0:
            return port
    except Exception:
        pass
    return default


def kill_port(port: int) -> None:
    """Kill processes listening on ``port`` without disturbing SSH.

    Mirrors: ``lsof -ti:PORT | xargs -r kill -9`` but filters out anything whose
    command name contains "ssh" as a safety guard.
    """
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return  # lsof unavailable; nothing we can safely do

    pids = [p for p in out.stdout.split() if p.isdigit()]
    for pid in pids:
        if _is_ssh(pid):
            continue
        subprocess.run(["kill", "-9", pid], check=False)


def _is_ssh(pid: str) -> bool:
    """Return True if the process command name references ssh."""
    try:
        comm = subprocess.run(
            ["ps", "-p", pid, "-o", "comm="],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return "ssh" in comm.stdout.lower()


def render(st) -> None:
    """Render the safe-exit button into a Streamlit sidebar/page.

    ``st`` is passed in to avoid importing streamlit at module load.
    """
    port = current_port()
    if st.button(f"⏻ Stop app (port {port})"):
        st.warning(f"Shutting down Streamlit on port {port}…")
        kill_port(port)
