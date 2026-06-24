"""Safe-exit helper for the Streamlit app.

Shuts down **only this Streamlit server's own process** (SIGTERM to the current
PID). It never runs an ``lsof``/port kill, so any other process that shares or
forwards the app's port (e.g. an IDE remote-server port-forward, or an SSH tunnel)
is left untouched.
"""

from __future__ import annotations

import os
import signal


def shutdown(sig: int = signal.SIGTERM) -> None:
    """Terminate the current Streamlit server process — and only it.

    Signals this very process by PID, so the port is released by the app exiting
    rather than by killing whatever is bound to it.
    """
    os.kill(os.getpid(), sig)


def render(st) -> None:
    """Render the safe-exit button into a Streamlit sidebar/page.

    ``st`` is passed in to avoid importing streamlit at module load.
    """
    if st.button("⏻ Stop app"):
        st.warning("Shutting down the app…")
        shutdown()
