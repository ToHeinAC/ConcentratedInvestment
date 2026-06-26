# ConcentratedInvestment — Streamlit app container (Railway deploy).
FROM python:3.11-slim

# Pinned uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/app/nltk_data \
    STREAMLIT_SERVER_PORT=8505 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install the package + deps first for better layer caching (re-runs only when
# pyproject/src change).
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

# Pre-download the VADER lexicon so live sentiment needs no runtime network fetch
# (the lazy loader in features/sentiment.py would otherwise download on first use).
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/app/nltk_data')"

# Streamlit theme + server config (headless, no usage stats), read at startup.
COPY .streamlit ./.streamlit

# Runtime data dir (SQLite db + portfolio CSVs). On Railway a persistent volume
# mounts over this at /app/data; the mkdir keeps the app working without one. Local
# data/ is never copied in — it is gitignored and excluded via .dockerignore.
RUN mkdir -p /app/data

EXPOSE 8505

# Railway runs its own healthcheck; this also covers a plain `docker run`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8505/_stcore/health')"

CMD ["streamlit", "run", "src/concinvest/app/streamlit_app.py", \
     "--server.port", "8505", "--server.address", "0.0.0.0"]
