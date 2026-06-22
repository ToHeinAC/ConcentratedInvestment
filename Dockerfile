# ConcentratedInvestment — Streamlit app container (Phase 0 stub).
FROM python:3.11-slim

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

# Streamlit runs on a port > 8510 per project convention.
EXPOSE 8511
ENV STREAMLIT_SERVER_PORT=8511 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "src/concinvest/app/streamlit_app.py", \
     "--server.port", "8511", "--server.address", "0.0.0.0"]
