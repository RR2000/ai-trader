# Paper-trading service: the simulations of simulations.toml, NumPy inference of the 2023 model, no TensorFlow in the
# image (329 MB in all, measured in CI).
# Python 3.14: 3.15 is out but h5py has no wheels for it yet.
FROM python:3.14.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    MODEL_PATH=/app/model/trading_model_episode_30.h5 \
    PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt && pip check

COPY src/model/completed/trading_model_episode_30.h5 model/
COPY simulations.toml .
COPY trader/ trader/

USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"]
CMD ["python", "-m", "trader"]
