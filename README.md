# MLOps Batch Job

This repository contains a minimal, deterministic batch job that loads YAML config, reads OHLCV data, computes a rolling mean on `close`, generates a binary signal, and writes metrics plus logs.

## Local Run

```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container includes `data.csv` and `config.yaml`, writes `metrics.json` and `run.log`, and prints the final metrics JSON to stdout.

## Publish Docker via GitHub (GHCR)

A GitHub Actions workflow is included at `.github/workflows/docker-publish.yml`.

It publishes images to:

```text
ghcr.io/param20h/primetrade
```

Published tags:
- `latest` on pushes to `main`
- `sha-<short-commit>` on pushes to `main`
- release tag names on git tags like `v1.0.0`

To pull and run the published image:

```bash
docker pull ghcr.io/param20h/primetrade:latest
docker run --rm ghcr.io/param20h/primetrade:latest
```

## Example Metrics

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 39,
  "seed": 42,
  "status": "success"
}
```
