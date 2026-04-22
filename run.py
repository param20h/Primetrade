from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml


class JobError(Exception):
    """Raised when the batch job cannot complete successfully."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLOps batch job")
    parser.add_argument("--input", default="data.csv", help="Path to input CSV file")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--output", default="metrics.json", help="Path to metrics JSON output")
    parser.add_argument("--log-file", default="run.log", help="Path to log file")
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_config(config_path: Path) -> dict:
    if not config_path.exists() or not config_path.is_file():
        raise JobError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise JobError(f"Invalid YAML config: {exc}") from exc
    except OSError as exc:
        raise JobError(f"Could not read config file: {exc}") from exc

    if not isinstance(config, dict):
        raise JobError("Invalid config structure: expected a mapping")

    required_fields = ("seed", "window", "version")
    for field in required_fields:
        if field not in config:
            raise JobError(f"Missing required config field: {field}")

    seed = config["seed"]
    window = config["window"]
    version = config["version"]

    if not isinstance(seed, int):
        raise JobError("Invalid config structure: seed must be an integer")
    if not isinstance(window, int) or window <= 0:
        raise JobError("Invalid config structure: window must be a positive integer")
    if not isinstance(version, str) or not version.strip():
        raise JobError("Invalid config structure: version must be a non-empty string")

    return {"seed": seed, "window": window, "version": version}


def load_close_series(input_path: Path) -> List[float]:
    if not input_path.exists() or not input_path.is_file():
        raise JobError(f"Input file not found: {input_path}")
    if input_path.stat().st_size == 0:
        raise JobError("Input file is empty")

    try:
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            raw_text = handle.read()

        def parse_dict_reader(text: str) -> Tuple[Optional[List[str]], List[dict]]:
            stream = io.StringIO(text)
            reader = csv.DictReader(stream)
            rows = list(reader)
            return reader.fieldnames, rows

        fieldnames, rows = parse_dict_reader(raw_text)
        if fieldnames is None:
            raise JobError("Invalid CSV format: missing header row")
        if "close" not in fieldnames:
            normalized_lines = []
            for raw_line in raw_text.splitlines():
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith('"') and stripped_line.endswith('"'):
                    stripped_line = stripped_line[1:-1]
                normalized_lines.append(stripped_line)

            normalized_text = "\n".join(normalized_lines)
            fieldnames, rows = parse_dict_reader(normalized_text)
            if fieldnames is None or "close" not in fieldnames:
                raise JobError("Missing required column: close")

        closes: List[float] = []
        for row_number, row in enumerate(rows, start=2):
            if row is None:
                raise JobError(f"Invalid CSV format near row {row_number}")
            value = row.get("close", "")
            if value is None or str(value).strip() == "":
                raise JobError(f"Missing close value at row {row_number}")
            try:
                closes.append(float(value))
            except ValueError as exc:
                raise JobError(f"Invalid close value at row {row_number}: {value}") from exc
    except csv.Error as exc:
        raise JobError(f"Invalid CSV format: {exc}") from exc
    except OSError as exc:
        raise JobError(f"Could not read input file: {exc}") from exc

    if not closes:
        raise JobError("Input file is empty")

    return closes


def compute_signals(closes: List[float], window: int) -> Tuple[List[float], List[int]]:
    rolling_means: List[float] = []
    signals: List[int] = []

    for index, close_value in enumerate(closes):
        if index + 1 < window:
            rolling_means.append(float("nan"))
            signals.append(0)
            continue

        window_slice = closes[index + 1 - window : index + 1]
        rolling_mean = float(np.mean(window_slice))
        rolling_means.append(rolling_mean)
        signals.append(1 if close_value > rolling_mean else 0)

    return rolling_means, signals


def write_metrics(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_success_metrics(version: str, seed: int, rows_processed: int, signal_rate: float, latency_ms: int) -> dict:
    return {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }


def build_error_metrics(version: str, message: str) -> dict:
    return {
        "version": version,
        "status": "error",
        "error_message": message,
    }


def run_job(input_path: Path, config_path: Path, output_path: Path, log_file: Path) -> int:
    logger = setup_logger(log_file)
    logger.info("Job start timestamp: %s", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    start_perf_counter = time.perf_counter()

    version = "unknown"
    try:
        config = load_config(config_path)
        version = config["version"]
        seed = config["seed"]
        window = config["window"]
        np.random.seed(seed)
        logger.info("Config loaded + validated: seed=%s window=%s version=%s", seed, window, version)

        closes = load_close_series(input_path)
        logger.info("Rows loaded: %s", len(closes))

        logger.info("Processing step: rolling mean")
        _, signals = compute_signals(closes, window)
        logger.info("Processing step: signal generation")

        signal_rate = float(np.mean(signals))
        latency_ms = int(round((time.perf_counter() - start_perf_counter) * 1000))
        metrics = build_success_metrics(version, seed, len(closes), signal_rate, latency_ms)
        write_metrics(output_path, metrics)
        logger.info("Metrics summary: %s", json.dumps(metrics, sort_keys=True))
        logger.info("Job end + status: success")
        print(json.dumps(metrics, indent=2))
        return 0
    except JobError as exc:
        logger.exception("Validation or processing error: %s", exc)
        metrics = build_error_metrics(version, str(exc))
        write_metrics(output_path, metrics)
        logger.info("Job end + status: error")
        print(json.dumps(metrics, indent=2))
        return 1
    except Exception as exc:
        logger.exception("Unexpected job failure: %s", exc)
        metrics = build_error_metrics(version, f"Unexpected error: {exc}")
        write_metrics(output_path, metrics)
        logger.info("Job end + status: error")
        print(json.dumps(metrics, indent=2))
        return 1


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    input_path = Path(args.input)
    config_path = Path(args.config)
    output_path = Path(args.output)
    log_file = Path(args.log_file)

    if not input_path.is_absolute():
        input_path = base_dir / input_path
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    if not log_file.is_absolute():
        log_file = base_dir / log_file

    return run_job(input_path, config_path, output_path, log_file)


if __name__ == "__main__":
    sys.exit(main())
