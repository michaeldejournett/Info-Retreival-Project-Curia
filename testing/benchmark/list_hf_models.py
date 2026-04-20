from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List

try:
    import requests
except Exception:  # pragma: no cover - dependency/environment specific
    requests = None

from testing.benchmark.env_loader import load_project_env_once


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Hugging Face hosted models available to the current token via router API."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of models to print (default: 50). Use 0 for no limit.",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default="",
        help="Case-insensitive substring filter for model IDs.",
    )
    return parser.parse_args()


def _collect_model_ids(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())

    return model_ids


def main() -> int:
    args = _parse_args()

    if requests is None:
        print("requests package is not installed", file=sys.stderr)
        return 1

    load_project_env_once()
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("missing HUGGINGFACE_API_TOKEN/HF_TOKEN", file=sys.stderr)
        return 1

    headers = {"Authorization": f"Bearer {token}"}
    url = "https://router.huggingface.co/v1/models"

    try:
        response = requests.get(url, headers=headers, timeout=30)
    except Exception as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    payload: Any
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.status_code >= 400:
        message = ""
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, str):
                message = error_obj
            elif isinstance(error_obj, dict):
                nested = error_obj.get("message")
                if isinstance(nested, str):
                    message = nested
        if not message:
            message = response.text.strip()[:200]
        print(f"http {response.status_code}: {message}", file=sys.stderr)
        return 1

    model_ids = _collect_model_ids(payload)
    if args.contains:
        needle = args.contains.lower()
        model_ids = [model_id for model_id in model_ids if needle in model_id.lower()]

    model_ids = sorted(set(model_ids))
    if args.limit > 0:
        model_ids = model_ids[: args.limit]

    for model_id in model_ids:
        print(model_id)

    print(f"\n{len(model_ids)} model(s) shown", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
