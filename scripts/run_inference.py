#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import importlib.metadata
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import requests
from output_parser import parse_and_validate
from PIL import Image
from pilot_utils import (
    CONDITIONS,
    MODEL_IDS,
    ROOT,
    relative,
    sha256_file,
    slugify_model_id,
    write_json,
)


def manifest_rows(condition: str) -> List[Dict[str, str]]:
    with (ROOT / "data" / "report_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if condition == "all" else [row for row in rows if row["condition"] == condition]


def output_paths(model_id: str, condition: str, document_id: str) -> Dict[str, Path]:
    model_slug = slugify_model_id(model_id)
    return {
        "raw": ROOT / "results" / "raw" / model_slug / condition / f"{document_id}.txt",
        "normalized": ROOT
        / "results"
        / "normalized"
        / model_slug
        / condition
        / f"{document_id}.json",
        "record": ROOT / "results" / "run_records" / model_slug / condition / f"{document_id}.json",
    }


def build_local_transformers_backend(
    model_id: str,
    revision: Optional[str],
    dtype_name: str,
) -> tuple[Any, Dict[str, Any]]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("The Transformers backend requires `uv sync --extra inference`.") from exc

    token = os.getenv("HF_TOKEN") or None
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        revision=revision,
        token=token,
        torch_dtype=dtype,
        device_map="auto",
    )
    metadata = {
        "resolved_revision": getattr(pipe.model.config, "_commit_hash", None),
        "device": str(pipe.model.device),
        "dtype": str(getattr(pipe.model, "dtype", "unknown")),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
    }
    return pipe, metadata


def local_transformers_call(
    pipe: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    result = pipe(
        text=messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = result[0]["generated_text"]
    if isinstance(generated, list):
        raw_text = generated[-1]["content"]
    else:
        raw_text = str(generated)
    return raw_text


def endpoint_call(
    model_id: str,
    revision: str,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    endpoint_url: str,
) -> tuple[str, Dict[str, Any]]:
    token = os.getenv("MEDGEMMA_ENDPOINT_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    response = requests.post(
        endpoint_url,
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    raw_text = body["choices"][0]["message"]["content"]
    return raw_text, {
        "resolved_revision": revision,
        "endpoint_response_id": body.get("id"),
        "endpoint_model": body.get("model"),
    }


def record_run(
    model_id: str,
    backend: str,
    requested_revision: Optional[str],
    row: Mapping[str, str],
    paths: Mapping[str, Path],
    raw_text: str,
    backend_metadata: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    parsed, parse_metadata = parse_and_validate(raw_text)
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw"].write_text(raw_text, encoding="utf-8")
    if parsed is not None:
        write_json(paths["normalized"], parsed)
    write_json(
        paths["record"],
        {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "backend": backend,
            "requested_revision": requested_revision,
            **dict(backend_metadata),
            "document_id": row["document_id"],
            "semantic_case_id": row["semantic_case_id"],
            "template_id": row["template_id"],
            "condition": row["condition"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "prompt_sha256": sha256_file(ROOT / "prompts" / "extraction_prompt.txt"),
            "schema_sha256": sha256_file(ROOT / "schema" / "extraction.schema.json"),
            "elapsed_seconds": elapsed_seconds,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "raw_output_path": relative(paths["raw"]),
            "normalized_output_path": (
                relative(paths["normalized"]) if parsed is not None else None
            ),
            **parse_metadata,
        },
    )


def run(args: argparse.Namespace) -> None:
    if args.model not in MODEL_IDS:
        raise SystemExit(f"Model must be one of: {', '.join(MODEL_IDS)}")
    rows = manifest_rows(args.condition)
    if args.limit:
        rows = rows[: args.limit]
    prompt = (ROOT / "prompts" / "extraction_prompt.txt").read_text(encoding="utf-8")

    if args.backend == "endpoint":
        endpoint_url = args.endpoint_url or os.getenv("MEDGEMMA_ENDPOINT_URL")
        if not endpoint_url:
            raise SystemExit("Endpoint backend requires --endpoint-url or MEDGEMMA_ENDPOINT_URL.")
        if not args.revision:
            raise SystemExit(
                "Endpoint runs require --revision so the deployed model snapshot is recorded."
            )
    else:
        endpoint_url = None

    local_pipe = None
    local_metadata: Dict[str, Any] = {}
    if args.backend == "transformers":
        local_pipe, local_metadata = build_local_transformers_backend(
            args.model,
            args.revision,
            args.dtype,
        )

    completed = 0
    try:
        for row in rows:
            paths = output_paths(args.model, row["condition"], row["document_id"])
            if paths["raw"].exists() and not args.overwrite:
                print(f"SKIP existing {relative(paths['raw'])}")
                continue
            image_path = ROOT / row["image_path"]
            started = time.perf_counter()
            if args.backend == "transformers":
                raw_text = local_transformers_call(
                    local_pipe,
                    image_path,
                    prompt,
                    args.max_new_tokens,
                )
                backend_metadata = local_metadata
            else:
                raw_text, backend_metadata = endpoint_call(
                    args.model,
                    args.revision,
                    image_path,
                    prompt,
                    args.max_new_tokens,
                    endpoint_url,
                )
            elapsed = time.perf_counter() - started
            record_run(
                args.model,
                args.backend,
                args.revision,
                row,
                paths,
                raw_text,
                backend_metadata,
                elapsed,
            )
            completed += 1
            print(f"WROTE {row['document_id']} {row['condition']} ({elapsed:.1f}s)")
    finally:
        if local_pipe is not None:
            del local_pipe
    print(f"Completed {completed} new inference calls.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen MedGemma extraction prompt over pilot images."
    )
    parser.add_argument("--backend", choices=("transformers", "endpoint"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--endpoint-url")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Transformers backend compute dtype; ignored by endpoint runs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
