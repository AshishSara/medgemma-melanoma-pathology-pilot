#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import importlib.metadata
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import requests
from output_parser import parse_and_validate
from PIL import Image
from pilot_utils import (
    CONDITIONS,
    DEVELOPMENT_CASE_IDS,
    FORMAL_DTYPE,
    FORMAL_MAX_NEW_TOKENS,
    FORMAL_RESPONSE_MODE,
    JSON_ASSISTANT_PREFIX,
    MODEL_IDS,
    MODEL_REVISIONS,
    PROTOCOL_VERSION,
    ROOT,
    read_json,
    relative,
    sha256_bytes,
    sha256_file,
    slugify_model_id,
    write_json,
)


def manifest_rows(condition: str, scope: str) -> List[Dict[str, str]]:
    with (ROOT / "data" / "report_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if scope == "formal":
        rows = [row for row in rows if row["semantic_case_id"] not in DEVELOPMENT_CASE_IDS]
    else:
        rows = [row for row in rows if row["semantic_case_id"] in DEVELOPMENT_CASE_IDS]
    return rows if condition == "all" else [row for row in rows if row["condition"] == condition]


def output_paths(
    model_id: str,
    condition: str,
    document_id: str,
    scope: str = "formal",
) -> Dict[str, Path]:
    model_slug = slugify_model_id(model_id)
    base = ROOT / "results" if scope == "formal" else ROOT / "results" / "development"
    return {
        "raw": base / "raw" / model_slug / condition / f"{document_id}.txt",
        "continuation": base
        / "generated_continuations"
        / model_slug
        / condition
        / f"{document_id}.txt",
        "normalized": base / "normalized" / model_slug / condition / f"{document_id}.json",
        "record": base / "run_records" / model_slug / condition / f"{document_id}.json",
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
        dtype=dtype,
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


def extract_assistant_text(generated: Any) -> str:
    content = generated
    if isinstance(content, list):
        if not content:
            raise ValueError("Transformers returned an empty generated_text list")
        last = content[-1]
        content = last.get("content") if isinstance(last, Mapping) else last
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [
            item.get("text", "")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if fragments:
            return "".join(fragments)
    raise ValueError(f"Unsupported generated assistant content: {type(content).__name__}")


def local_transformers_call(
    pipe: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    response_mode: str,
) -> tuple[str, str]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    call_kwargs: Dict[str, Any] = {}
    if response_mode == "json-prefill":
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": JSON_ASSISTANT_PREFIX}],
            }
        )
        call_kwargs["continue_final_message"] = True
    try:
        result = pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **call_kwargs,
        )
    finally:
        image.close()
    generated = result[0]["generated_text"]
    assistant_text = extract_assistant_text(generated)
    if response_mode == "json-prefill":
        if not assistant_text.startswith(JSON_ASSISTANT_PREFIX):
            raise RuntimeError("Transformers did not preserve the declared JSON assistant prefix")
        continuation = assistant_text[len(JSON_ASSISTANT_PREFIX) :]
    else:
        continuation = assistant_text
    return assistant_text, continuation


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
    continuation_text: str,
    backend_metadata: Mapping[str, Any],
    elapsed_seconds: float,
    response_mode: str,
    max_new_tokens: int,
    requested_dtype: Optional[str],
    scope: str,
    repository_commit: str,
) -> None:
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw"].write_text(raw_text, encoding="utf-8")
    paths["continuation"].parent.mkdir(parents=True, exist_ok=True)
    paths["continuation"].write_text(continuation_text, encoding="utf-8")
    parsed, parse_metadata = parse_and_validate(raw_text)
    if parsed is not None:
        write_json(paths["normalized"], parsed)
    elif paths["normalized"].exists():
        paths["normalized"].unlink()
    prefix = JSON_ASSISTANT_PREFIX if response_mode == "json-prefill" else ""
    raw_bytes = raw_text.encode("utf-8")
    continuation_bytes = continuation_text.encode("utf-8")
    write_json(
        paths["record"],
        {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": PROTOCOL_VERSION,
            "scope": scope,
            "repository_commit": repository_commit,
            "model_id": model_id,
            "backend": backend,
            "requested_revision": requested_revision,
            **dict(backend_metadata),
            "response_mode": response_mode,
            "assistant_prefix": prefix,
            "assistant_prefix_sha256": sha256_bytes(prefix.encode("utf-8")),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "requested_dtype": requested_dtype,
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
            "raw_output_bytes": len(raw_bytes),
            "raw_output_sha256": sha256_bytes(raw_bytes),
            "generated_continuation_path": relative(paths["continuation"]),
            "generated_continuation_bytes": len(continuation_bytes),
            "generated_continuation_sha256": sha256_bytes(continuation_bytes),
            "normalized_output_path": (
                relative(paths["normalized"]) if parsed is not None else None
            ),
            **parse_metadata,
        },
    )


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def verify_existing_output(
    args: argparse.Namespace,
    row: Mapping[str, str],
    paths: Mapping[str, Path],
    revision: str,
) -> None:
    if not paths["record"].exists() or not paths["continuation"].exists():
        raise SystemExit(
            f"Existing raw output lacks its audit record/continuation: {relative(paths['raw'])}"
        )
    record = read_json(paths["record"])
    prompt_sha = sha256_file(ROOT / "prompts" / "extraction_prompt.txt")
    schema_sha = sha256_file(ROOT / "schema" / "extraction.schema.json")
    raw_sha = sha256_file(paths["raw"])
    continuation_sha = sha256_file(paths["continuation"])
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": args.scope,
        "model_id": args.model,
        "backend": args.backend,
        "requested_revision": revision,
        "resolved_revision": revision,
        "response_mode": args.response_mode,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "requested_dtype": args.dtype if args.backend == "transformers" else None,
        "document_id": row["document_id"],
        "semantic_case_id": row["semantic_case_id"],
        "template_id": row["template_id"],
        "condition": row["condition"],
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
        "raw_output_sha256": raw_sha,
        "generated_continuation_sha256": continuation_sha,
    }
    mismatches = [
        f"{key}: expected {value!r}, found {record.get(key)!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if mismatches:
        detail = "\n".join(f"- {item}" for item in mismatches)
        raise SystemExit(
            f"Existing output provenance does not match this run:\n{detail}\n"
            "Use --overwrite only after preserving the earlier attempt."
        )


def run(args: argparse.Namespace) -> None:
    if args.model not in MODEL_IDS:
        raise SystemExit(f"Model must be one of: {', '.join(MODEL_IDS)}")
    pinned_revision = MODEL_REVISIONS[args.model]
    revision = args.revision or pinned_revision
    if revision != pinned_revision:
        raise SystemExit(
            f"{PROTOCOL_VERSION} requires pinned revision {pinned_revision} for {args.model}."
        )
    if args.backend == "endpoint" and args.response_mode != "direct":
        raise SystemExit(
            "The endpoint backend has no verified equivalent of Transformers JSON prefill; "
            "use --response-mode direct only for separately labeled exploratory runs."
        )

    rows = manifest_rows(args.condition, args.scope)
    if args.limit:
        rows = rows[: args.limit]
    prompt = (ROOT / "prompts" / "extraction_prompt.txt").read_text(encoding="utf-8")
    commit = repository_commit()

    if args.backend == "endpoint":
        endpoint_url = args.endpoint_url or os.getenv("MEDGEMMA_ENDPOINT_URL")
        if not endpoint_url:
            raise SystemExit("Endpoint backend requires --endpoint-url or MEDGEMMA_ENDPOINT_URL.")
    else:
        endpoint_url = None

    local_pipe = None
    local_metadata: Dict[str, Any] = {}
    if args.backend == "transformers":
        local_pipe, local_metadata = build_local_transformers_backend(
            args.model,
            revision,
            args.dtype,
        )

    completed = 0
    try:
        for row in rows:
            paths = output_paths(
                args.model,
                row["condition"],
                row["document_id"],
                scope=args.scope,
            )
            image_path = ROOT / row["image_path"]
            actual_image_sha = sha256_file(image_path)
            if actual_image_sha != row["image_sha256"]:
                raise SystemExit(
                    f"Image hash mismatch for {row['document_id']} {row['condition']}: "
                    f"manifest {row['image_sha256']}, actual {actual_image_sha}"
                )
            if paths["raw"].exists() and not args.overwrite:
                verify_existing_output(args, row, paths, revision)
                print(f"SKIP existing {relative(paths['raw'])}")
                continue
            started = time.perf_counter()
            if args.backend == "transformers":
                raw_text, continuation_text = local_transformers_call(
                    local_pipe,
                    image_path,
                    prompt,
                    args.max_new_tokens,
                    args.response_mode,
                )
                backend_metadata = local_metadata
            else:
                raw_text, backend_metadata = endpoint_call(
                    args.model,
                    revision,
                    image_path,
                    prompt,
                    args.max_new_tokens,
                    endpoint_url,
                )
                continuation_text = raw_text
            elapsed = time.perf_counter() - started
            record_run(
                args.model,
                args.backend,
                revision,
                row,
                paths,
                raw_text,
                continuation_text,
                backend_metadata,
                elapsed,
                args.response_mode,
                args.max_new_tokens,
                args.dtype if args.backend == "transformers" else None,
                args.scope,
                commit,
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
    parser.add_argument(
        "--scope",
        choices=("formal", "development"),
        default="formal",
        help="Formal excludes MEL-001; development routes MEL-001 outputs outside scored results.",
    )
    parser.add_argument(
        "--response-mode",
        choices=("direct", "json-prefill"),
        default=FORMAL_RESPONSE_MODE,
    )
    parser.add_argument("--endpoint-url")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default=FORMAL_DTYPE,
        help="Transformers backend compute dtype; ignored by endpoint runs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=FORMAL_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
