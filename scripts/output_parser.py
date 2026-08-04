from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft202012Validator
from pilot_utils import ROOT, read_json

OUTER_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def deterministic_unfence(raw_text: str) -> str:
    stripped = raw_text.strip()
    match = OUTER_FENCE.fullmatch(stripped)
    return match.group(1).strip() if match else stripped


def parse_and_validate(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidate = deterministic_unfence(raw_text)
    metadata: Dict[str, Any] = {
        "outer_fence_removed": candidate != raw_text.strip(),
        "parse_valid": False,
        "schema_valid": False,
        "parse_error": None,
        "schema_errors": [],
    }
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        metadata["parse_error"] = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        return None, metadata
    if not isinstance(parsed, dict):
        metadata["parse_error"] = "Top-level JSON value is not an object"
        return None, metadata

    metadata["parse_valid"] = True
    schema = read_json(ROOT / "schema" / "extraction.schema.json")
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda error: list(error.path))
    metadata["schema_errors"] = [
        {
            "path": ".".join(str(part) for part in error.path),
            "message": error.message,
        }
        for error in errors
    ]
    metadata["schema_valid"] = not errors
    return parsed, metadata


def parse_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    return parse_and_validate(path.read_text(encoding="utf-8"))
