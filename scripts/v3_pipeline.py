#!/usr/bin/env python3
"""Evidence-gated deterministic compiler for the MedGemma v3 pilot.

This module deliberately has no access to the corpus ground truth.  A non-null
candidate value is accepted only when the independent audit marks the field
present and its cited, non-history OCR evidence parses to the same canonical
value.  The compiler is therefore a verifier and serializer, not a second
extractor.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from jsonschema import Draft202012Validator
from pilot_utils import ROOT

COMPILER_VERSION = "v3-evidence-compiler-2"

ATOMIC_FIELD_PATHS = (
    "document_id",
    "specimen_site",
    "laterality",
    "diagnosis",
    "breslow_thickness_mm",
    "breslow_qualifier",
    "ulceration",
    "mitotic_rate_per_mm2",
    "mitotic_qualifier",
    "margins.invasive_peripheral",
    "margins.invasive_deep",
    "margins.in_situ_peripheral",
    "margins.in_situ_deep",
    "staging.pT",
    "staging.pN",
    "staging.pM",
    "staging.stage_group",
)

CLINICAL_FIELD_PATHS = ATOMIC_FIELD_PATHS[1:]
MARGIN_KEYS = (
    "invasive_peripheral",
    "invasive_deep",
    "in_situ_peripheral",
    "in_situ_deep",
)
STAGING_KEYS = ("pT", "pN", "pM", "stage_group")

_LINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_DOCUMENT_ID = re.compile(r"\bMEL[\s-]*(\d{3})[\s-]*([AB])\b", re.IGNORECASE)
_NUMBER = r"(\d+(?:[.,]\d+)?)"
_LABEL_VALUE_SEPARATOR = r"(?:\s*:\s*|\s+-\s+|\s+)"
_CURRENT_FIELD_LABEL = re.compile(
    r"^(?:"
    r"procedure|specimen\s+site|diagnosis|breslow\s+thickness|ulceration|"
    r"mitotic\s+rate|(?:invasive\s+melanoma|melanoma\s+in\s+situ)"
    r"\s*(?:-\s*)?(?:peripheral|deep)\s+margin|"
    r"(?:reported\s+)?(?:pT|pN|pM)|stage\s+group"
    r")(?:\s*:|\s+-|\s+)",
    re.IGNORECASE,
)
_SITE_OCR_CONFUSIONS = {
    frozenset(("0", "o")),
    frozenset(("1", "i")),
    frozenset(("1", "l")),
    frozenset(("5", "s")),
    frozenset(("8", "b")),
    frozenset(("g", "q")),
    frozenset(("i", "l")),
}
_MISSING = object()
_INVALID = object()


class V3CompilationError(ValueError):
    """Raised when the evidence-gated result cannot satisfy the target schema."""

    def __init__(self, message: str, audit_log: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit_log = dict(audit_log or {})


def _clean_text(text: str) -> str:
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
        .strip()
    )


def _looks_like_narrative_current_start(text: str) -> bool:
    """Recognize the first current-specimen line after a history block."""

    if "," not in text or not text.rstrip().endswith(":"):
        return False
    lowered = text.casefold()
    procedure_terms = (
        "biopsy",
        "excision",
        "re-excision",
        "wide local",
        "sentinel",
    )
    return any(term in lowered for term in procedure_terms)


def _looks_like_current_heading(text: str) -> bool:
    """Recognize explicit current-specimen/synoptic headings, not prose."""

    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return bool(
        re.fullmatch(
            r"(?:additional\s+)?current\s+specimen"
            r"(?:\s+findings)?(?:\s+synoptic(?:\s+(?:report|summary|data|findings))?)?",
            normalized,
        )
        or re.fullmatch(
            r"(?:current\s+specimen\s+)?(?:melanoma\s+)?synoptic"
            r"(?:\s+(?:report|summary|data|findings))?",
            normalized,
        )
    )


def sectionize_ocr_lines(lines: Sequence[str | Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return OCR lines with deterministic header/current/history/footer tags.

    Existing metadata is preserved. Missing line IDs are assigned in input
    order. Supplied IDs must be unique and syntactically safe because audit
    responses refer to them verbatim.
    """

    output: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    state = "header"

    for index, source in enumerate(lines, start=1):
        if isinstance(source, str):
            item: Dict[str, Any] = {
                "line_id": f"L{index:03d}",
                "text": source,
            }
        elif isinstance(source, Mapping):
            item = deepcopy(dict(source))
            item.setdefault("line_id", f"L{index:03d}")
        else:
            raise TypeError(f"OCR line {index} must be a string or mapping")

        line_id = item.get("line_id")
        text = item.get("text")
        if not isinstance(line_id, str) or not _LINE_ID.fullmatch(line_id):
            raise ValueError(f"Invalid OCR line ID at position {index}: {line_id!r}")
        if line_id in seen_ids:
            raise ValueError(f"Duplicate OCR line ID: {line_id}")
        if not isinstance(text, str):
            raise ValueError(f"OCR line {line_id} has no string text")
        seen_ids.add(line_id)

        cleaned = _clean_text(text)
        lowered = cleaned.casefold()

        if "clinical history" in lowered and "prior report" in lowered:
            state = "history"
            section = "history"
        elif _looks_like_current_heading(cleaned):
            state = "current"
            section = "current"
        elif lowered in {"comment", "comments", "footer"}:
            state = "footer"
            section = "footer"
        elif state == "history" and _looks_like_narrative_current_start(cleaned):
            state = "current"
            section = "current"
        elif state == "header" and (
            _CURRENT_FIELD_LABEL.match(cleaned) or _looks_like_narrative_current_start(cleaned)
        ):
            state = "current"
            section = "current"
        elif state == "header":
            section = "header"
        else:
            section = state

        item["text"] = cleaned
        item["section"] = section
        output.append(item)

    return output


def _value_at_path(value: Mapping[str, Any], path: str) -> Any:
    node: Any = value
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _audit_entry(audit: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    entry = _value_at_path(audit, path)
    return entry if isinstance(entry, Mapping) else None


def _enum_token(value: Any, allowed: Iterable[str]) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return _INVALID
    token = re.sub(r"[\s-]+", "_", value.strip().casefold())
    return token if token in set(allowed) else _INVALID


def _canonical_site(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return _INVALID
    site = re.sub(r"\s+", " ", value.strip().casefold()).strip(" .,:;")
    site = re.sub(r"^(?:left|right|midline)\s+", "", site)
    return site or _INVALID


def _canonical_diagnosis(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return _INVALID
    token = value.strip().casefold()
    if token in {
        "invasive_melanoma",
        "melanoma_in_situ",
        "residual_melanoma_in_situ_no_invasive",
    }:
        return token
    words = re.sub(r"[_\s]+", " ", token)
    if "residual melanoma in situ" in words and "no residual invasive melanoma" in words:
        return "residual_melanoma_in_situ_no_invasive"
    if words.startswith("invasive melanoma"):
        return "invasive_melanoma"
    if words.startswith("melanoma in situ"):
        return "melanoma_in_situ"
    return _INVALID


def _canonical_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) and float(value) >= 0 else _INVALID
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*", value):
        return float(value.strip().replace(",", "."))
    return _INVALID


def _canonical_document_id(value: Any) -> Any:
    if not isinstance(value, str):
        return _INVALID
    match = _DOCUMENT_ID.fullmatch(value.strip())
    return f"MEL-{match.group(1)}-{match.group(2).upper()}" if match else _INVALID


def _canonical_stage(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return _INVALID
    text = re.sub(r"\s+", " ", value.strip().rstrip("."))
    if kind == "stage_group":
        match = re.fullmatch(r"stage\s+([0-9ivx]+[a-d]?)", text, re.IGNORECASE)
        return f"Stage {match.group(1).upper()}" if match else _INVALID

    prefix = {"pT": "pT", "pN": "pN", "pM": "pM"}[kind]
    if not text.casefold().startswith(prefix.casefold()):
        return _INVALID
    suffix = text[len(prefix) :].strip()
    if not suffix:
        return _INVALID
    if suffix.casefold() == "not assigned":
        return f"{prefix} not assigned"
    if kind == "pT":
        match = re.fullmatch(r"(?:(?:is)|(?:\d+[a-d]?))", suffix, re.IGNORECASE)
    else:
        match = re.fullmatch(r"(?:\d+[a-d]?)", suffix, re.IGNORECASE)
    return f"{prefix}{suffix.lower()}" if match else _INVALID


def _canonical_candidate(path: str, value: Any) -> Any:
    if value is _MISSING:
        return _MISSING
    if path == "document_id":
        return _canonical_document_id(value)
    if path == "specimen_site":
        return _canonical_site(value)
    if path == "laterality":
        return _enum_token(value, ("left", "right", "midline"))
    if path == "diagnosis":
        return _canonical_diagnosis(value)
    if path in {"breslow_thickness_mm", "mitotic_rate_per_mm2"}:
        return _canonical_number(value)
    if path == "breslow_qualifier":
        return _enum_token(value, ("exact", "at_least", "approximate"))
    if path == "mitotic_qualifier":
        return _enum_token(value, ("exact", "at_least"))
    if path == "ulceration":
        return _enum_token(value, ("present", "not_identified", "cannot_be_assessed"))
    if path.startswith("margins."):
        return _enum_token(value, ("involved", "not_involved", "cannot_be_assessed"))
    if path.startswith("staging."):
        return _canonical_stage(value, path.split(".", 1)[1])
    raise KeyError(path)


def _number(text: str) -> float:
    return float(text.replace(",", "."))


def _site_phrase(text: str) -> str | None:
    match = re.search(
        rf"\bspecimen\s+site{_LABEL_VALUE_SEPARATOR}(.+)$",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .")
    if _looks_like_narrative_current_start(text):
        return text.split(",", 1)[0].strip()
    return None


def _parse_document_id(text: str) -> List[Any]:
    if "document" not in text.casefold():
        return []
    return [
        f"MEL-{match.group(1)}-{match.group(2).upper()}" for match in _DOCUMENT_ID.finditer(text)
    ]


def _parse_site(text: str) -> List[Any]:
    phrase = _site_phrase(text)
    return [] if phrase is None else [_canonical_site(phrase)]


def _parse_laterality(text: str) -> List[Any]:
    phrase = _site_phrase(text)
    if phrase is None:
        return []
    match = re.match(r"\s*(left|right|midline)\b", phrase, re.IGNORECASE)
    return [match.group(1).casefold()] if match else []


def _parse_diagnosis(text: str) -> List[Any]:
    match = re.search(
        rf"\bdiagnosis{_LABEL_VALUE_SEPARATOR}(.+)$",
        text,
        re.IGNORECASE,
    )
    phrase = match.group(1) if match else text
    lowered = phrase.strip().casefold()
    if not match and not lowered.startswith(
        ("invasive melanoma", "melanoma in situ", "residual melanoma in situ")
    ):
        return []
    parsed = _canonical_diagnosis(phrase)
    return [] if parsed is _INVALID else [parsed]


def _parse_breslow(text: str) -> List[Any]:
    match = re.search(
        rf"\bbreslow\s+thickness(?:\s+is)?{_LABEL_VALUE_SEPARATOR}"
        rf"(?:(approximately|about|at\s+least|greater\s+than)\s+)?{_NUMBER}\s*mm\b",
        text,
        re.IGNORECASE,
    )
    return [] if not match else [_number(match.group(2))]


def _parse_breslow_qualifier(text: str) -> List[Any]:
    if not _parse_breslow(text):
        return []
    lowered = text.casefold()
    if "at least" in lowered or "greater than" in lowered:
        return ["at_least"]
    if "approximately" in lowered or "about" in lowered:
        return ["approximate"]
    return ["exact"]


def _parse_ulceration(text: str) -> List[Any]:
    match = re.search(
        rf"\bulceration{_LABEL_VALUE_SEPARATOR}"
        r"(present|not\s+identified|cannot\s+be\s+assessed)\b",
        text,
        re.IGNORECASE,
    )
    return (
        []
        if not match
        else [
            _enum_token(
                match.group(1),
                (
                    "present",
                    "not_identified",
                    "cannot_be_assessed",
                ),
            )
        ]
    )


def _parse_mitotic(text: str) -> List[Any]:
    match = re.search(
        rf"\bmitotic\s+rate(?:\s+is)?{_LABEL_VALUE_SEPARATOR}"
        rf"(?:(at\s+least|greater\s+than)\s+)?{_NUMBER}\s*"
        r"(?:per\s*|/\s*)mm\s*2\b",
        text,
        re.IGNORECASE,
    )
    return [] if not match else [_number(match.group(2))]


def _parse_mitotic_qualifier(text: str) -> List[Any]:
    if not _parse_mitotic(text):
        return []
    lowered = text.casefold()
    return ["at_least" if "at least" in lowered or "greater than" in lowered else "exact"]


def _parse_margin(text: str, component: str, depth: str) -> List[Any]:
    pattern = re.compile(
        rf"\b{re.escape(component)}\s*(?:-\s*)?{depth}\s+margin"
        rf"{_LABEL_VALUE_SEPARATOR}"
        r"(cannot\s+be\s+assessed|not\s+involved|involved)\b",
        re.IGNORECASE,
    )
    return [
        _enum_token(
            match.group(1),
            ("involved", "not_involved", "cannot_be_assessed"),
        )
        for match in pattern.finditer(text)
    ]


def _parse_stage(text: str, kind: str) -> List[Any]:
    if kind == "stage_group":
        pattern = re.compile(
            rf"\bstage\s+group{_LABEL_VALUE_SEPARATOR}"
            r"(stage\s+[0-9ivx]+[a-d]?)\b",
            re.IGNORECASE,
        )
        return [_canonical_stage(match.group(1), "stage_group") for match in pattern.finditer(text)]

    label = re.escape(kind)
    pattern = re.compile(
        rf"\b{label}{_LABEL_VALUE_SEPARATOR}"
        rf"({label}\s+not\s+assigned|{label}(?:is|\d+[a-d]?))\b",
        re.IGNORECASE,
    )
    return [_canonical_stage(match.group(1), kind) for match in pattern.finditer(text)]


FieldParser = Callable[[str], List[Any]]

_FIELD_PARSERS: Dict[str, FieldParser] = {
    "document_id": _parse_document_id,
    "specimen_site": _parse_site,
    "laterality": _parse_laterality,
    "diagnosis": _parse_diagnosis,
    "breslow_thickness_mm": _parse_breslow,
    "breslow_qualifier": _parse_breslow_qualifier,
    "ulceration": _parse_ulceration,
    "mitotic_rate_per_mm2": _parse_mitotic,
    "mitotic_qualifier": _parse_mitotic_qualifier,
    "margins.invasive_peripheral": lambda text: _parse_margin(
        text, "invasive melanoma", "peripheral"
    ),
    "margins.invasive_deep": lambda text: _parse_margin(text, "invasive melanoma", "deep"),
    "margins.in_situ_peripheral": lambda text: _parse_margin(
        text, "melanoma in situ", "peripheral"
    ),
    "margins.in_situ_deep": lambda text: _parse_margin(text, "melanoma in situ", "deep"),
    "staging.pT": lambda text: _parse_stage(text, "pT"),
    "staging.pN": lambda text: _parse_stage(text, "pN"),
    "staging.pM": lambda text: _parse_stage(text, "pM"),
    "staging.stage_group": lambda text: _parse_stage(text, "stage_group"),
}


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
        )
    return type(left) is type(right) and left == right


def _unique_values(values: Iterable[Any]) -> List[Any]:
    unique: List[Any] = []
    for value in values:
        if not any(_values_equal(value, existing) for existing in unique):
            unique.append(value)
    return unique


def _bounded_site_ocr_substitution(
    candidate_site: Any, evidence_site: Any
) -> Dict[str, Any] | None:
    """Allow one aligned, known OCR-glyph substitution in one site token.

    This is deliberately not edit distance or dictionary correction: token
    insertions/deletions, reordered tokens, multiple substitutions, and
    non-whitelisted character pairs all fail closed.
    """

    if not isinstance(candidate_site, str) or not isinstance(evidence_site, str):
        return None
    candidate_tokens = re.findall(r"[a-z0-9]+", candidate_site.casefold())
    evidence_tokens = re.findall(r"[a-z0-9]+", evidence_site.casefold())
    if len(candidate_tokens) != len(evidence_tokens):
        return None

    differing_tokens = [
        index
        for index, (candidate_token, evidence_token) in enumerate(
            zip(candidate_tokens, evidence_tokens)
        )
        if candidate_token != evidence_token
    ]
    if len(differing_tokens) != 1:
        return None

    token_index = differing_tokens[0]
    candidate_token = candidate_tokens[token_index]
    evidence_token = evidence_tokens[token_index]
    if len(candidate_token) != len(evidence_token) or len(candidate_token) < 3:
        return None

    differing_characters = [
        index
        for index, (candidate_character, evidence_character) in enumerate(
            zip(candidate_token, evidence_token)
        )
        if candidate_character != evidence_character
    ]
    if len(differing_characters) != 1:
        return None

    character_index = differing_characters[0]
    candidate_character = candidate_token[character_index]
    evidence_character = evidence_token[character_index]
    if frozenset((candidate_character, evidence_character)) not in _SITE_OCR_CONFUSIONS:
        return None

    return {
        "type": "single_confusable_glyph_substitution",
        "token_index": token_index,
        "character_index": character_index,
        "candidate_token": candidate_token,
        "ocr_token": evidence_token,
        "candidate_character": candidate_character,
        "ocr_character": evidence_character,
    }


def _blank_result() -> Dict[str, Any]:
    return {
        "document_id": None,
        "specimen_site": None,
        "laterality": None,
        "diagnosis": None,
        "breslow_thickness_mm": None,
        "breslow_qualifier": None,
        "ulceration": None,
        "mitotic_rate_per_mm2": None,
        "mitotic_qualifier": None,
        "margins": {key: None for key in MARGIN_KEYS},
        "staging": {key: None for key in STAGING_KEYS},
    }


def _set_path(value: Dict[str, Any], path: str, child: Any) -> None:
    parts = path.split(".")
    if len(parts) == 1:
        value[parts[0]] = child
    else:
        value[parts[0]][parts[1]] = child


def _field_decision(
    path: str,
    candidate: Mapping[str, Any],
    audit: Mapping[str, Any],
    line_index: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Dict[str, Any]]:
    raw_candidate = _value_at_path(candidate, path)
    canonical = _canonical_candidate(path, raw_candidate)
    entry = _audit_entry(audit, path)
    status = entry.get("status") if entry else None
    evidence_ids = entry.get("evidence_line_ids") if entry else None

    log: Dict[str, Any] = {
        "candidate_value": None if raw_candidate is _MISSING else raw_candidate,
        "candidate_omitted": raw_candidate is _MISSING,
        "canonical_candidate": None if canonical in {_MISSING, _INVALID} else canonical,
        "audit_status": status,
        "evidence_line_ids": evidence_ids if isinstance(evidence_ids, list) else [],
        "evidence_sections": [],
        "parsed_evidence_values": [],
        "normalization": None,
        "accepted": False,
        "reason": "",
        "final_value": None,
    }

    if raw_candidate is _MISSING or raw_candidate is None:
        log["reason"] = "candidate_null_or_omitted"
        return None, log
    if canonical is _INVALID:
        log["reason"] = "candidate_not_canonicalizable"
        return None, log
    if canonical is None:
        log["reason"] = "candidate_null_or_omitted"
        return None, log
    if entry is None:
        log["reason"] = "missing_audit_entry"
        return None, log
    if status != "present":
        log["reason"] = f"audit_status_{status or 'missing'}"
        return None, log
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or len(evidence_ids) > 2
        or any(not isinstance(line_id, str) for line_id in evidence_ids)
    ):
        log["reason"] = "invalid_evidence_line_ids"
        return None, log
    if len(set(evidence_ids)) != len(evidence_ids):
        log["reason"] = "duplicate_evidence_line_id"
        return None, log
    if any(line_id not in line_index for line_id in evidence_ids):
        log["reason"] = "unknown_evidence_line_id"
        return None, log

    evidence = [line_index[line_id] for line_id in evidence_ids]
    sections = [line["section"] for line in evidence]
    log["evidence_sections"] = sections
    if "history" in sections:
        log["reason"] = "history_evidence_rejected"
        return None, log
    if path == "document_id":
        allowed_sections = {"header"}
    else:
        allowed_sections = {"current"}
    if any(section not in allowed_sections for section in sections):
        log["reason"] = "wrong_evidence_section"
        return None, log

    parsed_values: List[Any] = []
    parser = _FIELD_PARSERS[path]
    for line in evidence:
        parsed_values.extend(parser(str(line["text"])))
    parsed_values = [value for value in parsed_values if value is not _INVALID]
    log["parsed_evidence_values"] = parsed_values
    unique = _unique_values(parsed_values)
    if len(unique) != 1:
        log["reason"] = "evidence_missing_or_ambiguous"
        return None, log
    if not _values_equal(canonical, unique[0]):
        if path != "specimen_site":
            log["reason"] = "candidate_evidence_disagreement"
            return None, log
        normalization = _bounded_site_ocr_substitution(canonical, unique[0])
        if normalization is None:
            log["reason"] = "candidate_evidence_disagreement"
            return None, log
        log["normalization"] = normalization

    log.update(
        {
            "accepted": True,
            "reason": (
                "accepted_bounded_site_ocr_substitution"
                if log["normalization"] is not None
                else "accepted"
            ),
            "final_value": canonical,
        }
    )
    return canonical, log


def _recover_document_id_from_header(
    final: Dict[str, Any],
    fields_log: Dict[str, Dict[str, Any]],
    candidate: Mapping[str, Any],
    line_index: Mapping[str, Mapping[str, Any]],
) -> None:
    """Recover auditor-omitted identity only with exact labeled header evidence."""

    if final["document_id"] is not None:
        return
    field_log = fields_log["document_id"]
    if field_log["audit_status"] not in {"absent", "unclear"}:
        return
    raw_candidate = _value_at_path(candidate, "document_id")
    canonical = _canonical_candidate("document_id", raw_candidate)
    if canonical in {_MISSING, _INVALID, None}:
        return

    parsed_occurrences: List[tuple[str, Any]] = []
    for line_id, line in line_index.items():
        if line["section"] != "header":
            continue
        for parsed in _parse_document_id(str(line["text"])):
            if parsed is not _INVALID:
                parsed_occurrences.append((line_id, parsed))

    if len(parsed_occurrences) != 1:
        return
    line_id, parsed = parsed_occurrences[0]
    if not _values_equal(canonical, parsed):
        return

    final["document_id"] = canonical
    field_log.update(
        {
            "evidence_line_ids": [line_id],
            "evidence_sections": ["header"],
            "parsed_evidence_values": [parsed],
            "normalization": {
                "type": "labeled_header_identity_fallback",
                "source_field": "document_id",
            },
            "accepted": True,
            "reason": "accepted_labeled_header_identity_fallback",
            "final_value": canonical,
        }
    )


def _recover_laterality_from_site_evidence(
    final: Dict[str, Any],
    fields_log: Dict[str, Dict[str, Any]],
    candidate: Mapping[str, Any],
    line_index: Mapping[str, Mapping[str, Any]],
) -> None:
    """Recover auditor-omitted laterality from accepted specimen-site evidence."""

    if final["laterality"] is not None or final["specimen_site"] is None:
        return
    field_log = fields_log["laterality"]
    site_log = fields_log["specimen_site"]
    if field_log["audit_status"] not in {"absent", "unclear"} or not site_log["accepted"]:
        return

    raw_candidate = _value_at_path(candidate, "laterality")
    canonical = _canonical_candidate("laterality", raw_candidate)
    if canonical in {_MISSING, _INVALID, None}:
        return

    evidence_ids = site_log["evidence_line_ids"]
    if not evidence_ids:
        return
    parsed_values: List[Any] = []
    for line_id in evidence_ids:
        line = line_index.get(line_id)
        if line is None or line["section"] != "current":
            return
        parsed_values.extend(_parse_laterality(str(line["text"])))
    parsed_values = [value for value in parsed_values if value is not _INVALID]
    unique = _unique_values(parsed_values)
    if len(unique) != 1 or not _values_equal(canonical, unique[0]):
        return

    final["laterality"] = canonical
    field_log.update(
        {
            "evidence_line_ids": list(evidence_ids),
            "evidence_sections": ["current"] * len(evidence_ids),
            "parsed_evidence_values": parsed_values,
            "normalization": {
                "type": "linked_field_evidence",
                "source_field": "specimen_site",
            },
            "accepted": True,
            "reason": "accepted_from_specimen_site_evidence",
            "final_value": canonical,
        }
    )


def _derive_missing_qualifier_from_measurement(
    final: Dict[str, Any],
    fields_log: Dict[str, Dict[str, Any]],
    candidate: Mapping[str, Any],
    line_index: Mapping[str, Mapping[str, Any]],
    value_path: str,
    qualifier_path: str,
) -> None:
    """Derive only a missing qualifier from an already accepted measurement."""

    if (
        _value_at_path(final, value_path) is None
        or _value_at_path(final, qualifier_path) is not None
    ):
        return
    raw_qualifier = _value_at_path(candidate, qualifier_path)
    if raw_qualifier is not _MISSING and raw_qualifier is not None:
        return

    value_log = fields_log[value_path]
    qualifier_log = fields_log[qualifier_path]
    if not value_log["accepted"]:
        return
    evidence_ids = value_log["evidence_line_ids"]
    if not evidence_ids:
        return

    parsed_values: List[Any] = []
    parser = _FIELD_PARSERS[qualifier_path]
    for line_id in evidence_ids:
        line = line_index.get(line_id)
        if line is None or line["section"] != "current":
            return
        parsed_values.extend(parser(str(line["text"])))
    parsed_values = [value for value in parsed_values if value is not _INVALID]
    unique = _unique_values(parsed_values)
    if len(unique) != 1:
        return

    derived = unique[0]
    _set_path(final, qualifier_path, derived)
    qualifier_log.update(
        {
            "evidence_line_ids": list(evidence_ids),
            "evidence_sections": ["current"] * len(evidence_ids),
            "parsed_evidence_values": parsed_values,
            "normalization": {
                "type": "linked_measurement_qualifier",
                "source_field": value_path,
            },
            "accepted": True,
            "reason": "derived_from_accepted_measurement_evidence",
            "final_value": derived,
        }
    )


def _force_qualifier_coherence(
    final: Dict[str, Any],
    fields_log: Dict[str, Dict[str, Any]],
    value_path: str,
    qualifier_path: str,
) -> None:
    value = _value_at_path(final, value_path)
    qualifier = _value_at_path(final, qualifier_path)
    if value is None and qualifier is not None:
        _set_path(final, qualifier_path, None)
        fields_log[qualifier_path].update(
            {
                "accepted": False,
                "reason": "coherence_value_is_null",
                "final_value": None,
            }
        )
    elif value is not None and qualifier is None:
        _set_path(final, value_path, None)
        fields_log[value_path].update(
            {
                "accepted": False,
                "reason": "coherence_qualifier_is_null",
                "final_value": None,
            }
        )


def compile_prediction(
    candidate: Mapping[str, Any],
    audit: Mapping[str, Any],
    ocr_lines: Sequence[str | Mapping[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile a candidate into the unchanged target schema.

    Ground truth is never consulted. The returned audit log records every
    acceptance and rejection, including history evidence attempts.
    """

    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be a mapping")
    if not isinstance(audit, Mapping):
        raise TypeError("audit must be a mapping")

    sectioned = sectionize_ocr_lines(ocr_lines)
    line_index = {str(line["line_id"]): line for line in sectioned}
    final = _blank_result()
    fields_log: Dict[str, Dict[str, Any]] = {}

    for path in ATOMIC_FIELD_PATHS:
        compiled, field_log = _field_decision(path, candidate, audit, line_index)
        _set_path(final, path, compiled)
        fields_log[path] = field_log

    _recover_document_id_from_header(final, fields_log, candidate, line_index)
    _recover_laterality_from_site_evidence(final, fields_log, candidate, line_index)
    _derive_missing_qualifier_from_measurement(
        final,
        fields_log,
        candidate,
        line_index,
        "breslow_thickness_mm",
        "breslow_qualifier",
    )
    _derive_missing_qualifier_from_measurement(
        final,
        fields_log,
        candidate,
        line_index,
        "mitotic_rate_per_mm2",
        "mitotic_qualifier",
    )

    _force_qualifier_coherence(
        final,
        fields_log,
        "breslow_thickness_mm",
        "breslow_qualifier",
    )
    _force_qualifier_coherence(
        final,
        fields_log,
        "mitotic_rate_per_mm2",
        "mitotic_qualifier",
    )

    schema_path = ROOT / "schema" / "extraction.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(final),
        key=lambda error: [str(part) for part in error.path],
    )
    schema_errors = [
        {
            "path": ".".join(str(part) for part in error.path),
            "message": error.message,
        }
        for error in errors
    ]
    audit_log: Dict[str, Any] = {
        "compiler_version": COMPILER_VERSION,
        "fields": fields_log,
        "accepted_non_null_count": sum(
            bool(item["accepted"]) and item["final_value"] is not None
            for item in fields_log.values()
        ),
        "rejected_non_null_count": sum(
            item["candidate_value"] is not None
            and not item["candidate_omitted"]
            and not item["accepted"]
            for item in fields_log.values()
        ),
        "history_evidence_rejection_count": sum(
            item["reason"] == "history_evidence_rejected" for item in fields_log.values()
        ),
        "schema_valid": not errors,
        "schema_errors": schema_errors,
    }
    if errors:
        raise V3CompilationError(
            "Evidence-gated prediction does not satisfy the unchanged target schema",
            audit_log,
        )
    return final, audit_log
