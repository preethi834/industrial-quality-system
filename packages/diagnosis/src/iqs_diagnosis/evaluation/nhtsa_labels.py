"""NHTSA recall labelling sheet: build it (sample campaigns) and validate the labels.

One CSV row per recall campaign. The first columns are read-only recall text; the
labeller fills the last six (``LABEL_COLUMNS``). The rules are in
``eval_sets/diagnosis/LABELLING_GUIDE.md`` (section "NHTSA recalls").

The sheet's manifest (JSON next to the CSV) records how the sample was drawn and the
stratum of every row. The stratum is kept **out of the CSV** so the labeller does not
see which rows matched the manufacturing-term filter.

These labels are ground truth for the real-data evaluation, so the module lives in
``evaluation`` (the leak guard keeps the agent path from importing it).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from iqs_common.taxonomy import Taxonomy
from iqs_diagnosis.ingestion.nhtsa import ReadStats, Recall

SHEET_VERSION: Final = 1
TEXT_COLUMNS: Final = (
    "row_id",
    "campno",
    "rcdate",
    "mfgname",
    "component_group",
    "components",
    "makes",
    "model_years",
    "n_rows",
    "potaff",
    "defect_summary",
    "consequence_summary",
    "corrective_summary",
)
LABEL_COLUMNS: Final = (
    "label_status",
    "cause_code",
    "secondary_cause_codes",
    "evidence",
    "confidence",
    "notes",
)
COLUMNS: Final = TEXT_COLUMNS + LABEL_COLUMNS
STATUSES: Final = ("labelled", "outside_taxonomy", "not_determinable")
CONFIDENCES: Final = ("high", "medium")

# Words that suggest the defect summary names a production cause (pre-filter only;
# the label always comes from reading the text). Documented in the guide.
MANUFACTURING_TERMS: Final = (
    r"manufactur\w*",
    r"assembl\w*",
    r"weld\w*",
    r"supplier",
    r"torqu\w*",
    r"contaminat\w*",
    r"improperly",
    r"out of specification|out-of-specification|out of spec\b",
    r"machin\w*",
    r"heat[- ]treat\w*",
    r"porosity",
    r"tolerance",
    r"fixture",
    r"calibrat\w*",
    r"production (?:line|process)",
)
_TERMS_RE: Final = re.compile(r"\b(?:" + "|".join(MANUFACTURING_TERMS) + r")", re.IGNORECASE)

Stratum = Literal["term", "control"]


class SampleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[str, ...]  # component groups, in priority order
    since: str = "20150101"  # RCDATE >= since (YYYYMMDD)
    recall_type: str = "V"  # vehicle recalls only
    per_group_term: int = 60
    per_group_control: int = 15
    seed: int = 20261121


def matches_terms(text: str) -> bool:
    return _TERMS_RE.search(text) is not None


def _group_of(r: Recall, groups: Sequence[str]) -> str | None:
    return next((g for g in groups if g in r.component_groups), None)


def sample_recalls(
    recalls: Sequence[Recall], cfg: SampleConfig
) -> list[tuple[Recall, str, Stratum]]:
    """Per group: ``per_group_term`` campaigns whose defect summary matches a
    manufacturing term and ``per_group_control`` from the rest (estimates how many
    labellable recalls the filter misses). Deterministic for a given input and seed.

    A campaign that touches several chosen groups is assigned to the first in
    ``cfg.groups`` and can only be drawn once.
    """
    rng = np.random.default_rng(cfg.seed)
    eligible = [
        r
        for r in sorted(recalls, key=lambda r: r.campno)
        if r.recall_type == cfg.recall_type and r.rcdate >= cfg.since and r.defect_summary
    ]
    out: list[tuple[Recall, str, Stratum]] = []
    for g in cfg.groups:
        pool = [r for r in eligible if _group_of(r, cfg.groups) == g]
        for stratum, n in (("term", cfg.per_group_term), ("control", cfg.per_group_control)):
            cand = [r for r in pool if matches_terms(r.defect_summary) == (stratum == "term")]
            take = min(n, len(cand))
            idx = sorted(rng.choice(len(cand), size=take, replace=False)) if take else []
            out += [(cand[i], g, stratum) for i in idx]  # type: ignore[misc]
    return out


def _text_row(row_id: str, r: Recall, group: str) -> dict[str, str]:
    return {
        "row_id": row_id,
        "campno": r.campno,
        "rcdate": r.rcdate,
        "mfgname": r.mfgname,
        "component_group": group,
        "components": "; ".join(r.components),
        "makes": "; ".join(r.makes),
        "model_years": r.model_years,
        "n_rows": str(r.n_rows),
        "potaff": "" if r.potaff is None else str(r.potaff),
        "defect_summary": r.defect_summary,
        "consequence_summary": r.consequence_summary,
        "corrective_summary": r.corrective_summary,
    }


def write_sheet(
    sample: Sequence[tuple[Recall, str, Stratum]],
    csv_path: Path,
    manifest_path: Path,
    *,
    cfg: SampleConfig,
    source: dict[str, Any],
    read_stats: ReadStats,
    n_recalls: int,
) -> None:
    """Write the CSV (rows shuffled, stratum hidden) and its manifest. Refuses to
    overwrite an existing sheet: labels are never thrown away by a rebuild."""
    for p in (csv_path, manifest_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; a labelling sheet is never overwritten")
    rng = np.random.default_rng(cfg.seed + 1)
    order = rng.permutation(len(sample))
    rows, strata = [], {}
    for k, i in enumerate(order, 1):
        r, g, stratum = sample[int(i)]
        row_id = f"r{k:04d}"
        rows.append({**_text_row(row_id, r, g), **dict.fromkeys(LABEL_COLUMNS, "")})
        strata[row_id] = {"campno": r.campno, "group": g, "stratum": stratum}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel and LibreOffice open it as UTF-8; every field is quoted.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    counts = Counter((g, s) for _, g, s in sample)
    manifest = {
        "sheet_version": SHEET_VERSION,
        "csv": csv_path.name,
        "source": source,
        "read_stats": read_stats.model_dump(),
        "n_campaigns_in_file": n_recalls,
        "sample": cfg.model_dump(mode="json"),
        "manufacturing_terms": list(MANUFACTURING_TERMS),
        "counts": {f"{g}|{s}": n for (g, s), n in sorted(counts.items())},
        "rows": strata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- labels


class NhtsaLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    row_id: str
    campno: str
    status: Literal["labelled", "outside_taxonomy", "not_determinable"]
    cause_code: str | None
    secondary_cause_codes: tuple[str, ...]
    confidence: str | None
    evidence: str
    stratum: Stratum
    group: str


class SheetCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    labels: list[NhtsaLabel]
    todo: list[str]  # row ids with no status yet
    errors: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.n_rows,
            "done": len(self.labels),
            "todo": len(self.todo),
            "errors": len(self.errors),
            "status": dict(Counter(lb.status for lb in self.labels)),
            "cause": dict(Counter(lb.cause_code for lb in self.labels if lb.cause_code)),
            "labelled_by_stratum": dict(
                Counter(lb.stratum for lb in self.labels if lb.status == "labelled")
            ),
            "done_by_stratum": dict(Counter(lb.stratum for lb in self.labels)),
        }


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def check_sheet(csv_path: Path, manifest_path: Path, taxonomy: Taxonomy) -> SheetCheck:
    """Validate every row; collects all problems instead of stopping at the first."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strata: dict[str, dict[str, str]] = manifest["rows"]
    codes = set(taxonomy.leaf_codes)
    labels: list[NhtsaLabel] = []
    todo: list[str] = []
    errors: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            return SheetCheck(
                n_rows=len(strata),
                labels=[],
                todo=[],
                errors=["header: columns changed or reordered"],
            )
        seen: set[str] = set()
        for row in reader:
            rid = row["row_id"]
            err = []
            meta = strata.get(rid)
            if meta is None or meta["campno"] != row["campno"]:
                errors.append(f"{rid}: row_id/campno does not match the manifest")
                continue
            seen.add(rid)
            status = row["label_status"].strip().lower()
            cause = row["cause_code"].strip()
            secondary = tuple(
                c.strip() for c in row["secondary_cause_codes"].split(";") if c.strip()
            )
            conf = row["confidence"].strip().lower()
            evidence = row["evidence"].strip()
            if not status:
                if cause or secondary or evidence or conf:
                    err.append("label columns filled but label_status is empty")
                else:
                    todo.append(rid)
                    continue
            elif status not in STATUSES:
                err.append(f"label_status {status!r} not one of {', '.join(STATUSES)}")
            else:
                text = _norm(
                    " ".join(
                        row[k]
                        for k in ("defect_summary", "consequence_summary", "corrective_summary")
                    )
                )
                if status == "labelled":
                    if cause not in codes:
                        err.append(f"cause_code {cause!r} is not a taxonomy cause")
                    if conf not in CONFIDENCES:
                        err.append("confidence must be high or medium (low -> not_determinable)")
                    bad = [c for c in secondary if c not in codes or c == cause]
                    if bad:
                        err.append(f"secondary_cause_codes invalid or equal to primary: {bad}")
                else:
                    if cause or secondary:
                        err.append(f"{status}: cause_code and secondary_cause_codes must be empty")
                if status in ("labelled", "outside_taxonomy"):
                    if not evidence:
                        err.append(f"{status}: evidence (a quote from the summaries) is required")
                    elif _norm(evidence) not in text:
                        err.append("evidence is not a verbatim quote from the summaries")
            if err:
                errors += [f"{rid}: {e}" for e in err]
                continue
            labels.append(
                NhtsaLabel(
                    row_id=rid,
                    campno=row["campno"],
                    status=status,
                    cause_code=cause or None,
                    secondary_cause_codes=secondary,
                    confidence=conf or None,
                    evidence=evidence,
                    stratum=meta["stratum"],
                    group=meta["group"],
                )
            )
    missing = sorted(set(strata) - seen)
    if missing:
        errors.append(f"rows missing from the CSV: {', '.join(missing[:10])}")
    return SheetCheck(n_rows=len(strata), labels=labels, todo=todo, errors=errors)
