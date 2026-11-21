"""Read the NHTSA recalls flat file (``FLAT_RCL_*.txt``) into one record per campaign.

Format (``RCL.txt``, last updated May 2025): tab-delimited, no header, no quoting, dates
``YYYYMMDD``, 29 fields. Older records can have fewer fields (fields 23-29 were added
between 2007 and 2025); missing trailing fields are read as empty. A line with more
than 29 fields is counted as malformed and skipped (never repaired silently).

The flat file has one row per campaign x make x model x year (x component). A labelling
case is one **campaign** (``CAMPNO``): the defect, consequence and corrective summaries
are per campaign, so the rows are grouped and the vehicles are counted, not repeated.

This module reads only; it never downloads. The file is fetched by hand after approval
(see ``eval_sets/diagnosis/LABELLING_GUIDE.md``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

RCL_FIELDS: Final = (
    "RECORD_ID",
    "CAMPNO",
    "MAKETXT",
    "MODELTXT",
    "YEARTXT",
    "MFGCAMPNO",
    "COMPNAME",
    "MFGNAME",
    "BGMAN",
    "ENDMAN",
    "RCLTYPECD",
    "POTAFF",
    "ODATE",
    "INFLUENCED_BY",
    "MFGTXT",
    "RCDATE",
    "DATEA",
    "RPNO",
    "FMVSS",
    "DESC_DEFECT",
    "CONEQUENCE_DEFECT",  # sic, as in RCL.txt
    "CORRECTIVE_ACTION",
    "NOTES",
    "RCL_CMPT_ID",
    "MFR_COMP_NAME",
    "MFR_COMP_DESC",
    "MFR_COMP_PTNO",
    "DO_NOT_DRIVE",
    "PARK_OUTSIDE",
)
N_FIELDS: Final = len(RCL_FIELDS)


class ReadStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lines: int = 0
    rows: int = 0
    short_rows: int = 0  # fewer than 29 fields, padded
    malformed: int = 0  # more than 29 fields, skipped
    decode_replacements: int = 0  # lines with undecodable bytes (replaced)


class Recall(BaseModel):
    """One recall campaign (all flat-file rows with the same ``CAMPNO``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    campno: str
    rcdate: str
    mfgname: str
    recall_type: str
    influenced_by: str
    components: tuple[str, ...]  # distinct COMPNAME values, sorted
    component_groups: tuple[str, ...]  # their top level (before the first ":"), sorted
    makes: tuple[str, ...]
    n_rows: int
    model_years: str  # "2015-2018", or "" if all unknown
    potaff: int | None
    defect_summary: str
    consequence_summary: str
    corrective_summary: str


def component_group(compname: str) -> str:
    """``"SERVICE BRAKES, HYDRAULIC:ANTILOCK"`` -> ``"SERVICE BRAKES, HYDRAULIC"``."""
    return compname.split(":", 1)[0].strip()


def read_rows(path: Path, stats: ReadStats | None = None) -> Iterator[dict[str, str]]:
    """Rows of the flat file as ``{field: value}`` (values stripped)."""
    st = stats if stats is not None else ReadStats()
    with path.open("rb") as fh:
        for raw in fh:
            st.lines += 1
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if "�" in text:
                st.decode_replacements += 1
            if not text.strip():
                continue
            parts = text.split("\t")
            if len(parts) > N_FIELDS:
                st.malformed += 1
                continue
            if len(parts) < N_FIELDS:
                st.short_rows += 1
                parts += [""] * (N_FIELDS - len(parts))
            st.rows += 1
            yield {k: v.strip() for k, v in zip(RCL_FIELDS, parts, strict=True)}


def _years(values: Iterable[str]) -> str:
    ys = sorted({int(v) for v in values if v.isdigit() and v != "9999"})
    if not ys:
        return ""
    return str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]}-{ys[-1]}"


def _longest(values: Iterable[str]) -> str:
    # Summaries are per campaign; if rows disagree, keep the most complete text.
    return max(values, key=len, default="")


def group_campaigns(rows: Iterable[dict[str, str]]) -> list[Recall]:
    """One :class:`Recall` per ``CAMPNO``, sorted by campaign number."""
    by_camp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["CAMPNO"]:
            by_camp[r["CAMPNO"]].append(r)
    out = []
    for camp, rs in sorted(by_camp.items()):
        comps = sorted({r["COMPNAME"] for r in rs if r["COMPNAME"]})
        pot = [int(r["POTAFF"]) for r in rs if r["POTAFF"].isdigit()]
        out.append(
            Recall(
                campno=camp,
                rcdate=min((r["RCDATE"] for r in rs if r["RCDATE"]), default=""),
                mfgname=_longest(r["MFGNAME"] for r in rs),
                recall_type=_longest(r["RCLTYPECD"] for r in rs),
                influenced_by=_longest(r["INFLUENCED_BY"] for r in rs),
                components=tuple(comps),
                component_groups=tuple(sorted({component_group(c) for c in comps})),
                makes=tuple(sorted({r["MAKETXT"] for r in rs if r["MAKETXT"]})),
                n_rows=len(rs),
                model_years=_years(r["YEARTXT"] for r in rs),
                potaff=max(pot) if pot else None,
                defect_summary=_longest(r["DESC_DEFECT"] for r in rs),
                consequence_summary=_longest(r["CONEQUENCE_DEFECT"] for r in rs),
                corrective_summary=_longest(r["CORRECTIVE_ACTION"] for r in rs),
            )
        )
    return out


def read_recalls(path: Path) -> tuple[list[Recall], ReadStats]:
    stats = ReadStats()
    recalls = group_campaigns(read_rows(path, stats))
    return recalls, stats
