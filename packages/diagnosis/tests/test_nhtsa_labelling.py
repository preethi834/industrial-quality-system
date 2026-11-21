"""NHTSA flat-file reader and labelling sheet, on a synthetic flat file (no real data)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.evaluation.nhtsa_labels import (
    COLUMNS,
    SampleConfig,
    check_sheet,
    matches_terms,
    sample_recalls,
    write_sheet,
)
from iqs_diagnosis.ingestion.nhtsa import N_FIELDS, RCL_FIELDS, component_group, read_recalls

from .conftest import TAXONOMY_PATH

TERM = "THE CALIPER BRACKET WAS IMPROPERLY TORQUED DURING ASSEMBLY AT THE SUPPLIER."
PLAIN = "THE BRAKE HOSE MAY RUB AGAINST THE TIRE, CAUSING A LEAK."


def _line(**kw: str) -> str:
    row = dict.fromkeys(RCL_FIELDS, "")
    row.update(
        RECORD_ID="1",
        MAKETXT="ACME",
        MODELTXT="ROADSTER",
        YEARTXT="2018",
        MFGNAME="ACME MOTORS",
        RCLTYPECD="V",
        POTAFF="1200",
        INFLUENCED_BY="MFR",
        RCDATE="20180301",
        CONEQUENCE_DEFECT="INCREASED RISK OF A CRASH.",
        CORRECTIVE_ACTION="DEALERS WILL REPLACE THE PART.",
    )
    row.update(kw)
    return "\t".join(row[f] for f in RCL_FIELDS)


def _flat_file(tmp_path: Path) -> Path:
    lines = []
    for i in range(12):  # brakes: 8 term + 4 plain campaigns
        desc = TERM if i < 8 else PLAIN
        camp = f"18V{i:03d}000"
        comp = "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS"
        lines.append(_line(CAMPNO=camp, COMPNAME=comp, DESC_DEFECT=desc, YEARTXT="2017"))
        lines.append(_line(CAMPNO=camp, COMPNAME=comp, DESC_DEFECT=desc, YEARTXT="2019"))
    for i in range(6):  # steering: 3 term + 3 plain
        desc = TERM.replace("CALIPER BRACKET", "TIE ROD") if i < 3 else PLAIN
        lines.append(_line(CAMPNO=f"19V{i:03d}000", COMPNAME="STEERING:LINKAGES", DESC_DEFECT=desc))
    # Excluded: too old, equipment recall, no defect text.
    lines.append(
        _line(CAMPNO="09V001000", COMPNAME="STEERING", DESC_DEFECT=TERM, RCDATE="20090101")
    )
    lines.append(_line(CAMPNO="18E001000", COMPNAME="STEERING", DESC_DEFECT=TERM, RCLTYPECD="E"))
    lines.append(_line(CAMPNO="18V999000", COMPNAME="STEERING", DESC_DEFECT=""))
    short = "\t".join(_line(CAMPNO="18V500000", COMPNAME="STRUCTURE").split("\t")[:22])
    lines.append(short)
    lines.append(_line(CAMPNO="18V600000") + "\textra")  # 30 fields: malformed
    p = tmp_path / "FLAT_RCL.txt"
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return p


def test_reader_groups_campaigns_and_counts_odd_rows(tmp_path: Path) -> None:
    recalls, stats = read_recalls(_flat_file(tmp_path))
    assert N_FIELDS == 29
    assert stats.malformed == 1 and stats.short_rows == 1
    brake = next(r for r in recalls if r.campno == "18V000000")
    assert brake.n_rows == 2 and brake.model_years == "2017-2019"
    assert brake.component_groups == ("SERVICE BRAKES, HYDRAULIC",)
    assert brake.potaff == 1200 and brake.defect_summary == TERM
    assert len({r.campno for r in recalls}) == len(recalls)
    assert component_group("STEERING:LINKAGES") == "STEERING"


def test_term_filter() -> None:
    assert matches_terms(TERM)
    assert not matches_terms(PLAIN)
    assert matches_terms("the welds were porous")
    assert not matches_terms("the reassembly instructions")  # word start only


def _cfg() -> SampleConfig:
    return SampleConfig(
        groups=("SERVICE BRAKES, HYDRAULIC", "STEERING"),
        per_group_term=5,
        per_group_control=2,
        seed=7,
    )


def test_sample_is_deterministic_stratified_and_filtered(tmp_path: Path) -> None:
    recalls, _ = read_recalls(_flat_file(tmp_path))
    a = sample_recalls(recalls, _cfg())
    assert [(r.campno, g, s) for r, g, s in a] == [
        (r.campno, g, s) for r, g, s in sample_recalls(recalls, _cfg())
    ]
    by = {(g, s): [r for r, gg, ss in a if (gg, ss) == (g, s)] for _, g, s in a}
    assert len(by[("SERVICE BRAKES, HYDRAULIC", "term")]) == 5
    assert len(by[("SERVICE BRAKES, HYDRAULIC", "control")]) == 2
    assert len(by[("STEERING", "term")]) == 3  # only 3 exist
    assert all(matches_terms(r.defect_summary) for r in by[("STEERING", "term")])
    assert not any(matches_terms(r.defect_summary) for r in by[("STEERING", "control")])
    camps = {r.campno for r, _, _ in a}
    assert not camps & {"09V001000", "18E001000", "18V999000"}
    assert len(camps) == len(a)


def _sheet(tmp_path: Path) -> tuple[Path, Path]:
    recalls, stats = read_recalls(_flat_file(tmp_path))
    out, man = tmp_path / "sheet.csv", tmp_path / "sheet.manifest.json"
    write_sheet(
        sample_recalls(recalls, _cfg()),
        out,
        man,
        cfg=_cfg(),
        source={"file": "FLAT_RCL.txt"},
        read_stats=stats,
        n_recalls=len(recalls),
    )
    return out, man


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _save(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def test_sheet_hides_stratum_and_is_never_overwritten(tmp_path: Path) -> None:
    out, man = _sheet(tmp_path)
    rows = _rows(out)
    assert tuple(rows[0]) == COLUMNS and "stratum" not in rows[0]
    assert all(r["label_status"] == "" for r in rows)
    with pytest.raises(FileExistsError):
        _sheet(tmp_path)
    res = check_sheet(out, man, load_taxonomy(TAXONOMY_PATH))
    assert res.errors == [] and len(res.todo) == len(rows) == res.n_rows


def test_label_rules(tmp_path: Path) -> None:
    out, man = _sheet(tmp_path)
    rows = _rows(out)
    term = [r for r in rows if "TORQUED" in r["defect_summary"]]
    plain = [r for r in rows if "TORQUED" not in r["defect_summary"]]
    good = {
        "label_status": "labelled",
        "cause_code": "METHOD.WRONG_PROCESS_PARAMETERS",
        "evidence": "improperly   torqued during assembly",  # case/space-insensitive
        "confidence": "high",
    }
    term[0].update(good)
    term[1].update({**good, "cause_code": "MACHINE.NOT_A_CAUSE"})
    term[2].update({**good, "evidence": "torqued by a robot"})  # not a quote
    term[3].update({**good, "confidence": "low"})
    term[4].update({**good, "secondary_cause_codes": "METHOD.WRONG_PROCESS_PARAMETERS"})
    plain[0].update({"label_status": "outside_taxonomy", "evidence": "may rub against the tire"})
    plain[1].update({"label_status": "not_determinable", "cause_code": "MACHINE.TOOL_WEAR"})
    plain[2].update({"label_status": "", "notes": "", "confidence": "high"})
    plain[3].update({"label_status": "maybe"})
    _save(out, rows)
    res = check_sheet(out, man, load_taxonomy(TAXONOMY_PATH))
    errs = {e.split(":", 1)[0]: e for e in res.errors}
    assert term[0]["row_id"] not in errs and plain[0]["row_id"] not in errs
    for bad in (term[1], term[2], term[3], term[4], plain[1], plain[2], plain[3]):
        assert bad["row_id"] in errs, bad["row_id"]
    assert "verbatim" in errs[term[2]["row_id"]]
    done = {lb.row_id: lb for lb in res.labels}
    assert done[term[0]["row_id"]].stratum == "term"
    assert done[plain[0]["row_id"]].status == "outside_taxonomy"
    s = res.summary()
    assert s["status"] == {"labelled": 1, "outside_taxonomy": 1}


def test_edited_campno_or_header_is_caught(tmp_path: Path) -> None:
    out, man = _sheet(tmp_path)
    rows = _rows(out)
    rows[0]["campno"] = "XXXX"
    _save(out, rows)
    res = check_sheet(out, man, load_taxonomy(TAXONOMY_PATH))
    assert any("does not match the manifest" in e for e in res.errors)
    assert any("missing from the CSV" in e for e in res.errors)
    out.write_text(out.read_text(encoding="utf-8-sig").replace("notes", "comments", 1))
    assert "header" in check_sheet(out, man, load_taxonomy(TAXONOMY_PATH)).errors[0]


def test_sheet_saved_by_excel_with_semicolons_and_cp1252(tmp_path: Path) -> None:
    out, man = _sheet(tmp_path)
    rows = _rows(out)
    rows[0].update(
        {
            "label_status": "not_determinable",
            "notes": "Prüfung: no cause given",  # non-ASCII survives cp1252
        }
    )
    with out.open("w", encoding="cp1252", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    res = check_sheet(out, man, load_taxonomy(TAXONOMY_PATH))
    assert res.errors == [] and len(res.labels) == 1 and len(res.todo) == len(rows) - 1
