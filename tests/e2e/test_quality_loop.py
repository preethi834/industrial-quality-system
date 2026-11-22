"""Step 7 gate 1: the quality loop on the compose stack (plumbing, NOT detection quality).

Data: SIMULATOR process data (tool_wear_m1, one seed per repeat), SYNTHETIC image-to-part
linkage, synthetic stripe images, random-init tiny PatchCore. Nothing here says anything
about detection on real images (that is the 7h demo and step 8c).

Per repeat (own category, own run, own detector), as blueprint gate 1 / notes §7.3:

 1-3. world prepared by conftest (simulate, link, fit, library);
 4.   stream images in part order to ``POST /v1/discovery/inspect`` until the alert
      (from 100 parts before the fault onset; earlier parts are good training images and
      route ``normal`` by construction, see configs/simulator/image_linkage.yaml);
 5.   poll ``GET /v1/cases?source=vision`` -> exactly one case, awaiting confirmation;
      30 more parts stream in meanwhile -> the cluster grows, no second alert;
 6.   ``/cause-set`` is a conformal set with ``guarantee_applies: false``;
 7.   the same ``new_defect_cluster`` event is set back to ``pending`` -> still one case;
 8.   ``POST /confirm`` with the true cause (oracle engineer: the harness reads ground truth);
 9.   Neo4j edge, count n -> n+1, candidate-pool row, calibration file unchanged, cluster
      ``named``;
 10.  more scratch images -> ``known:scratch``, still one case, no new alert.

Measured values go to ``reports/quality_loop/e2e_<utc>.json``.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text

from .conftest import CALIBRATION, REPO_ROOT, WORK, Stack, WorldRun

pytestmark = pytest.mark.e2e

PRE_ONSET_PARTS = 100
GROW_PARTS = 30
KNOWN_AFTER_NAMING = 20
POLL_S = 0.05


def _wait(what: str, fn: Any, timeout_s: float = 180.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        out = fn()
        if out:
            return out
        time.sleep(POLL_S)
    raise TimeoutError(f"timed out waiting for {what}")


def _inspect(stack: Stack, run: WorldRun, row: Any) -> dict[str, Any]:
    data = (run.pool / row.image).read_bytes()
    r = stack.vision.post(
        "/v1/discovery/inspect",
        files={"file": (Path(row.image).name, data, "image/png")},
        data={
            "category": run.category,
            "part_id": row.part_id,
            "run_id": run.run_id,
            "line_id": "L1",
            "inspected_at": pd.Timestamp(row.inspected_ts).isoformat(),
        },
    )
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _vision_cases(stack: Stack, cluster_id: str | None = None) -> list[dict[str, Any]]:
    params = {"source": "vision"} | ({"cluster_id": cluster_id} if cluster_id else {})
    r = stack.diagnosis.get("/v1/cases", params=params)
    assert r.status_code == 200, r.text
    out: list[dict[str, Any]] = r.json()
    return out


def _count(stack: Stack, context: str, cause: str) -> int:
    rows = stack.diagnosis.get("/v1/feedback/counts").json()["counts"]
    return sum(c["count"] for c in rows if (c["context"], c["cause_code"]) == (context, cause))


def _calibration_sha_in_container(stack: Stack) -> str:
    out = stack.compose("exec", "-T", "diagnosis", "sha256sum", f"/app/{CALIBRATION}").stdout
    return out.split()[0]


def _one_repeat(stack: Stack, run: WorldRun, engine: Any, neo: Any) -> dict[str, Any]:
    m = run.manifest.sort_values("seq").reset_index(drop=True)
    onset_idx = int((m["inspected_ts"] < run.onset).sum())
    start = max(0, onset_idx - PRE_ONSET_PARTS)
    rows = list(m.iloc[start:].itertuples())
    seen: list[tuple[Any, dict[str, Any]]] = []
    first_anomalous_t: float | None = None
    alert: dict[str, Any] | None = None
    i = 0

    # 4. stream until the alert
    while alert is None:
        assert i < len(rows), "manifest exhausted without an alert"
        row = rows[i]
        i += 1
        res = _inspect(stack, run, row)
        seen.append((row, res))
        if res["decision"] == "anomalous" and first_anomalous_t is None:
            first_anomalous_t = time.perf_counter()
        assert not res["routing"].startswith("known"), "known class before any confirmation"
        if res.get("alert_event_id"):
            alert = res
    t_alert = time.perf_counter()
    assert first_anomalous_t is not None
    cluster_id = alert["cluster_id"]
    alert_row = seen[-1][0]
    n_after_onset = sum(1 for r, _ in seen if r.inspected_ts >= run.onset)
    n_anomalous = sum(1 for _, res in seen if res["decision"] == "anomalous")
    cluster0 = stack.vision.get(f"/v1/clusters/{cluster_id}").json()

    # 5. exactly one case, run up to the confirmation
    cases = _wait("case opened", lambda: _vision_cases(stack, cluster_id))
    t_opened = time.perf_counter()
    assert len(cases) == 1
    case_id = cases[0]["case_id"]
    _wait(
        "awaiting_confirmation",
        lambda: _vision_cases(stack, cluster_id)[0]["phase"] == "awaiting_confirmation",
    )
    t_waiting = time.perf_counter()
    for row in rows[i : i + GROW_PARTS]:  # the line keeps producing while the case waits
        res = _inspect(stack, run, row)
        seen.append((row, res))
        assert not res.get("alert_event_id"), "second alert for a growing cluster"
        assert not res["routing"].startswith("known")
    i += GROW_PARTS
    case = stack.diagnosis.get(f"/v1/cases/{case_id}").json()
    assert case["linkage"] == "synthetic_simulator" and "SYNTHETIC" in case["description"]

    # 6. cause set: conformal, but no guarantee for a vision-triggered case
    cs = stack.diagnosis.get(f"/v1/cases/{case_id}/cause-set").json()
    assert cs["kind"] == "conformal" and cs["guarantee_applies"] is False
    assert stack.diagnosis.get(f"/v1/cases/{case_id}/hypotheses").status_code == 200
    assert stack.diagnosis.get(f"/v1/cases/{case_id}/report-8d").status_code == 200

    # 7. redelivery of the same event inside the stack -> still one case, agent not re-run
    event_id = alert["alert_event_id"]
    with engine.begin() as c:
        c.execute(
            text(
                "UPDATE platform.events SET status = 'pending', processed_at = NULL WHERE id = :id"
            ),
            {"id": event_id},
        )
    with engine.connect() as c:
        _wait(
            "redelivery processed",
            lambda: (
                c.execute(
                    text("SELECT status FROM platform.events WHERE id = :id"), {"id": event_id}
                ).scalar()
                == "processed"
            ),
            60,
        )
        c.commit()
    assert [x["case_id"] for x in _vision_cases(stack)].count(case_id) == 1
    assert len(_vision_cases(stack, cluster_id)) == 1
    assert _vision_cases(stack, cluster_id)[0]["phase"] == "awaiting_confirmation"
    with engine.connect() as c:
        n_tickets = c.execute(
            text("SELECT count(*) FROM qms.tickets WHERE case_id = :id"), {"id": case_id}
        ).scalar()
        n_seen = c.execute(
            text("SELECT count(*) FROM diagnosis.processed_events WHERE event_id = :id"),
            {"id": event_id},
        ).scalar()
    assert (n_tickets, n_seen) == (1, 1)

    # 8. the (oracle) engineer confirms the injected cause
    context = f"defect_type:{run.category}/scratch"
    before = _count(stack, context, run.true_cause)
    r = stack.diagnosis.post(
        f"/v1/cases/{case_id}/confirm",
        json={
            "cause_code": run.true_cause,
            "engineer": "e2e-oracle-engineer",
            "defect_class_name": "scratch",
        },
    )
    assert r.status_code == 200, r.text
    fb = r.json()["feedback"]
    assert fb["graph"]["status"] == "written", fb

    # 9. what the system learned
    assert _count(stack, context, run.true_cause) == before + 1
    cands = stack.diagnosis.get("/v1/feedback/candidates").json()["candidates"]
    assert [c["case_id"] for c in cands].count(case_id) == 1
    assert all(c["used_in_calibration_run"] is None for c in cands)
    recs, _, _ = neo.execute_query(
        "MATCH (:Case {id: $id})-[r:CONFIRMED_CAUSE]->(:Cause {code: $code}) RETURN count(r) AS n",
        id=case_id,
        code=run.true_cause,
    )
    assert recs[0]["n"] == 1
    recs, _, _ = neo.execute_query(
        "MATCH (:DefectType {key: $key})-[i:INDICATES]->(:Cause {code: $code}) RETURN i.count AS n",
        key=f"{run.category}/scratch",
        code=run.true_cause,
    )
    assert [x["n"] for x in recs] == [before + 1]
    named = _wait(
        "cluster named",
        lambda: (
            (c := stack.vision.get(f"/v1/clusters/{cluster_id}").json())["status"] == "named" and c
        ),
        60,
    )
    t_named = time.perf_counter()
    assert named["defect_class_name"] == "scratch" and named["cause_code"] == run.true_cause

    # 10. later scratches are a known class; no new alert, no new case
    known_after: list[str] = []
    while len(known_after) < KNOWN_AFTER_NAMING and i < len(rows):
        row = rows[i]
        i += 1
        res = _inspect(stack, run, row)
        assert not res.get("alert_event_id"), "alert after the class was named"
        if row.shown_as == "defect":
            known_after.append(res["routing"])
        else:
            assert res["routing"] == "normal"
    assert len(known_after) >= KNOWN_AFTER_NAMING, "too few defect parts left in the run"
    frac_known = known_after.count("known:scratch") / len(known_after)
    assert frac_known >= 0.9
    assert len(_vision_cases(stack, cluster_id)) == 1
    with engine.connect() as c:
        n_alerts = c.execute(
            text(
                "SELECT count(*) FROM platform.events WHERE type = 'new_defect_cluster' "
                "AND payload -> 'payload' ->> 'category' = :cat"
            ),
            {"cat": run.category},
        ).scalar()
    assert n_alerts == 1

    return {
        "repeat": run.repeat,
        "category": run.category,
        "simulator_run_id": run.run_id,
        "scenario_seed": run.seed,
        "true_cause": run.true_cause,
        "parts_streamed_from": "onset-100",
        "images_after_onset_to_alert": n_after_onset,
        "anomalous_images_to_alert": n_anomalous,
        "cluster_size_at_alert": cluster0["size"],
        "alert_part_seq": int(alert_row.seq),
        "false_known_before_confirmation": 0,
        "cases_opened": 1,
        "latency_first_anomalous_to_case_opened_s": round(t_opened - first_anomalous_t, 3),
        "latency_first_anomalous_to_awaiting_confirmation_s": round(
            t_waiting - first_anomalous_t, 3
        ),
        "latency_alert_to_case_opened_s": round(t_opened - t_alert, 3),
        "latency_alert_to_awaiting_confirmation_s": round(t_waiting - t_alert, 3),
        "conformal_set": cs.get("set") or cs.get("causes"),
        "top3": cs.get("top3"),
        "novel": cs.get("novel"),
        "count_before_after": [before, before + 1],
        "awaiting_confirmation_to_cluster_named_s": round(t_named - t_waiting, 3),
        "known_fraction_after_naming": round(frac_known, 3),
        "n_defects_after_naming": len(known_after),
    }


def test_quality_loop_on_compose_stack(stack: Stack) -> None:
    """Gate 1 (plumbing on simulator data + synthetic linkage; not a detection result)."""
    repo_sha = hashlib.sha256((REPO_ROOT / CALIBRATION).read_bytes()).hexdigest()
    sha_before = _calibration_sha_in_container(stack)
    assert sha_before == repo_sha
    engine = create_engine(stack.db_url)
    results = []
    t0 = time.perf_counter()
    with GraphDatabase.driver(stack.bolt_uri, auth=stack.neo4j_auth) as neo:
        for run in stack.runs:
            results.append(_one_repeat(stack, run, engine, neo))
    total_s = time.perf_counter() - t0

    # Across repeats: one alert and one confirmation each; nothing failed or pending.
    with engine.connect() as c:
        by: dict[str, int] = dict(
            c.execute(
                text("SELECT type || ':' || status, count(*) FROM platform.events GROUP BY 1")
            )
            .tuples()
            .all()
        )
        n_cases = c.execute(text("SELECT count(*) FROM diagnosis.cases")).scalar()
        cal_rows = c.execute(text("SELECT count(*) FROM diagnosis.calibration_runs")).scalar()
    n = len(stack.runs)
    assert by == {"new_defect_cluster:processed": n, "case_confirmed:processed": n}, by
    assert n_cases == n
    sha_after = _calibration_sha_in_container(stack)
    assert sha_after == sha_before  # 5. the active calibration run is byte-identical
    engine.dispose()

    lat = [r["latency_first_anomalous_to_case_opened_s"] for r in results]
    lat_wait = [r["latency_first_anomalous_to_awaiting_confirmation_s"] for r in results]
    report = {
        "what": "step-7 gate 1 e2e on the compose stack (iqs-e2e project)",
        "data": (
            "SIMULATOR process data (tool_wear_m1), SYNTHETIC image-to-part linkage, "
            "synthetic stripe images (iqs_vision.testing), random-init tiny PatchCore; "
            "plumbing only, no detection-quality claim"
        ),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": "dev laptop (WSL2, 16 vCPU, ~9.7 GiB VM)",
        "repeats": n,
        "calibration_sha256": sha_after,
        "calibration_rows": cal_rows,
        "median_latency_first_anomalous_to_case_opened_s": statistics.median(lat),
        "median_latency_first_anomalous_to_awaiting_confirmation_s": statistics.median(lat_wait),
        "loop_total_s": round(total_s, 1),
        "stack": json.loads((WORK / "stack_info.json").read_text()),
        "runs": results,
    }
    out = REPO_ROOT / "reports/quality_loop"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / f"e2e_{stamp}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("\n[step-7 e2e]", json.dumps({k: v for k, v in report.items() if k != "runs"}))
