"""Step 7 quality-loop demo on REAL images (gate 3) and the false-cluster measurement (#46).

    uv run --frozen python scripts/demo_loop.py --phase demo            # tool wear -> scratch
    uv run --frozen python scripts/demo_loop.py --phase false-clusters  # good parts only

Data labels (printed and written into every report):
- images: REAL MVTec AD ``metal_nut`` test images (read-only, ``data/raw/mvtec_ad``);
- detector: the step-1 deploy PatchCore (``models/patchcore/metal_nut/deploy``, read-only),
  with its step-2 conformal threshold (``calibration.json``);
- process data: SIMULATED (``iqs_sim``), linked to images SYNTHETICALLY (``iqs_sim.linker``,
  ``configs/simulator/image_linkage.yaml``): real images, synthetic process linkage.

Each phase starts its own compose stack (project ``iqs-demo``: base + loop files, throwaway
credentials, own ports and volumes), drives it only through the HTTP APIs (plus one SQL read
of ``platform.events`` for the summary), writes ``reports/quality_loop/demo_<phase>_<utc>.json``
and removes the stack. The script plays the world and, with ``--oracle``, the engineer: it
reads ground truth; the services never can (only ``observed/`` is mounted).

``data/`` and ``models/`` are only read. Models are bind-mounted read-only into the vision
container; the world (runs, linkage, library) lives in ``.demo-work/`` (git-ignored).

Needs ``make build`` first and ``make down`` (one stack at a time on the 10 GB VM).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / ".demo-work"
PROJECT = "iqs-demo"
COMPOSE_FILES = ("deploy/compose/docker-compose.yml", "deploy/compose/docker-compose.loop.yml")
CATEGORY = "metal_nut"
POOL = REPO / "data/raw/mvtec_ad"
MODELS = REPO / "models"
SCENARIOS = REPO / "configs/simulator/scenarios"
NOVEL_SCENARIOS = ("drive_fault_m1", "handling_error_shift_b")
PORTS = {"pg": 25433, "bolt": 27687, "http": 27474, "vision": 28001, "diagnosis": 28002}
LABEL = (
    "REAL MVTec AD metal_nut images, step-1 deploy PatchCore with step-2 conformal threshold; "
    "SIMULATED process data; SYNTHETIC image-to-part linkage"
)


def log(msg: str) -> None:
    print(f"[demo {datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ world


def build_library(path: Path, root: Path) -> int:
    from iqs_common.taxonomy import load_taxonomy
    from iqs_diagnosis.evaluation.labels import load_case_label
    from iqs_diagnosis.ingestion.simulator import load_case
    from iqs_diagnosis.retrieval.library import LibraryRecord, write_library_file
    from iqs_diagnosis.symptoms import extract_profile
    from iqs_diagnosis.symptoms.vocabulary import load_vocabulary
    from iqs_sim.pipeline import run_scenario
    from iqs_sim.scenarios import load_scenario

    taxonomy = load_taxonomy(REPO / "configs/taxonomy/causes.yaml")
    vocab = load_vocabulary(REPO / "configs/diagnosis/symptoms.yaml")
    records = []
    for sc in sorted(SCENARIOS.glob("*.yaml")):
        if sc.stem in NOVEL_SCENARIOS:
            continue
        run = root / sc.stem
        run_scenario(load_scenario(sc), run, taxonomy)
        case, label = load_case(run), load_case_label(run)
        if case is not None and label is not None:
            records.append(
                LibraryRecord(
                    profile=extract_profile(case, vocab),
                    cause_code=label.cause_code,
                    source="simulator",
                    confirmed_by="simulator_ground_truth",
                )
            )
    write_library_file(path, records)
    return len(records)


def build_world(scenario_name: str, seed: int, run_id: str) -> dict[str, Any]:
    from iqs_common.taxonomy import load_taxonomy
    from iqs_sim.linker import LinkageConfig, read_manifest, write_linkage
    from iqs_sim.pipeline import run_scenario
    from iqs_sim.scenarios import load_scenario

    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "loop").mkdir(parents=True)
    n_lib = build_library(WORK / "loop/library/library.jsonl", WORK / "library_runs")
    taxonomy = load_taxonomy(REPO / "configs/taxonomy/causes.yaml")
    scenario = load_scenario(SCENARIOS / f"{scenario_name}.yaml").with_overrides(seed=seed)
    world = WORK / "world" / run_id
    run_scenario(scenario, world, taxonomy)
    cfg = LinkageConfig.load(REPO / "configs/simulator/image_linkage.yaml")
    info = write_linkage(world, POOL, CATEGORY, cfg)  # writes into the world dir only
    shutil.copytree(world / "observed", WORK / "loop/runs" / run_id / "observed")
    for p in [WORK, *WORK.joinpath("loop").rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    faults_path = world / "ground_truth" / "faults.parquet"
    faults = pd.read_parquet(faults_path) if faults_path.exists() else pd.DataFrame()
    return {
        "world": world,
        "manifest": read_manifest(world).sort_values("seq").reset_index(drop=True),
        "true_cause": None if faults.empty else str(faults["cause_code"].iloc[0]),
        "onset": None if faults.empty else pd.Timestamp(faults["onset_ts"].iloc[0]),
        "library_cases": n_lib,
        "linkage_info": {k: v for k, v in info.items() if isinstance(v, str | int | float)},
    }


# ------------------------------------------------------------------ stack


class Stack:
    def __init__(self) -> None:
        self.api_key = secrets.token_hex(16)
        pg_pw, self.neo_pw = secrets.token_hex(12), secrets.token_hex(12)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("IQS_", "COMPOSE_"))}
        env.update(
            IQS_DATABASE__NAME="iqs_demo",
            IQS_DATABASE__USER="iqs_demo",
            IQS_DATABASE__PASSWORD=pg_pw,
            IQS_DATABASE__PORT=str(PORTS["pg"]),
            IQS_NEO4J__USER="neo4j",
            IQS_NEO4J__PASSWORD=self.neo_pw,
            IQS_API__KEYS=self.api_key,
            IQS_LOOP_DIR=str(WORK / "loop"),
            # models/patchcore and models/weights are symlinks into the main checkout; a
            # bind mount does not follow symlinks inside it, so mount their real parent
            # (read-only, see docker-compose.loop.yml).
            IQS_LOOP_MODELS_DIR=str((MODELS / "patchcore").resolve().parent),
            IQS_NEO4J_BOLT_HOST_PORT=str(PORTS["bolt"]),
            IQS_NEO4J_HTTP_HOST_PORT=str(PORTS["http"]),
            IQS_VISION_HOST_PORT=str(PORTS["vision"]),
            IQS_DIAGNOSIS_HOST_PORT=str(PORTS["diagnosis"]),
            COMPOSE_PROFILES="apps",
        )
        self.env = env
        self.db_url = f"postgresql+psycopg://iqs_demo:{pg_pw}@127.0.0.1:{PORTS['pg']}/iqs_demo"
        h = {"X-API-Key": self.api_key}
        self.vision = httpx.Client(
            base_url=f"http://127.0.0.1:{PORTS['vision']}", headers=h, timeout=120
        )
        self.diagnosis = httpx.Client(
            base_url=f"http://127.0.0.1:{PORTS['diagnosis']}", headers=h, timeout=300
        )

    def compose(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["docker", "compose", "--project-directory", str(REPO), "-p", PROJECT]
        for f in COMPOSE_FILES:
            cmd += ["-f", str(REPO / f)]
        return subprocess.run(
            [*cmd, *args], env=self.env, cwd=REPO, check=check, capture_output=True, text=True
        )

    def up(self) -> float:
        t0 = time.perf_counter()
        self.compose("down", "-v", "--remove-orphans", check=False)
        self.compose("up", "-d", "--wait", "postgres", "neo4j")
        menv = {k: v for k, v in self.env.items() if not k.startswith("IQS_")}
        menv.update(
            {k: self.env[k] for k in self.env if k.startswith("IQS_DATABASE__")},
            IQS_DATABASE__HOST="127.0.0.1",
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "alembic",
                "-c",
                "infrastructure/database/alembic.ini",
                "upgrade",
                "head",
            ],
            env=menv,
            cwd=REPO,
            check=True,
            capture_output=True,
        )
        self.compose("up", "-d", "--wait", "vision", "diagnosis")
        self.compose("up", "-d", "--wait", "vision-worker", "diagnosis-worker")
        for c in (self.vision, self.diagnosis):
            wait(f"{c.base_url} ready", lambda c=c: _ok(c, "/ready"), 300, 1.0)
        return time.perf_counter() - t0

    def down(self, logs_to: Path) -> None:
        logs_to.write_text(self.compose("logs", "--no-color", check=False).stdout)
        self.compose("down", "-v", "--remove-orphans", check=False)
        self.vision.close()
        self.diagnosis.close()

    def inspect(self, row: Any, run_id: str) -> tuple[dict[str, Any], float]:
        data = (POOL / row.image).read_bytes()
        t0 = time.perf_counter()
        r = self.vision.post(
            "/v1/discovery/inspect",
            files={"file": (Path(row.image).name, data, "image/png")},
            data={
                "category": CATEGORY,
                "part_id": row.part_id,
                "run_id": run_id,
                "line_id": "L1",
                "inspected_at": pd.Timestamp(row.inspected_ts).isoformat(),
            },
        )
        dt = time.perf_counter() - t0
        r.raise_for_status()
        out: dict[str, Any] = r.json()
        return out, dt

    def vision_cases(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self.diagnosis.get(
            "/v1/cases", params={"source": "vision"}
        ).json()
        return out

    def events(self) -> dict[str, int]:
        engine = create_engine(self.db_url)
        try:
            with engine.connect() as c:
                rows = c.execute(
                    text("SELECT type || ':' || status, count(*) FROM platform.events GROUP BY 1")
                ).all()
        finally:
            engine.dispose()
        return {str(k): int(v) for k, v in rows}


def _ok(c: httpx.Client, path: str) -> bool:
    try:
        return c.get(path).status_code == 200
    except httpx.HTTPError:
        return False


def wait(what: str, fn: Any, timeout_s: float = 300.0, poll_s: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        out = fn()
        if out:
            return out
        time.sleep(poll_s)
    raise TimeoutError(f"timed out waiting for {what}")


def model_info(stack: Stack) -> dict[str, Any]:
    models = stack.vision.get("/v1/models").json()
    return next((m for m in models if m.get("category") == CATEGORY), {})


# ------------------------------------------------------------------ phases


def phase_demo(stack: Stack, w: dict[str, Any], run_id: str, args: argparse.Namespace) -> dict:
    m: pd.DataFrame = w["manifest"]
    onset = w["onset"]
    assert onset is not None, "scenario has no fault"
    onset_idx = int((m["inspected_ts"] < onset).sum())
    rows = list(m.iloc[max(0, onset_idx - args.pre_onset) :].itertuples())
    seen: list[dict[str, Any]] = []
    latencies: list[float] = []
    first_anom: float | None = None
    alert: dict[str, Any] | None = None
    i = 0
    log(f"streaming from part {rows[0].seq} ({args.pre_onset} before onset at part {onset_idx})")
    while alert is None and i < len(rows):
        row = rows[i]
        i += 1
        res, dt = stack.inspect(row, run_id)
        latencies.append(dt)
        seen.append({"seq": int(row.seq), "image": row.image, "shown_as": row.shown_as, **res})
        if res["decision"] == "anomalous" and first_anom is None:
            first_anom = time.perf_counter()
        if res.get("alert_event_id"):
            alert = res
    out: dict[str, Any] = {"images_streamed_to_alert": len(seen)}
    if alert is None:
        out["result"] = "no alert in the whole run"
        return out
    t_alert = time.perf_counter()
    cluster_id = alert["cluster_id"]
    alert_seq = seen[-1]["seq"]
    before_alert = [s for s in seen]
    log(f"ALERT at part {alert_seq}: cluster {cluster_id}")

    def case_for_cluster() -> list[dict[str, Any]]:
        return [c for c in stack.vision_cases() if c["source_cluster_id"] == cluster_id]

    cases = wait("case opened", case_for_cluster, 300)
    t_open = time.perf_counter()
    case_id = cases[0]["case_id"]
    wait(
        "awaiting_confirmation",
        lambda: case_for_cluster()[0]["phase"] == "awaiting_confirmation",
        600,
    )
    t_wait = time.perf_counter()
    cs = stack.diagnosis.get(f"/v1/cases/{case_id}/cause-set").json()
    hyp = stack.diagnosis.get(f"/v1/cases/{case_id}/hypotheses").json()
    log(f"case {case_id}: top3={cs.get('top3')} set={cs.get('causes')} novel={cs.get('novel')}")

    cause = w["true_cause"] if args.oracle else args.confirm_cause
    before = _count(stack, f"defect_type:{CATEGORY}/scratch", cause)
    r = stack.diagnosis.post(
        f"/v1/cases/{case_id}/confirm",
        json={
            "cause_code": cause,
            "engineer": "oracle-engineer (ground truth)" if args.oracle else "demo-engineer",
            "defect_class_name": "scratch",
        },
    )
    r.raise_for_status()
    feedback = r.json()["feedback"]
    after = _count(stack, f"defect_type:{CATEGORY}/scratch", cause)
    named = wait(
        "cluster named",
        lambda: (
            (c := stack.vision.get(f"/v1/clusters/{cluster_id}").json())["status"] == "named" and c
        ),
        120,
    )
    log(
        f"confirmed {cause}; count {before} -> {after}; "
        f"cluster named '{named['defect_class_name']}'"
    )

    after_rows: list[dict[str, Any]] = []
    n_def = 0
    while n_def < args.after_naming and i < len(rows):
        row = rows[i]
        i += 1
        res, dt = stack.inspect(row, run_id)
        latencies.append(dt)
        after_rows.append(
            {"seq": int(row.seq), "image": row.image, "shown_as": row.shown_as, **res}
        )
        n_def += row.shown_as == "defect"
    defects_after = [s for s in after_rows if s["shown_as"] == "defect"]
    seen_images = {s["image"] for s in before_alert}
    out.update(
        {
            "result": "loop closed",
            "true_cause": w["true_cause"],
            "confirmed_cause": cause,
            "confirmed_by": "oracle (ground truth)" if args.oracle else "argument",
            "alert_part_seq": alert_seq,
            "images_after_onset_to_alert": sum(1 for s in before_alert if s["seq"] >= onset_idx),
            "anomalous_to_alert": sum(1 for s in before_alert if s["decision"] == "anomalous"),
            "anomalous_good_images_to_alert": sum(
                1 for s in before_alert if s["decision"] == "anomalous" and s["shown_as"] == "good"
            ),
            "defect_images_to_alert": sum(1 for s in before_alert if s["shown_as"] == "defect"),
            "defect_images_missed_to_alert": sum(
                1
                for s in before_alert
                if s["shown_as"] == "defect" and s["decision"] != "anomalous"
            ),
            "cluster_size_at_alert": alert
            and stack.vision.get(f"/v1/clusters/{cluster_id}").json().get("size"),
            "latency_first_anomalous_to_case_opened_s": round(t_open - (first_anom or t_alert), 2),
            "latency_first_anomalous_to_awaiting_confirmation_s": round(
                t_wait - (first_anom or t_alert), 2
            ),
            "latency_alert_to_awaiting_confirmation_s": round(t_wait - t_alert, 2),
            "cause_set": cs,
            "hypotheses_verdicts": [
                (h.get("cause_code"), h.get("verdict")) for h in hyp.get("hypotheses", [])
            ],
            "feedback": {k: feedback.get(k) for k in ("count_before", "count_after", "graph")},
            "count_before_after": [before, after],
            "after_naming": {
                "parts": len(after_rows),
                "defect_parts": len(defects_after),
                "routing_of_defects": dict(Counter(s["routing"] for s in defects_after)),
                "routing_of_goods": dict(
                    Counter(s["routing"] for s in after_rows if s["shown_as"] != "defect")
                ),
                "defect_images_also_seen_before_alert": sum(
                    1 for s in defects_after if s["image"] in seen_images
                ),
                "new_alerts": sum(1 for s in after_rows if s.get("alert_event_id")),
            },
            "vision_cases_total": len(stack.vision_cases()),
            "inspect_latency_median_s": round(statistics.median(latencies), 3),
        }
    )
    return out


def _count(stack: Stack, context: str, cause: str | None) -> int:
    rows = stack.diagnosis.get("/v1/feedback/counts").json()["counts"]
    return sum(c["count"] for c in rows if (c["context"], c["cause_code"]) == (context, cause))


def phase_false_clusters(
    stack: Stack, w: dict[str, Any], run_id: str, args: argparse.Namespace
) -> dict:
    m: pd.DataFrame = w["manifest"]
    rows = list(m.itertuples())[: args.max_parts]
    per_image: dict[str, list[float]] = {}
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    first_alert_part: int | None = None
    log(f"streaming {len(rows)} parts of a fault-free run (good images only)")
    for n, row in enumerate(rows, 1):
        res, dt = stack.inspect(row, run_id)
        latencies.append(dt)
        per_image.setdefault(row.image, []).append(res["score"])
        results.append({"seq": int(row.seq), "image": row.image, "shown_as": row.shown_as, **res})
        if res.get("alert_event_id") and first_alert_part is None:
            first_alert_part = n
            log(f"FALSE ALERT after {n} parts (image {row.image}, score {res['score']:.3f})")
        if n % 250 == 0:
            log(f"{n} parts, anomalous so far {sum(r['decision'] == 'anomalous' for r in results)}")
    time.sleep(5)  # let the diagnosis worker pick up a last event
    cases = stack.vision_cases()
    threshold = results[0]["threshold"] if results else None
    distinct = {
        img: {"score": round(s[0], 4), "draws": len(s), "above": s[0] > (threshold or 1e9)}
        for img, s in sorted(per_image.items())
    }
    clusters = stack.vision.get("/v1/clusters", params={"category": CATEGORY}).json()
    return {
        "parts_streamed": len(results),
        "shown_as": dict(Counter(r["shown_as"] for r in results)),
        "threshold": threshold,
        "anomalous_parts": sum(r["decision"] == "anomalous" for r in results),
        "distinct_good_images": len(distinct),
        "distinct_good_images_above_threshold": sorted(
            img for img, d in distinct.items() if d["above"]
        ),
        "per_image": distinct,
        "false_clusters": [
            {k: c[k] for k in ("cluster_id", "size", "status", "radius")} for c in clusters
        ],
        "false_alert_after_parts": first_alert_part,
        "bogus_cases_opened": len(cases),
        "bogus_case_phases": [c["phase"] for c in cases],
        "inspect_latency_median_s": round(statistics.median(latencies), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--phase", choices=["demo", "false-clusters"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pre-onset", type=int, default=30, help="parts streamed before the fault")
    ap.add_argument("--after-naming", type=int, default=20, help="defect parts after naming")
    ap.add_argument("--max-parts", type=int, default=10_000, help="false-clusters: parts")
    ap.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--confirm-cause", default=None)
    ap.add_argument("--keep", action="store_true", help="leave the stack running")
    args = ap.parse_args()
    if not args.oracle and not args.confirm_cause:
        ap.error("--no-oracle needs --confirm-cause")
    for p in (POOL / CATEGORY, MODELS / "patchcore" / CATEGORY / "deploy", MODELS / "weights"):
        if not p.exists():
            ap.error(f"missing {p} (read-only symlinks, owner decision D2)")

    scenario = "tool_wear_m1" if args.phase == "demo" else "baseline"
    run_id = f"run-de{args.seed:010x}" if args.phase == "demo" else f"run-0f{args.seed:010x}"
    log(f"phase {args.phase}: {LABEL}")
    t0 = time.perf_counter()
    w = build_world(scenario, args.seed, run_id)
    prep_s = time.perf_counter() - t0
    stack = Stack()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "reports/quality_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        up_s = stack.up()
        log(f"stack up in {up_s:.0f} s")
        info = model_info(stack)
        t1 = time.perf_counter()
        body = (phase_demo if args.phase == "demo" else phase_false_clusters)(
            stack, w, run_id, args
        )
        events = stack.events()
        report = {
            "what": f"step-7 quality loop, phase {args.phase}",
            "data": LABEL,
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": "dev laptop (WSL2, 16 vCPU, ~9.7 GiB VM), CPU inference",
            "scenario": scenario,
            "scenario_seed": args.seed,
            "simulator_run_id": run_id,
            "category": CATEGORY,
            "model": {
                k: info.get(k)
                for k in ("version", "threshold", "threshold_method", "threshold_alpha")
                if k in info
            }
            or info,
            "library_cases": w["library_cases"],
            "prepare_s": round(prep_s, 1),
            "stack_up_s": round(up_s, 1),
            "phase_s": round(time.perf_counter() - t1, 1),
            "outbox_events": events,
            **body,
        }
        path = out_dir / f"demo_{args.phase.replace('-', '_')}_{stamp}.json"
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        log(f"report {path.relative_to(REPO)}")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("per_image", "cause_set")},
                indent=1,
                default=str,
            )
        )
    finally:
        if not args.keep:
            stack.down(WORK / f"compose_logs_{args.phase}_{stamp}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
