#!/usr/bin/env python3
"""P5a: isolated transport microbench, development gates, conditional unique r3."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import signal
import subprocess
import time

import p3_runner as p3
import p4a_runner as p4

ROOT = p3.ROOT
SOURCE = p4.SOURCE
PROFILE_PATH = SOURCE / "benchmark/pd_4090/p5a_profiles.json"
DEV = ROOT / "bench/dev_toolagent_1200_240_x2.5.jsonl"
FORMAL = p3.TRACE
PROFILES = json.loads(PROFILE_PATH.read_text())["profiles"]


def replay(cluster, formal=False):
    label = "sglang_pd_r3" if formal else "dev300"
    out = ROOT / "results" if formal else cluster.path / "dev300"
    trace = FORMAL if formal else DEV
    cmd = [p3.PYTHON, str(ROOT / "bench/bench_client.py"), "--trace", str(trace),
           "--base-url", p3.BASE, "--label", label, "--out-dir", str(out),
           "--warmup", "2", "--force-close"]
    if not formal:
        cmd += ["--limit", "300", "--time-scale", "1"]
        with (ROOT / "logs/p5a" / f"development-{cluster.meta['profile']}.started.json").open("x") as f:
            json.dump({"command": cmd, "log_dir": str(cluster.path)}, f, indent=2)
    else:
        with (out / f"{label}.formal-started.json").open("x") as f:
            json.dump({"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "command": cmd, "log_dir": str(cluster.path)}, f, indent=2)
    stage = "formal" if formal else "dev300"
    p3.save(cluster.path / f"{stage}-command.json", cmd)
    print("RUN " + shlex.join(cmd), flush=True)
    with (cluster.path / f"{stage}-client.log").open("x") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=4300)
        except BaseException:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    summary = json.loads((out / f"{label}_summary.json").read_text())
    cluster.meta[stage + "_summary"] = summary
    cluster.meta[stage + "_returncode"] = rc
    p3.save(cluster.path / "run.json", cluster.meta)
    print(f"{label}: complete={summary['completed_requests']} fail={summary['failed_requests']} duration={summary['duration_sec']:.3f}", flush=True)
    assert summary["planned_requests"] == (1551 if formal else 300)
    if not formal:
        assert rc == 0 and summary["completed_requests"] == 300 and summary["failed_requests"] == 0 and summary["unsent_requests"] == 0
    return summary


def micro(cluster, pairs):
    rng = random.Random(530020)
    records = []
    for p, d in pairs:
        for n in [64, 4096, 16384, 31663]:
            p3.flush(cluster)
            room = rng.randrange(1, 2**62)
            payload = {"input_ids": [128000] + [rng.randrange(128000) for _ in range(n-1)],
                       "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
                       "bootstrap_host": "127.0.0.1", "bootstrap_port": 8998 + p,
                       "bootstrap_room": room}
            def call(gpu):
                response = p3.post(f"http://127.0.0.1:{31000+gpu}/generate", json=payload, timeout=120)
                response.raise_for_status()
                return response.json()
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fp = pool.submit(call, p)
                fd = pool.submit(call, d)
                presult, dresult = fp.result(), fd.result()
            assert dresult["meta_info"]["completion_tokens"] == 8
            row = {"prefill_gpu": p, "decode_gpu": d, "tokens": n, "room": room,
                   "e2e_s": time.monotonic()-start, "prefill_response": presult,
                   "decode_response": dresult, "warmup": n == 64}
            records.append(row)
            p3.save(cluster.path / "micro-requests.json", records)
            print(f"micro {p}->{d} n={n} e2e={row['e2e_s']:.3f}", flush=True)
    parse_micro(cluster.path)


def parse_micro(path):
    events = {}
    pattern = re.compile(r"P5A_KV event=(\w+) ns=(\d+) room=(\d+) tokens=(\d+) start=(\d+) end=(\d+) last=(\d+)")
    for logfile in path.glob("worker*.log"):
        for match in pattern.finditer(logfile.read_text()):
            event, ns, room, tokens, start, end, last = match.groups()
            events.setdefault(int(room), []).append(dict(event=event, ns=int(ns), start=int(start), end=int(end), last=int(last)))
    result = []
    for row in json.loads((path / "micro-requests.json").read_text()):
        ev = events[row["room"]]
        sends = sorted((e for e in ev if e["event"] == "send"), key=lambda e:e["ns"])
        recv = next(e["ns"] for e in ev if e["event"] == "received")
        final = next(e for e in sends if e["last"])
        residual = (recv-final["ns"])/1e9
        span = (recv-sends[0]["ns"])/1e9
        total_bytes = row["tokens"] * 131072
        result.append({k:v for k,v in row.items() if not k.endswith("_response")} | {
            "kv_bytes": total_bytes, "prefill_complete_to_received_s": residual,
            "first_send_to_received_s": span, "effective_total_kv_GBps": total_bytes/span/1e9,
            "last_chunk_bytes": (final["end"]-final["start"])*131072,
            "events": ev})
    p3.save(path / "micro-summary.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["micro", "development", "experiment"])
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--pairs", default="0:2,0:3,1:2,1:3")
    args = parser.parse_args()
    assert not any(k.startswith("P3_") for k in os.environ)
    assert not subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True).strip()
    source_commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    profile = PROFILES[args.profile]
    os.environ["P3_PD_BACKEND"] = profile["backend"]
    orig = p3.commands
    def commands(label):
        jobs = orig(label)
        for name, cmd, env, port in jobs:
            env["PYTHONPATH"] = str(SOURCE/"python")
            env["PYTHONUNBUFFERED"] = "1"
            if name != "router":
                env.update(profile["env"])
                env["P5A_KV_TIMING"] = "1"
            else:
                env["PYTHONPATH"] = str(SOURCE/"sgl-router/py_src")+":"+str(SOURCE/"python")
        return jobs
    p3.commands = commands
    fingerprint = hashlib.sha256(json.dumps(commands("sglang_pd"), sort_keys=True).encode()).hexdigest()
    if args.mode == "experiment":
        assert not (ROOT/"results/sglang_pd_r3.formal-started.json").exists()
        assert not list((ROOT/"results").glob("sglang_pd_r3_*"))
        assert args.gate and args.baseline
        gate = json.loads((args.gate/"run.json").read_text())
        base = json.loads((args.baseline/"run.json").read_text())
        assert gate["status"] == base["status"] == "PASS"
        assert base["profile"] == "mooncake-r2"
        assert gate["configuration_sha256"] == fingerprint and gate["git_commit"] == source_commit
        for pth, record in [(args.gate, gate), (args.baseline, base)]:
            s = record["dev300_summary"]
            assert s["trace"] == str(DEV) and s["planned_requests"] == s["completed_requests"] == 300
            assert s["failed_requests"] == s["unsent_requests"] == 0
            assert json.loads((pth/"correctness-comparison.json").read_text())["passed"] == 20
            assert len(json.loads((pth/"long-warmup.json").read_text())) == 2
        a, b = gate["dev300_summary"], base["dev300_summary"]
        ratio = a["throughput"]["input_tokens_per_sec"] / b["throughput"]["input_tokens_per_sec"]
        ttft_ratio = a["latencies_sec"]["ttft"]["p50"] / b["latencies_sec"]["ttft"]["p50"]
        assert ratio >= 1.5 or ttft_ratio <= 0.5
        p3.save(ROOT/"logs/p5a/formal-entry-check.json", {"throughput_ratio":ratio,"ttft_ratio":ttft_ratio,"gate":str(args.gate),"baseline":str(args.baseline),"git_commit":source_commit})
    cluster = p3.Cluster("sglang_pd", ROOT/"logs/p5a"/args.tag)
    cluster.meta.update(profile=args.profile, source_dir=str(SOURCE), git_commit=source_commit,
                        configuration_sha256=fingerprint, mode=args.mode,
                        development_trace=str(DEV), profile_file=str(PROFILE_PATH))
    def term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, term)
    try:
        cluster.start()
        (cluster.path/"p5a_runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
        (cluster.path/"profiles.snapshot.json").write_bytes(PROFILE_PATH.read_bytes())
        if args.mode == "micro":
            micro(cluster, [tuple(map(int,p.split(":"))) for p in args.pairs.split(",")])
        else:
            p4.validate(cluster)
            if args.mode == "experiment":
                # Keep the P3b/P4a streaming smoke lifecycle, using development
                # inputs exclusively before the unique formal marker is made.
                p3.TRACE = DEV
                p3.bench(cluster, smoke=True)
                p3.TRACE = FORMAL
            p3.flush(cluster)
            cluster.reset_router()
            replay(cluster, formal=args.mode == "experiment")
        cluster.meta["status"] = "PASS"
    except BaseException as exc:
        cluster.meta.update(status="FAIL", error=repr(exc))
        raise
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
