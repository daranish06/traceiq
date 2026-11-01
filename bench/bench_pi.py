#!/usr/bin/env python3
"""
Benchmark on the device (Pi or laptop). Writes bench/results_<host>.csv and prints a summary.
  python bench/bench_pi.py --images datasets_sample --runs 20 --llm   (on the Pi)
Vision: per-stage latency (tiling, inference, merge) mean/p95 over N boards, boards/min.
Llama (llama_main): model load ms, time to first token, prompt tokens, generated tokens, tokens/s.
System: peak RAM used, CPU temperature, throttling flags (vcgencmd), before/after.
"""
import argparse, csv, json, re, socket, statistics, subprocess, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "app"), str(ROOT / "advisor")]
from inspect_board import Detector  # noqa: E402
import advisor as A  # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def sysinfo():
    mem = sh("free -m | awk '/Mem:/{print $3}'")
    return {"ram_used_mb": int(mem) if mem.isdigit() else None,
            "temp_c": sh("vcgencmd measure_temp | grep -o '[0-9.]*'") or None,
            "throttled": sh("vcgencmd get_throttled | cut -d= -f2") or None}


def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/pcb_combined_y11n_ncnn_model"))
    ap.add_argument("--thresholds", default=str(ROOT / "models/class_thresholds.json"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--llm", action="store_true", help="also benchmark Llama via llama_main")
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    a = ap.parse_args()

    host = socket.gethostname()
    before = sysinfo()
    thr = json.loads(Path(a.thresholds).read_text()) if Path(a.thresholds).exists() else {}
    det = Detector(a.model, class_thresholds=thr)
    imgs = sorted(Path(a.images).glob("*.jpg"))
    det(cv2.imread(str(imgs[0])))  # warm-up
    rows, peak = [], before["ram_used_mb"] or 0
    t_all = time.perf_counter()
    for i in range(a.runs):
        img = cv2.imread(str(imgs[i % len(imgs)]))
        t0 = time.perf_counter(); dets, t = det(img); t["total_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        t["n_dets"] = len(dets); rows.append(t)
        peak = max(peak, sysinfo()["ram_used_mb"] or 0)
    wall = time.perf_counter() - t_all
    after = sysinfo()
    summ = {"host": host, "model": Path(a.model).name, "boards": a.runs, "tiles_per_board": rows[0]["tiles"],
            "vision_mean_ms": round(statistics.mean(r["total_ms"] for r in rows), 1),
            "vision_p95_ms": pct([r["total_ms"] for r in rows], 95),
            "infer_mean_ms": round(statistics.mean(r["infer_ms"] for r in rows), 1),
            "boards_per_min": round(60 * a.runs / wall, 1),
            "peak_ram_mb": peak, "temp_before": before["temp_c"], "temp_after": after["temp_c"],
            "throttled_after": after["throttled"]}

    if a.llm:
        cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
               for k, v in json.loads(Path(a.config).read_text())["llama_main"].items()}
        recs = [A.make_record("short", 0.9, (1200, 400, 1240, 430), 3034, 1586),
                A.make_record("mouse_bite", 0.8, (300, 1300, 330, 1320), 3034, 1586)]
        insp = A.Inspection("BENCH", "", A.verdict_for(recs), recs)
        cmd = [cfg["llama_main"], "--model_path", cfg["pte"], "--tokenizer_path", cfg["tokenizer"],
               "--seq_len", "1024", "--temperature", "0", "--prompt", A.build_prompt(insp)]
        env = {"LD_LIBRARY_PATH": str(Path(cfg["llama_main"]).parent)}
        llm = []
        for _ in range(3):
            t0 = time.perf_counter()
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            wall_s = time.perf_counter() - t0
            m = re.search(r"PyTorchObserver\s*(\{.*\})", out.stdout + out.stderr)
            st = json.loads(m.group(1)) if m else {}
            gen = st.get("generated_tokens", 0)
            dec_ms = st.get("inference_end_ms", 0) - st.get("prompt_eval_end_ms", 0)
            llm.append({"wall_s": round(wall_s, 2),
                        "load_ms": st.get("model_load_end_ms", 0) - st.get("model_load_start_ms", 0),
                        "ttft_ms": st.get("first_token_ms", 0) - st.get("inference_start_ms", 0),
                        "prompt_tokens": st.get("prompt_tokens"), "generated_tokens": gen,
                        "tok_per_s": round(gen / (dec_ms / 1000), 1) if dec_ms > 0 else None})
            print("llama run:", llm[-1])
        summ.update({f"llm_{k}": llm[-1][k] for k in llm[-1]})
        summ["llm_first_run_load_ms"] = llm[0]["load_ms"]
        summ["temp_after_llm"] = sysinfo()["temp_c"]

    print("\n" + "\n".join(f"{k:24s} {v}" for k, v in summ.items()))
    out = ROOT / "bench" / f"results_{host}.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ)); new and w.writeheader(); w.writerow(summ)
    print("appended to", out)


if __name__ == "__main__":
    main()
