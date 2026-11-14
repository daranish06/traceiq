#!/usr/bin/env python3
"""
Benchmark on the device (Pi or laptop). Writes bench/results_<host>.csv and prints a summary.
  python bench/bench_pi.py --stage 1 --images samples/stage1 --runs 20          (on the Pi)
  python bench/bench_pi.py --stage 2 --images samples/stage2 --runs 40 --llm
Vision: per-stage latency (tiling, inference, merge) mean/p95 over N boards, boards/min.
Llama (llama_main): model load ms, time to first token, prompt tokens, generated tokens, tokens/s.
System: peak RAM used, CPU temperature, throttling flags (vcgencmd), before/after.
"""
import argparse, csv, json, re, socket, statistics, subprocess, sys, threading, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "app"), str(ROOT / "advisor")]
from inspect_board import STAGES, Detector, list_images, stage_paths  # noqa: E402
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
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--laptop", action="store_true", help="use the .pt model instead of NCNN")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--llm", action="store_true", help="also benchmark Llama via llama_main")
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    a = ap.parse_args()

    host = socket.gethostname()
    before = sysinfo()
    model, thr_file = stage_paths(a.stage, pi=not a.laptop)
    thr = json.loads(Path(thr_file).read_text()) if Path(thr_file).exists() else {}
    st = STAGES[a.stage]
    det = Detector(model, class_thresholds=thr, tile=st["imgsz"], tiling=st["tiling"])
    a.model = model
    imgs = list_images(a.images)
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
    summ = {"host": host, "stage": a.stage, "model": Path(a.model).name, "boards": a.runs,
            "tiles_per_board_mean": round(statistics.mean(r["tiles"] for r in rows), 1),
            "vision_mean_ms": round(statistics.mean(r["total_ms"] for r in rows), 1),
            "vision_p95_ms": pct([r["total_ms"] for r in rows], 95),
            "preprocess_mean_ms": round(statistics.mean(r["tile_ms"] for r in rows), 1),
            "infer_mean_ms": round(statistics.mean(r["infer_ms"] for r in rows), 1),
            "postprocess_mean_ms": round(statistics.mean(r["merge_ms"] for r in rows), 1),
            "fps": round(a.runs / wall, 2),
            "boards_per_min": round(60 * a.runs / wall, 1),
            "peak_ram_mb": peak, "temp_before": before["temp_c"], "temp_after": after["temp_c"],
            "throttled_after": after["throttled"]}

    if a.llm:
        cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
               for k, v in json.loads(Path(a.config).read_text())["llama_main"].items()}
        if a.stage == 1:
            recs = [A.make_record("short", 0.9, (1200, 400, 1240, 430), 3034, 1586),
                    A.make_record("mouse_bite", 0.8, (300, 1300, 330, 1320), 3034, 1586)]
        else:
            recs = [A.make_record("solder_bridge", 0.9, (300, 300, 360, 340), 1280, 1280, None),
                    A.make_record("insufficient_solder", 0.8, (900, 900, 950, 950), 1280, 1280, None)]
        insp = A.Inspection("BENCH", "", A.verdict_for(recs), recs)
        cmd = [cfg["llama_main"], "--model_path", cfg["pte"], "--tokenizer_path", cfg["tokenizer"],
               "--seq_len", "1024", "--temperature", "0", "--prompt", A.build_prompt(insp)]
        env = {"LD_LIBRARY_PATH": str(Path(cfg["llama_main"]).parent)}
        llm = []
        for _ in range(3):
            peak_llm = [0]
            done = threading.Event()

            def watch():  # system RAM in use while Llama runs (vision model is also loaded)
                while not done.is_set():
                    peak_llm[0] = max(peak_llm[0], sysinfo()["ram_used_mb"] or 0)
                    time.sleep(0.2)
            th = threading.Thread(target=watch); th.start()
            t0 = time.perf_counter()
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            wall_s = time.perf_counter() - t0
            done.set(); th.join()
            m = re.search(r"PyTorchObserver\s*(\{.*\})", out.stdout + out.stderr)
            st = json.loads(m.group(1)) if m else {}
            gen = st.get("generated_tokens", 0)
            dec_ms = st.get("inference_end_ms", 0) - st.get("prompt_eval_end_ms", 0)
            llm.append({"wall_s": round(wall_s, 2),
                        "load_ms": st.get("model_load_end_ms", 0) - st.get("model_load_start_ms", 0),
                        "ttft_ms": st.get("first_token_ms", 0) - st.get("inference_start_ms", 0),
                        "prompt_tokens": st.get("prompt_tokens"), "generated_tokens": gen,
                        "tok_per_s": round(gen / (dec_ms / 1000), 1) if dec_ms > 0 else None,
                        "prefill_tok_per_s": round(st.get("prompt_tokens", 0) / (
                            (st.get("prompt_eval_end_ms", 0) - st.get("inference_start_ms", 0)) / 1000), 1)
                        if st.get("prompt_eval_end_ms", 0) > st.get("inference_start_ms", 0) else None,
                        "response_s_excl_load": round((st.get("inference_end_ms", 0)
                                                       - st.get("inference_start_ms", 0)) / 1000, 2),
                        "peak_system_ram_mb": peak_llm[0]})
            print("llama run:", llm[-1])
        for k in llm[-1]:  # mean of the 3 runs
            vals = [r[k] for r in llm if isinstance(r[k], (int, float))]
            summ[f"llm_{k}"] = round(statistics.mean(vals), 2) if vals else llm[-1][k]
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
