#!/usr/bin/env python3
"""
Advice quality check: run the full pipeline + Llama on N test boards and write a rating sheet.
  python advisor/eval_advice.py --n 25 --backend native
Writes bench/advice_eval.csv. The team then fills the rating columns (0 = wrong, 1 = partly, 2 = good):
  correct     - matches the detected defects and their locations
  grounded    - causes/actions are sensible for PCB fabrication (no invented facts)
  actionable  - an operator could act on it
Automatic metrics printed: guard pass rate (LLM answers that mention only detected defects and use the
Problem/Cause/Action format), and advice time.
"""
import argparse, csv, json, random, statistics, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "app"), str(ROOT / "advisor")]
import advisor as A  # noqa: E402
from inspect_board import Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--backend", default="native", choices=["native", "inproc", "llama_main"])
    ap.add_argument("--model", default=str(ROOT / "models/pcb_combined_y11n.pt"))
    ap.add_argument("--thresholds", default=str(ROOT / "models/class_thresholds.json"))
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    pku = sorted((ROOT / "datasets/pku_board/test/images").glob("*.jpg"))
    ds = sorted((ROOT / "datasets/dspcbsd_v3/test/images").glob("*.jpg"))
    imgs = random.sample(pku, min(len(pku), a.n // 2)) + random.sample(ds, min(len(ds), a.n - a.n // 2))
    cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
           for k, v in json.loads(Path(a.config).read_text())["native" if a.backend == "inproc" else a.backend].items()}
    thr = json.loads(Path(a.thresholds).read_text()) if Path(a.thresholds).exists() else {}
    det = Detector(a.model, class_thresholds=thr)

    rows, times, passed, llm_used = [], [], 0, 0
    for i, ip in enumerate(imgs, 1):
        img = cv2.imread(str(ip)); H, W = img.shape[:2]
        dets, _ = det(img)
        recs = [A.make_record(c, p, b, W, H) for c, p, b in dets if c in A.KNOWLEDGE]
        insp = A.Inspection(f"Q-{i:03d}", time.strftime("%Y-%m-%dT%H:%M:%S"), A.verdict_for(recs), recs)
        A.advise(insp, a.backend, cfg)
        if recs:
            llm_used += 1; passed += insp.guard_passed; times.append(insp.advisor_seconds)
        rows.append({"board": insp.board_id, "image": ip.name, "verdict": insp.verdict,
                     "defects": "; ".join(A.defect_summary(insp)), "backend": insp.advisor_backend,
                     "guard_passed": insp.guard_passed, "advisor_s": insp.advisor_seconds,
                     "llm_raw": insp.raw_llm, "advice_shown": insp.advice,
                     "correct_0_2": "", "grounded_0_2": "", "actionable_0_2": "", "notes": ""})
        print(f"[{i}/{len(imgs)}] {ip.name[:40]:40s} {insp.verdict:6s} guard={insp.guard_passed} {insp.advisor_seconds}s")

    out = ROOT / "bench/advice_eval.csv"; out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nboards with defects: {llm_used}/{len(imgs)}")
    if llm_used:
        print(f"guard pass rate (LLM answer used as-is): {passed}/{llm_used} = {100 * passed / llm_used:.0f}%")
        print(f"advice time: mean {statistics.mean(times):.1f} s, max {max(times):.1f} s")
    print("rating sheet:", out, "(fill the *_0_2 columns)")


if __name__ == "__main__":
    main()
