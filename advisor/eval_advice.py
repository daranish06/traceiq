#!/usr/bin/env python3
"""
Advice quality check: run the full pipeline + Llama on N test boards and write a rating sheet.
  python advisor/eval_advice.py --n 25 --backend inproc              # Stage 1 (bare boards)
  python advisor/eval_advice.py --n 25 --backend inproc --stage 2    # Stage 2 (assembled boards)
Writes bench/advice_eval.csv (Stage 2: bench/advice_eval_asm.csv). The team then fills the rating columns (0 = wrong, 1 = partly, 2 = good):
  correct     - matches the detected defects and their locations
  grounded    - causes/actions are sensible for PCB fabrication (no invented facts)
  actionable  - an operator could act on it
Automatic metrics printed: guard pass rate (LLM answers that mention only detected defects and use the
Problem/Cause/Action format), and advice time.
"""
import argparse, csv, json, random, statistics, sys, time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "app"), str(ROOT / "advisor")]
import advisor as A  # noqa: E402
from inspect_board import STAGES, Detector, label_path, list_images, stage_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--backend", default="native", choices=["native", "inproc", "llama_main"])
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    if a.stage == 1:
        pku = list_images(ROOT / "datasets/pku_board/test/images")
        ds = list_images(ROOT / "datasets/dspcbsd_v3/test/images")
        imgs = random.sample(pku, min(len(pku), a.n // 2)) + random.sample(ds, min(len(ds), a.n - a.n // 2))
    else:  # only images that contain defects, so every sample exercises the Llama
        asm = [p for p in list_images(ROOT / "datasets/asm_raw/final_solder/images/test")
               if label_path(p).exists() and label_path(p).read_text().strip()]
        imgs = random.sample(asm, min(len(asm), a.n))
    cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
           for k, v in json.loads(Path(a.config).read_text())["native" if a.backend == "inproc" else a.backend].items()}
    model, thr_file = stage_paths(a.stage)
    thr = json.loads(Path(thr_file).read_text()) if Path(thr_file).exists() else {}
    st = STAGES[a.stage]
    det = Detector(model, class_thresholds=thr, tile=st["imgsz"], tiling=st["tiling"])

    rows, times, passed, llm_used, reasons = [], [], 0, 0, Counter()
    for i, ip in enumerate(imgs, 1):
        img = cv2.imread(str(ip)); H, W = img.shape[:2]
        dets, _ = det(img)
        recs = [A.make_record(c, p, b, W, H, st["area_frac"]) for c, p, b in dets if c in A.KNOWLEDGE]
        insp = A.Inspection(f"Q-{i:03d}", time.strftime("%Y-%m-%dT%H:%M:%S"), A.verdict_for(recs), recs)
        A.advise(insp, a.backend, cfg)
        if recs:
            llm_used += 1; passed += insp.guard_passed; times.append(insp.advisor_seconds)
            if not insp.guard_passed:
                reasons[insp.advice.rsplit("rejected by guard: ", 1)[-1].rstrip(")").split(":")[0]] += 1
        rows.append({"board": insp.board_id, "image": ip.name, "verdict": insp.verdict,
                     "defects": "; ".join(A.defect_summary(insp)), "backend": insp.advisor_backend,
                     "guard_passed": insp.guard_passed, "advisor_s": insp.advisor_seconds,
                     "llm_raw": insp.raw_llm, "advice_shown": insp.advice,
                     "correct_0_2": "", "grounded_0_2": "", "actionable_0_2": "", "notes": ""})
        print(f"[{i}/{len(imgs)}] {ip.name[:40]:40s} {insp.verdict:6s} guard={insp.guard_passed} {insp.advisor_seconds}s")

    out = ROOT / ("bench/advice_eval.csv" if a.stage == 1 else "bench/advice_eval_asm.csv"); out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nboards with defects: {llm_used}/{len(imgs)}")
    if llm_used:
        print(f"guard pass rate (LLM answer used as-is): {passed}/{llm_used} = {100 * passed / llm_used:.0f}%")
        print(f"advice time: mean {statistics.mean(times):.1f} s, max {max(times):.1f} s")
        if reasons:
            print("guard rejections:", dict(reasons))
    print("rating sheet:", out, "(fill the *_0_2 columns)")


if __name__ == "__main__":
    main()
