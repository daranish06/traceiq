#!/usr/bin/env python3
"""
TraceIQ operator dashboard (Gradio, runs in a browser; on the Pi open http://traceiq.local:7860).

Flow per board: capture or upload -> detection result shown immediately -> Llama advice added
a few seconds later -> logged to output/inspections.csv (+ annotated image, + optional GPIO alert).

  python app/ui.py                                  # laptop, template advice
  python app/ui.py --backend inproc                 # laptop, Llama kept loaded in-process (ExecuTorch)
  python app/ui.py --pi --backend llama_main --gpio # Raspberry Pi (NCNN models, llama_main, buzzer/LEDs)

The operator picks Stage 1 (bare board) or Stage 2 (assembled board) in the dashboard;
each stage's model is loaded on first use and then kept in memory.
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "advisor"))
sys.path.insert(0, str(ROOT / "app"))
import advisor as A  # noqa: E402
from inspect_board import STAGES, Detector, annotate, append_logs, stage_paths  # noqa: E402

VERDICT_CSS = {"PASS": "#1a7f37", "REVIEW": "#b26b00", "FAIL": "#c62828"}


# ---------------------------------------------------------------- camera (Pi camera, else USB webcam)
class Camera:
    def __init__(self):
        self.kind, self.cam = None, None

    def _open(self):
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_still_configuration(main={"format": "RGB888"}))
            self.cam.start()
            time.sleep(1.0)  # let auto-exposure / autofocus settle
            self.kind = "picamera2"
        except Exception:  # noqa: BLE001
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cam, self.kind = cap, "opencv"

    def capture_bgr(self) -> np.ndarray | None:
        if self.cam is None:
            self._open()
        if self.kind == "picamera2":
            try:
                self.cam.autofocus_cycle()
            except Exception:  # noqa: BLE001
                pass
            return self.cam.capture_array()  # Picamera2 "RGB888" is BGR byte order
        if self.kind == "opencv":
            for _ in range(3):  # drop stale frames
                self.cam.read()
            ok, frame = self.cam.read()
            return frame if ok else None
        return None


# ---------------------------------------------------------------- optional GPIO alert
class Alert:
    def __init__(self, enabled: bool, buzzer=17, red=27, green=22):
        self.ok = False
        if not enabled:
            return
        try:
            from gpiozero import Buzzer, LED
            self.buzzer, self.red, self.green = Buzzer(buzzer), LED(red), LED(green)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print("GPIO alert disabled:", e)

    def show(self, verdict: str):
        if not self.ok:
            return
        self.red.off(); self.green.off()
        if verdict == "PASS":
            self.green.on()
        else:
            self.red.on()
            self.buzzer.beep(on_time=0.15, off_time=0.1, n=3 if verdict == "FAIL" else 1)


# ---------------------------------------------------------------- app
def build_app(a):
    import gradio as gr

    out_dir = Path(a.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    detectors: dict[int, Detector] = {}

    def detector(stage: int) -> Detector:
        if stage not in detectors:
            model, thr_file = stage_paths(stage, a.pi)
            if not Path(model).exists():
                raise gr.Error(f"{STAGES[stage]['name']} model not found: {model}")
            thr = json.loads(Path(thr_file).read_text()) if Path(thr_file).exists() else {}
            d = Detector(model, class_thresholds=thr, tile=STAGES[stage]["imgsz"], tiling=STAGES[stage]["tiling"])
            d(np.zeros((640, 640, 3), np.uint8))  # warm-up
            detectors[stage] = d
        return detectors[stage]

    detector(1)  # Stage 1 ready at start; Stage 2 loads on first use
    stage_names = {STAGES[k]["name"]: k for k in STAGES}
    cfg = {}
    if a.backend != "template":
        cfg = {k: str(Path(v).expanduser()) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v
               for k, v in json.loads(Path(a.config).read_text())["native" if a.backend == "inproc" else a.backend].items()}
    camera, alert = Camera(), Alert(a.gpio)
    csv_path = out_dir / "inspections.csv"

    def next_board_id() -> str:
        n = sum(1 for _ in open(csv_path)) - 1 if csv_path.exists() else 0
        return f"B-{n + 1:04d}"

    def stats_md() -> str:
        if not csv_path.exists():
            return "No boards inspected yet."
        rows = list(csv.DictReader(open(csv_path)))
        v = Counter(r["verdict"] for r in rows)
        kinds = Counter()
        for r in rows:
            for part in filter(None, r["defects"].split("; ")):
                kinds[part.split(" (")[0].split(" x")[0]] += 1
        fail_rate = 100 * (v["FAIL"] + v["REVIEW"]) / len(rows)
        top = ", ".join(f"{k} ({n})" for k, n in kinds.most_common(4)) or "-"
        return (f"**Boards inspected:** {len(rows)} &nbsp;|&nbsp; PASS {v['PASS']} · REVIEW {v['REVIEW']} · "
                f"FAIL {v['FAIL']} &nbsp;|&nbsp; **Reject/review rate:** {fail_rate:.0f}%  \n"
                f"**Most frequent defects:** {top}")

    def banner(insp: A.Inspection, extra: str = "") -> str:
        return (f"<div style='background:{VERDICT_CSS[insp.verdict]};color:white;padding:14px 18px;"
                f"border-radius:10px;font-size:28px;font-weight:700'>{insp.board_id} &nbsp; {insp.verdict}"
                f"<span style='font-size:16px;font-weight:400'> &nbsp; {len(insp.defects)} defect(s){extra}</span></div>")

    def inspect(image_rgb, source, stage_name):
        stage = stage_names[stage_name]
        det = detector(stage)
        if source == "Camera":
            bgr = camera.capture_bgr()
            if bgr is None:
                raise gr.Error("No camera found. Use 'Upload image' instead.")
        else:
            if image_rgb is None:
                raise gr.Error("Upload a board image first.")
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        t0 = time.perf_counter()
        dets, timing = det(bgr)
        recs = [A.make_record(c, p, b, W, H, STAGES[stage]["area_frac"]) for c, p, b in dets if c in A.KNOWLEDGE]
        insp = A.Inspection(next_board_id(), time.strftime("%Y-%m-%dT%H:%M:%S"), A.verdict_for(recs), recs)
        timing["vision_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        alert.show(insp.verdict)
        ann_rgb = cv2.cvtColor(annotate(bgr, insp), cv2.COLOR_BGR2RGB)
        table = [[A.KNOWLEDGE[d.defect_class]["label"], d.severity, d.region, round(d.confidence, 2)]
                 for d in insp.defects]
        timing_md = f"{STAGES[stage]['name']} · Vision: {timing['vision_ms']} ms ({timing['tiles']} tile(s))"
        # 1) show detection immediately
        yield (ann_rgb, banner(insp), table, "_Generating advice..._" if insp.defects else "No defects found.",
               timing_md, stats_md(), None)
        # 2) advice (Llama or template), then log
        A.advise(insp, a.backend, cfg)
        timing["advisor_ms"] = round(insp.advisor_seconds * 1e3, 1)
        ann_path = out_dir / "annotated" / f"{insp.board_id}.jpg"
        cv2.imwrite(str(ann_path), annotate(bgr, insp), [cv2.IMWRITE_JPEG_QUALITY, 90])
        append_logs(insp, timing, out_dir, source, str(ann_path))
        advice_md = insp.advice.replace("\n", "  \n") + f"\n\n`LOG: {insp.log_line}`"
        if not insp.guard_passed:
            advice_md += "\n\n_(LLM answer failed the safety check; template advice shown)_"
        timing_md += f" · Advice: {insp.advisor_seconds:.1f} s ({insp.advisor_backend})"
        yield ann_rgb, banner(insp), table, advice_md, timing_md, stats_md(), str(csv_path)

    with gr.Blocks(title="TraceIQ") as app:
        gr.Markdown("## TraceIQ: PCB defect inspection &nbsp;·&nbsp; on-device AI (YOLO + Llama 3.2 1B / ExecuTorch)")
        with gr.Row():
            with gr.Column(scale=1):
                stage_in = gr.Radio(list(stage_names), value=STAGES[1]["name"], label="Inspection stage")
                source = gr.Radio(["Upload image", "Camera"], value="Upload image", label="Input")
                image = gr.Image(type="numpy", label="Board image (for upload)", height=280)
                run = gr.Button("Inspect board", variant="primary", size="lg")
                stats = gr.Markdown(stats_md())
                log_file = gr.File(label="Inspection log (CSV)")
            with gr.Column(scale=2):
                verdict = gr.HTML()
                result = gr.Image(label="Result", height=460, interactive=False)
                timing = gr.Markdown()
        with gr.Row():
            defects = gr.Dataframe(headers=["Defect", "Severity", "Location", "Confidence"],
                                   label="Defects", wrap=True)
            advice = gr.Markdown(label="Operator advice")
        run.click(inspect, [image, source, stage_in], [result, verdict, defects, advice, timing, stats, log_file])
    return app


def main():
    ap = argparse.ArgumentParser(description="TraceIQ operator dashboard")
    ap.add_argument("--pi", action="store_true", help="use the NCNN models (Raspberry Pi)")
    ap.add_argument("--backend", choices=["template", "native", "inproc", "llama_main"], default="template")
    ap.add_argument("--config", default=str(ROOT / "advisor/advisor_config.json"))
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--gpio", action="store_true", help="buzzer GPIO17, red LED GPIO27, green LED GPIO22")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_app(a).queue().launch(server_name="0.0.0.0", server_port=a.port)


if __name__ == "__main__":
    main()
