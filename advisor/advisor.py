#!/usr/bin/env python3
"""
TraceIQ advisor: turns vision detections into structured defect records, a deterministic
PASS / REVIEW / FAIL verdict, and plain-language operator advice from Llama 3.2 1B (ExecuTorch).

Design (from the PS5 mentoring video): Llama gets structured findings, never raw pixels.
The verdict is decided by rules in code; Llama only explains and advises. A guard check makes
sure the advice mentions only defects that were actually detected; otherwise a template is used.

Backends:
  llama_main  - the ExecuTorch C++ runner (Raspberry Pi 5)
  native      - ExecuTorch Python runner (laptop testing)
  template    - no LLM, deterministic text (fallback / quick tests)
"""
from __future__ import annotations

import argparse, json, os, re, shlex, subprocess, time
from dataclasses import dataclass, asdict, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWLEDGE = json.loads((HERE / "defect_knowledge.json").read_text())
SEVERITY_RANK = {"minor": 1, "major": 2, "critical": 3}
RANK_NAME = {v: k for k, v in SEVERITY_RANK.items()}
LARGE_AREA_FRAC = 0.002  # defect box > 0.2% of the board image -> raise severity one level


# ---------------------------------------------------------------- records & verdict
@dataclass
class DefectRecord:
    defect_class: str
    confidence: float
    bbox_px: list[int]
    region: str
    area_px: int
    severity: str

    def to_prompt(self) -> dict:
        return {"type": self.defect_class, "severity": self.severity,
                "location": self.region, "confidence": round(self.confidence, 2)}


@dataclass
class Inspection:
    board_id: str
    timestamp: str
    verdict: str
    defects: list[DefectRecord] = field(default_factory=list)
    advice: str = ""
    log_line: str = ""
    advisor_backend: str = ""
    advisor_seconds: float = 0.0
    guard_passed: bool = True


def region_name(cx: float, cy: float, w: int, h: int) -> str:
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "centre"
    row = "upper" if cy < h / 3 else "lower" if cy > 2 * h / 3 else "middle"
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"


def make_record(cls: str, conf: float, xyxy, img_w: int, img_h: int) -> DefectRecord:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    base = SEVERITY_RANK[KNOWLEDGE[cls]["severity"]]
    if area > LARGE_AREA_FRAC * img_w * img_h:
        base = min(3, base + 1)
    return DefectRecord(cls, float(conf), [x1, y1, x2, y2],
                        region_name((x1 + x2) / 2, (y1 + y2) / 2, img_w, img_h), area, RANK_NAME[base])


def verdict_for(defects: list[DefectRecord]) -> str:
    """FAIL if any critical or major defect, REVIEW if only minor ones, PASS if none."""
    if not defects:
        return "PASS"
    worst = max(SEVERITY_RANK[d.severity] for d in defects)
    return "FAIL" if worst >= 2 else "REVIEW"


# ---------------------------------------------------------------- prompt
SYSTEM = ("You are a PCB quality-control assistant on a bare-board inspection line. "
          "Use ONLY the findings and reference notes given. Never add defects that are not in the findings. "
          "Be brief and specific.")


def build_prompt(insp: Inspection) -> str:
    kinds = sorted({d.defect_class for d in insp.defects})
    notes = {k: {"what": KNOWLEDGE[k]["what"], "likely_causes": KNOWLEDGE[k]["causes"],
                 "actions": KNOWLEDGE[k]["actions"]} for k in kinds}
    findings = [d.to_prompt() for d in insp.defects]
    user = (f"Board {insp.board_id}. Verdict (fixed by rules): {insp.verdict}.\n"
            f"Findings: {json.dumps(findings)}\n"
            f"Reference notes: {json.dumps(notes)}\n"
            "Write for the operator, under 90 words:\n"
            "1) What is wrong and where (one line per defect type)\n"
            "2) Most likely cause\n"
            "3) What the operator should do now\n"
            "4) LOG: one line, max 20 words")
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


# ---------------------------------------------------------------- LLM backends
_LOG_LINE = re.compile(r"^(I |W |E )?\s*(tokenizers:|Using Tiktokenizer|I \d\d:|PyTorchObserver)")


def clean_output(raw: str) -> str:
    text = "\n".join(l for l in raw.splitlines() if not _LOG_LINE.match(l.strip()))
    if "assistant<|end_header_id|>" in text:
        text = text.rsplit("assistant<|end_header_id|>", 1)[1]
    for stop in ("<|eot_id|>", "<|", "Prefill time", "PyTorchObserver", "Response: ["):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def run_llama_main(prompt: str, cfg: dict) -> str:
    cmd = [cfg["llama_main"], "--model_path", cfg["pte"], "--tokenizer_path", cfg["tokenizer"],
           "--seq_len", str(cfg.get("seq_len", 1024)), "--temperature", "0", "--prompt", prompt]
    env = dict(os.environ, LD_LIBRARY_PATH=f"{Path(cfg['llama_main']).parent}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.get("timeout", 180), env=env)
    return clean_output(out.stdout)


def run_native(prompt: str, cfg: dict) -> str:
    cmd = ["python", "-m", "examples.models.llama.runner.native", "--model", "llama3_2",
           "--pte", cfg["pte"], "--params", cfg["params"], "--tokenizer", cfg["tokenizer"],
           "--kv_cache", "--temperature", "0", "--max_len", str(cfg.get("max_len", 400)), "--prompt", prompt]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.get("timeout", 300),
                         cwd=cfg["executorch_dir"])
    return clean_output(out.stdout)


def template_advice(insp: Inspection) -> str:
    if not insp.defects:
        return f"No defects found. Board {insp.board_id} passes.\nLOG: {insp.board_id} PASS, no defects."
    lines, kinds = [], []
    for d in insp.defects:
        if d.defect_class not in kinds:
            kinds.append(d.defect_class)
    for k in kinds:
        where = ", ".join(sorted({d.region for d in insp.defects if d.defect_class == k}))
        n = sum(d.defect_class == k for d in insp.defects)
        lines.append(f"- {KNOWLEDGE[k]['label']} x{n} ({where})")
    top = max(insp.defects, key=lambda d: (SEVERITY_RANK[d.severity], d.confidence)).defect_class
    lines.append(f"Likely cause: {KNOWLEDGE[top]['causes'][0]}.")
    lines.append(f"Action: {KNOWLEDGE[top]['actions'][0]}.")
    lines.append(f"LOG: {insp.board_id} {insp.verdict}: " + ", ".join(KNOWLEDGE[k]["label"].lower() for k in kinds))
    return "\n".join(lines)


# ---------------------------------------------------------------- guard check
KEYWORDS = {
    "missing_hole": ["missing hole"], "mouse_bite": ["mouse bite", "mouse-bite"],
    "open_circuit": ["open circuit", "open-circuit", "broken trace"], "short": ["short circuit", "short "],
    "spur": ["spur "], "spurious_copper": ["spurious copper"], "hole_breakout": ["breakout", "break-out"],
    "conductor_scratch": ["scratch"], "conductor_foreign_object": ["foreign object on conductor"],
    "base_material_foreign_object": ["foreign object on base", "base material"],
}


def guard(advice: str, insp: Inspection) -> tuple[bool, str]:
    """Advice must be non-empty, mention every detected type, and no undetected type."""
    if len(advice) < 20:
        return False, "empty or too short"
    low = " " + advice.lower().replace("\n", " ") + " "
    found = {d.defect_class for d in insp.defects}
    for k, words in KEYWORDS.items():
        mentioned = any(w in low for w in words)
        if mentioned and k not in found and not (k == "spur" and "spurious" in low):
            return False, f"mentions undetected defect: {k}"
    missing = [k for k in found if not any(w in low for w in KEYWORDS[k])
               and KNOWLEDGE[k]["label"].lower() not in low]
    if missing:
        return False, f"does not mention detected: {missing}"
    return True, "ok"


def advise(insp: Inspection, backend: str = "template", cfg: dict | None = None) -> Inspection:
    t0 = time.perf_counter()
    if backend == "template" or not insp.defects:
        text, ok = template_advice(insp), True
        used = "template"
    else:
        try:
            text = (run_llama_main if backend == "llama_main" else run_native)(build_prompt(insp), cfg or {})
        except Exception as e:  # noqa: BLE001
            text = f"[advisor error: {e}]"
        ok, why = guard(text, insp)
        used = backend
        if not ok:
            text = template_advice(insp) + f"\n(note: LLM advice rejected by guard: {why})"
            used = f"{backend}->template"
    insp.advice, insp.guard_passed, insp.advisor_backend = text, ok, used
    insp.advisor_seconds = round(time.perf_counter() - t0, 2)
    m = re.search(r"LOG:\s*(.+)", text)
    insp.log_line = m.group(1).strip() if m else f"{insp.board_id} {insp.verdict}"
    return insp


# ---------------------------------------------------------------- CLI demo
def demo_inspection() -> Inspection:
    W, H = 3034, 1586
    dets = [("short", 0.91, (1200, 400, 1240, 430)), ("mouse_bite", 0.78, (300, 1300, 330, 1320)),
            ("mouse_bite", 0.66, (2600, 200, 2625, 222))]
    recs = [make_record(c, p, b, W, H) for c, p, b in dets]
    return Inspection("B-0001", time.strftime("%Y-%m-%dT%H:%M:%S"), verdict_for(recs), recs)


def main():
    ap = argparse.ArgumentParser(description="TraceIQ advisor demo")
    ap.add_argument("--backend", choices=["template", "native", "llama_main"], default="template")
    ap.add_argument("--config", default=str(HERE / "advisor_config.json"))
    a = ap.parse_args()
    cfg = {}
    if a.backend != "template":
        cfg = {k: os.path.expanduser(v) if isinstance(v, str) else v
               for k, v in json.loads(Path(a.config).read_text())[a.backend].items()}
    insp = advise(demo_inspection(), a.backend, cfg)
    print("PROMPT:\n" + build_prompt(insp) + "\n")
    print(json.dumps({**asdict(insp), "defects": [asdict(d) for d in insp.defects]}, indent=2))


if __name__ == "__main__":
    main()
