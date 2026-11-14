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

import argparse, json, os, re, shlex, subprocess, sys, time
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
    raw_llm: str = ""


def region_name(cx: float, cy: float, w: int, h: int) -> str:
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "centre"
    row = "upper" if cy < h / 3 else "lower" if cy > 2 * h / 3 else "middle"
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"


def make_record(cls: str, conf: float, xyxy, img_w: int, img_h: int,
                area_frac: float | None = LARGE_AREA_FRAC) -> DefectRecord:
    """area_frac: raise severity one level if the box is larger than this fraction of the image.
    None disables it (Stage 2 close-up photos, where every defect fills a large part of the image)."""
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    base = SEVERITY_RANK[KNOWLEDGE[cls]["severity"]]
    if area_frac is not None and area > area_frac * img_w * img_h:
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
SYSTEM = ("You are a PCB quality-control assistant. You explain inspection results to a factory operator "
          "in plain English. Talk only about the defects listed. Reply with exactly three short lines that start "
          "with 'Problem:', 'Cause:' and 'Action:', covering all listed defects.")

EXAMPLE_USER_1 = ("Inspection result for board EX-1: FAIL.\nDefects found:\n- Open circuit (critical) at lower-left.\n"
                "- Spur x2 (minor) at upper-right.\nReference: Open circuit: likely cause is over-etching through a "
                "narrow part of the trace; action: confirm with a continuity test across the break. Spur: likely cause "
                "is dust or a defect on the phototool; action: measure the spacing between the spur and the nearest "
                "conductor.\nExplain this to the operator.")
EXAMPLE_ASSISTANT_1 = ("Problem: An open circuit at the lower-left and two small spurs at the upper-right.\n"
                     "Cause: The trace was likely over-etched at a narrow point; the spurs likely come from dust on the "
                     "phototool.\nAction: Run a continuity test and reject the board if the trace is broken; measure "
                     "the spur spacing and clean the phototool.")

# Stage 2 (assembled board) one-shot examples. A 1B model copies its example closely, so each example has a
# single defect, and the one used is never a type that was detected (nothing extra to copy into the answer).
EXAMPLES_2 = {
    "solder_void": (
        "Inspection result for board EX-2: REVIEW.\nDefects found:\n- Solder void (minor) at lower-right.\n"
        "Reference: Solder void: likely cause is flux or moisture outgassing during reflow; action: check the void "
        "size against the limit and rework large voids.\nExplain this to the operator.",
        "Problem: A solder void at the lower-right.\n"
        "Cause: Gas from flux or moisture was trapped in the joint during reflow.\n"
        "Action: Check the void size against the limit and rework the joint if it is too large."),
    "burn": (
        "Inspection result for board EX-3: FAIL.\nDefects found:\n- Burn mark (critical) at lower-right.\n"
        "Reference: Burn mark: likely cause is a soldering iron held too long or too hot; action: reject the "
        "board and check for lifted pads and damaged parts.\nExplain this to the operator.",
        "Problem: A burn mark at the lower-right.\n"
        "Cause: The soldering iron was likely held too long or was too hot.\n"
        "Action: Reject the board and check the area for lifted pads and damaged parts."),
}
STAGE2_CLASSES = {"insufficient_solder", "excess_solder", "solder_bridge", "solder_void", "missing_component",
                  "shifted_component", "burn", "physical_damage", "oxidation"}


def example_for(insp: "Inspection") -> tuple[str, str]:
    kinds = {d.defect_class for d in insp.defects}
    if kinds & STAGE2_CLASSES:
        return EXAMPLES_2["burn" if "solder_void" in kinds else "solder_void"]
    return EXAMPLE_USER_1, EXAMPLE_ASSISTANT_1


MAX_PROMPT_TYPES = 3  # keep the prompt under the exported 512-token prefill limit


def prompt_kinds(insp: Inspection) -> list[str]:
    """Defect types given to Llama: the most severe (then most frequent) first, at most MAX_PROMPT_TYPES."""
    kinds = list(dict.fromkeys(d.defect_class for d in insp.defects))
    def rank(k):
        ds = [d for d in insp.defects if d.defect_class == k]
        return (max(SEVERITY_RANK[d.severity] for d in ds), len(ds))
    return sorted(kinds, key=rank, reverse=True)[:MAX_PROMPT_TYPES]


def defect_summary(insp: Inspection, kinds: list[str] | None = None) -> list[str]:
    """One plain-English line per defect type, e.g. 'Mouse bite x2 (minor) at lower-left and upper-right'."""
    lines = []
    for k in (kinds or dict.fromkeys(d.defect_class for d in insp.defects)):
        ds = [d for d in insp.defects if d.defect_class == k]
        sev = RANK_NAME[max(SEVERITY_RANK[d.severity] for d in ds)]
        where = " and ".join(dict.fromkeys(d.region for d in ds))
        n = f" x{len(ds)}" if len(ds) > 1 else ""
        lines.append(f"{KNOWLEDGE[k]['label']}{n} ({sev}) at {where}")
    return lines


def build_prompt(insp: Inspection) -> str:
    kinds = prompt_kinds(insp)
    found = "\n".join(f"- {l}." for l in defect_summary(insp, kinds))
    ref = " ".join(f"{KNOWLEDGE[k]['label']}: likely cause is {KNOWLEDGE[k]['causes'][0].lower()}; "
                   f"action: {KNOWLEDGE[k]['actions'][0].lower()}." for k in kinds)
    user = (f"Inspection result for board {insp.board_id}: {insp.verdict}.\n"
            f"Defects found:\n{found}\n"
            f"Reference: {ref}\n"
            "Explain this to the operator.")
    ex_user, ex_assistant = example_for(insp)
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{ex_user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ex_assistant}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "Problem:")  # prefill: the model continues in the required format


def max_len_for(prompt: str, new_tokens: int = 160) -> int:
    return min(1000, len(prompt) // 3 + new_tokens)


# ---------------------------------------------------------------- LLM backends
_LOG_LINE = re.compile(r"^(I |W |E )?\s*(tokenizers:|Using Tiktokenizer|I \d\d:|PyTorchObserver)")


_REPEAT = re.compile(r"(\b[\w-]+(?:\s[\w-]+){0,3}\b)(?:[,;]?\s+\1\b){2,}", re.I)


_SENT = re.compile(r"(?<=[.!?])\s+")


def trim_line(line: str, max_sent: int = 2) -> str:
    """Keep the first sentences of one Problem/Cause/Action line and stop at the first sign of looping:
    a repeated sentence, a sentence that restates an earlier one ('again', 'repeat'), or an unfinished
    last sentence cut off by the token limit."""
    head, _, body = line.partition(":")
    sents = [x.strip() for x in _SENT.split(body.strip()) if x.strip()]
    keep, seen = [], set()
    for x in sents:
        norm = " ".join(re.findall(r"[a-z]+", x.lower())[:5])
        if norm in seen or re.search(r"\b(again|repeat)\b", x, re.I) and keep:
            break
        seen.add(norm)
        keep.append(x)
        if len(keep) == max_sent:
            break
    if len(keep) > 1 and not re.search(r"[.!?]$", keep[-1]):
        keep.pop()  # unfinished sentence at the end
    out = " ".join(keep)
    if out and not re.search(r"[.!?]$", out):
        out += "."
    return f"{head}: {out}" if out else line


def clean_output(raw: str) -> str:
    text = "\n".join(l for l in raw.splitlines() if not _LOG_LINE.match(l.strip()))
    if "assistant<|end_header_id|>" in text:
        text = text.rsplit("assistant<|end_header_id|>", 1)[1]
    for stop in ("<|eot_id|>", "<|", "Prefill time", "PyTorchObserver", "Response: [",
                 "Write for the operator", "Explain this to the operator"):
        if stop in text:
            text = text.split(stop, 1)[0]
    text = text.strip()
    if not text.lower().startswith("problem:"):   # the prompt pre-fills "Problem:"
        text = "Problem: " + text
    m = _REPEAT.search(text)                       # cut runaway loops like "Under-etching, Under-etching, ..."
    if m:
        text = text[:m.start()] + m.group(1)
    keep, seen = {}, []
    for line in text.splitlines():                 # first Problem / Cause / Action line of each kind
        key = line.strip().split(":", 1)[0].lower()
        if key in ("problem", "cause", "action") and key not in keep:
            keep[key] = trim_line(line.strip())
    if len(keep) == 3:
        return "\n".join(keep[k] for k in ("problem", "cause", "action"))
    return text


LAST_RAW = {"stdout": "", "stderr": ""}


def run_llama_main(prompt: str, cfg: dict) -> str:
    cmd = [cfg["llama_main"], "--model_path", cfg["pte"], "--tokenizer_path", cfg["tokenizer"],
           "--seq_len", str(cfg.get("seq_len", 1024)), "--temperature", "0", "--prompt", prompt]
    env = dict(os.environ, LD_LIBRARY_PATH=f"{Path(cfg['llama_main']).parent}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.get("timeout", 180), env=env)
    LAST_RAW.update(stdout=out.stdout, stderr=out.stderr)
    return clean_output(out.stdout)


def run_native(prompt: str, cfg: dict) -> str:
    cmd = ["python", "-m", "examples.models.llama.runner.native", "--model", "llama3_2",
           "--pte", cfg["pte"], "--params", cfg["params"], "--tokenizer", cfg["tokenizer"],
           "--kv_cache", "--temperature", "0", "--max_len", str(max_len_for(prompt)), "--prompt", prompt]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.get("timeout", 300),
                         cwd=cfg["executorch_dir"])
    LAST_RAW.update(stdout=out.stdout, stderr=out.stderr)
    return clean_output(out.stdout)


class InprocLlama:
    """Keeps Llama loaded in this Python process (ExecuTorch pybindings), so only the first board pays
    the model-load cost. Uses ExecuTorch's NativeLlamaRunner from the executorch source tree."""
    _instance = None

    def __init__(self, cfg: dict):
        import contextlib, io, sys as _sys
        from types import SimpleNamespace
        _sys.path.insert(0, cfg["executorch_dir"])
        with contextlib.redirect_stdout(io.StringIO()):
            from examples.models.llama.runner.native import NativeLlamaRunner
            t0 = time.perf_counter()
            self.runner = NativeLlamaRunner(SimpleNamespace(
                params=cfg["params"], tokenizer=cfg["tokenizer"], tokenizer_config=None,
                max_len=cfg.get("max_len", 1000), kv_cache=True, pte=cfg["pte"]))
        self.load_s = round(time.perf_counter() - t0, 2)
        self.max_len = cfg.get("max_len", 1000)

    @classmethod
    def get(cls, cfg: dict) -> "InprocLlama":
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    def __call__(self, prompt: str, new_tokens: int = 160) -> str:
        import contextlib, io
        tok = self.runner.tokenizer
        ids = tok.encode(prompt, bos=True, eos=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):  # the runner prints tokens as it goes
            out = self.runner.generate(prompt_tokens=ids, max_seq_len=min(self.max_len, len(ids) + new_tokens),
                                       temperature=0.0, top_p=0.9)
        LAST_RAW.update(stdout=buf.getvalue(), stderr="")
        stops = {tok.eos_id, *getattr(tok, "stop_tokens", [])}
        keep = [t for t in out if t not in stops]
        try:
            text = tok.decode(keep)
        except Exception:  # noqa: BLE001  older tokenizer API
            text = "".join(tok.decode_token(t) for t in keep)
        return clean_output(text)


def run_inproc(prompt: str, cfg: dict) -> str:
    return InprocLlama.get(cfg)(prompt)


def template_advice(insp: Inspection) -> str:
    if not insp.defects:
        return "Problem: no defects found.\nCause: -\nAction: pass the board to the next step."
    top = max(insp.defects, key=lambda d: (SEVERITY_RANK[d.severity], d.confidence)).defect_class
    return (f"Problem: {'; '.join(defect_summary(insp))}.\n"
            f"Cause: {KNOWLEDGE[top]['causes'][0]}.\n"
            f"Action: {KNOWLEDGE[top]['actions'][0]}.")


def log_line_for(insp: Inspection) -> str:
    if not insp.defects:
        return f"{insp.board_id} PASS: no defects"
    return f"{insp.board_id} {insp.verdict}: " + ", ".join(defect_summary(insp))


# ---------------------------------------------------------------- guard check
PATTERNS = {
    "missing_hole": r"missing holes?|holes? (?:is |are )?missing|no hole",
    "mouse_bite": r"mouse[- ]bites?",
    "open_circuit": r"open[- ]circuits?|broken trace|trace is broken|open trace",
    "short": r"short[- ]circuits?|\bshorts\b|\ba short\b(?! (?:line|time|while|distance|lead))|bridg",
    "spur": r"\bspurs?\b",
    "spurious_copper": r"spurious copper|copper island|isolated copper",
    "hole_breakout": r"breakout|break-out|annular ring",
    "conductor_scratch": r"scratch",
    "conductor_foreign_object": r"foreign (?:object|material|particle)s?|debris",
    "base_material_foreign_object": r"foreign (?:object|material|particle)s?|debris|base material|laminate",
    # Stage 2 (assembled board)
    "insufficient_solder": r"insufficient solder|(?:too little|not enough|lack of) solder(?! paste)|starved",
    "excess_solder": r"excess(?:ive)? solder(?! paste)|too much solder(?! paste)|extra solder(?! paste)",
    "solder_bridge": r"solder bridg\w*|\bbridg\w*|short[- ]circuits?|\bshorts?\b",
    "solder_void": r"\bvoids?\b|gas pockets?|\bbubbles?\b",
    "missing_component": r"missing (?:component|part)s?|(?:component|part)s? (?:is |are )?missing|\babsent\b",
    "shifted_component": r"\bshifted\b|\bdisplaced\b|misaligned (?:component|part)|(?:component|part)s? (?:is |are )?"
                         r"(?:misaligned|out of position|off[- ]position)|tombston\w*|\bskew\w*|realign\w*|"
                         r"placed with an offset|placement offset",
    "burn": r"\bburn(?:t|ed|s|ing)?\b|burn marks?|scorch\w*|char(?:red|ring)\b",
    "physical_damage": r"physical(?:ly)? damage|\bcrack(?:ed|s)?\b|\bchipped\b|broken (?:part|component)",
    "oxidation": r"oxidi[sz]\w*|oxidation|corro\w*|tarnish\w*",
}
DENIAL = r"\b(?:not present|no defects?|nothing wrong|is fine|are fine|not a defect|no issues?)\b"


def drop_unsupported(advice: str, insp: Inspection) -> str:
    """Remove clauses in the Cause and Action lines that talk about a defect type that was not detected
    (e.g. '...; realign the part' when no shifted part was found). The Problem line is never edited:
    if it is wrong, the guard rejects the whole answer."""
    found = set(prompt_kinds(insp))
    others = [k for k in PATTERNS if k not in found]

    def unsupported(clause: str) -> bool:
        low = clause.lower()
        return any(re.search(PATTERNS[k], low) and not any(_shared(k, f, low) for f in found) for k in others)

    out = []
    for line in advice.splitlines():
        head, sep, body = line.partition(":")
        if sep and head.strip().lower() in ("cause", "action"):
            parts = re.split(r"(?<=[;.])\s+", body.strip())
            kept = [p for p in parts if not unsupported(p)]
            if kept and len(kept) < len(parts):
                text = " ".join(kept).rstrip(";").strip()
                line = f"{head}: {text if re.search(r'[.!?]$', text) else text + '.'}"
        out.append(line)
    return "\n".join(out)


def guard(advice: str, insp: Inspection) -> tuple[bool, str]:
    """Format OK, every detected type mentioned, no undetected type mentioned, no denial of a defect."""
    if len(advice) < 20:
        return False, "empty or too short"
    if not all(re.search(rf"(?im)^{k}:", advice) for k in ("problem", "cause", "action")):
        return False, "missing Problem/Cause/Action lines"
    low = advice.lower()
    found = set(prompt_kinds(insp))  # Llama is only told about these
    for k in found:
        if not re.search(PATTERNS[k], low):
            return False, f"does not mention detected: {k}"
    for k, pat in PATTERNS.items():
        if k in found or not re.search(pat, low):
            continue
        # phrases shared by two classes (e.g. "foreign object") are fine if the other class was detected
        if any(re.search(PATTERNS[f], low) and re.search(pat, low) and _shared(k, f, low) for f in found):
            continue
        return False, f"mentions undetected defect: {k}"
    if re.search(DENIAL, low):
        return False, "denies a detected defect"
    # locations in the Problem line must be ones where a defect was actually found
    allowed = {d.region for d in insp.defects if d.defect_class in found}
    problem = re.search(r"(?im)^problem:(.*)$", advice).group(1).lower()
    said = {f"{r}-{c}".replace("center", "centre").replace("middle-centre", "centre")
            for r, c in re.findall(_REGION_2, problem)}
    for m in re.finditer(r"\bcent(?:re|er)\b", problem):  # plain "centre" (not part of "upper-centre")
        if not re.search(r"(upper|middle|lower)[- ]$", problem[:m.start()]):
            said.add("centre")
    extra = said - allowed
    if extra:
        return False, f"mentions a location with no defect: {', '.join(sorted(extra))}"
    return True, "ok"


_REGION_2 = r"\b(upper|middle|lower)[- ](left|centre|center|right)\b"


def _shared(a: str, b: str, text: str) -> bool:
    """True if every match of class a's pattern is also a match of class b's pattern."""
    return all(re.search(PATTERNS[b], m.group(0)) for m in re.finditer(PATTERNS[a], text))


def advise(insp: Inspection, backend: str = "template", cfg: dict | None = None) -> Inspection:
    t0 = time.perf_counter()
    if backend == "template" or not insp.defects:
        text, ok = template_advice(insp), True
        used = "template"
    else:
        try:
            fn = {"llama_main": run_llama_main, "native": run_native, "inproc": run_inproc}[backend]
            text = fn(build_prompt(insp), cfg or {})
        except Exception as e:  # noqa: BLE001
            text = f"[advisor error: {type(e).__name__}: {e}]"
            print(f"ADVISOR ERROR ({backend}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        insp.raw_llm = text
        if not text.startswith("[advisor error"):
            text = drop_unsupported(text, insp)
        ok, why = guard(text, insp)
        if text.startswith("[advisor error"):
            why = text.strip("[]")
        used = backend
        if not ok:
            text = template_advice(insp) + f"\n(note: LLM advice rejected by guard: {why})"
            used = f"{backend}->template"
    insp.advice, insp.guard_passed, insp.advisor_backend = text, ok, used
    insp.advisor_seconds = round(time.perf_counter() - t0, 2)
    insp.log_line = log_line_for(insp)
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
    ap.add_argument("--backend", choices=["template", "native", "inproc", "llama_main"], default="template")
    ap.add_argument("--config", default=str(HERE / "advisor_config.json"))
    ap.add_argument("--debug", action="store_true", help="print the LLM's raw stdout/stderr")
    a = ap.parse_args()
    cfg = {}
    if a.backend != "template":
        cfg = {k: os.path.expanduser(v) if isinstance(v, str) else v
               for k, v in json.loads(Path(a.config).read_text())[
                   "native" if a.backend == "inproc" else a.backend].items()}
    insp = advise(demo_inspection(), a.backend, cfg)
    print("PROMPT:\n" + build_prompt(insp) + "\n")
    print(json.dumps({**asdict(insp), "defects": [asdict(d) for d in insp.defects]}, indent=2))
    print(f"\nprompt length: {len(build_prompt(insp))} chars (~{len(build_prompt(insp)) // 4} tokens)")
    if a.debug:
        print("\n----- RAW STDOUT (last 40 lines) -----\n" + "\n".join(LAST_RAW["stdout"].splitlines()[-40:]))
        print("\n----- RAW STDERR (last 25 lines) -----\n" + "\n".join(LAST_RAW["stderr"].splitlines()[-25:]))


if __name__ == "__main__":
    main()
