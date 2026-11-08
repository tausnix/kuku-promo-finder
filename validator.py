"""DETERMINISTIC: checks every LLM-proposed promo against the screenplay.

Nothing the model says is trusted:
  * line IDs must exist (and belong to the current or next episode)
  * timestamps are computed here from the parsed script, then snapped to camera-shot cuts
  * duration, hook position (first 3 s) and cliffhanger position (end) are measured, not asked
  * quotes must appear verbatim in the referenced line
Errors reject the promo (and trigger one repair round). Warnings keep it but send it to human review.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict

from config import Rules
from parser import BlockIndex, Block, Episode, fmt_ts
from signals import hook_signal, cliff_signal, is_fragment, HOOK_TYPES


@dataclass
class Issue:
    code: str
    message: str
    severity: str = "error"  # error | warning

    def to_dict(self) -> dict:
        return asdict(self)


def parse_llm_json(text: str) -> tuple[list[dict] | None, str | None]:
    """Return (promos, error). Tolerates code fences and leading prose; nothing else."""
    if not text or not text.strip():
        return None, "empty response"
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            return None, "no JSON object found"
        try:
            data = json.loads(t[start:end + 1])
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e.msg} at char {e.pos}"
    if isinstance(data, list):
        data = {"promos": data}
    if not isinstance(data, dict) or not isinstance(data.get("promos"), list):
        return None, "JSON must be an object with a 'promos' list"
    return [p for p in data["promos"] if isinstance(p, dict)], None


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace("…", "...")
    s = re.sub(r"[\"*]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,!?;:-'")


def quote_ok(quote: str, block: Block, rules: Rules) -> tuple[bool, str]:
    q = normalize(quote)
    if q and q == normalize(block.text):  # whole line quoted, e.g. "No!" - fine however short
        return True, ""
    if len(q) < rules.min_quote_chars:
        return False, "quote is empty or too short"
    if len(q.split()) > rules.max_quote_words:
        return False, f"quote longer than {rules.max_quote_words} words"
    if q not in normalize(block.text):
        return False, f"quote not found verbatim in {block.id} (script says: {block.text[:120]!r})"
    return True, ""


def _snap_start(t: float, ep: Episode, tol: float) -> float:
    for c in ep.camera_blocks:
        if c.start <= t < c.end and t - c.start <= tol:
            return c.start
    return t


def _snap_end(t: float, ep: Episode, tol: float) -> float:
    for c in ep.camera_blocks:
        if c.start < t <= c.end and c.end - t <= tol:
            return c.end
    return t


def _as_int_score(v) -> int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(round(f)) if 1 <= f <= 5 else None


def validate_promo(raw: dict, index: BlockIndex, allowed_eps: set[int], rules: Rules) -> tuple[dict | None, list[Issue]]:
    issues: list[Issue] = []
    err = lambda c, m: issues.append(Issue(c, m, "error"))
    warn = lambda c, m: issues.append(Issue(c, m, "warning"))

    segs_raw = raw.get("segments")
    if not isinstance(segs_raw, list) or not segs_raw:
        err("no_segments", "promo has no segments")
        return None, issues
    if len(segs_raw) > rules.max_segments:
        err("too_many_segments", f"{len(segs_raw)} segments > {rules.max_segments}")

    segments = []
    for i, s in enumerate(segs_raw):
        sid, eid = str((s or {}).get("start_id", "")).strip(), str((s or {}).get("end_id", "")).strip()
        a, b = index.get(sid), index.get(eid)
        if not a or not b:
            missing = [x for x, blk in ((sid, a), (eid, b)) if not blk]
            err("unknown_line_id", f"segment {i + 1}: ID(s) {missing} do not exist in the script")
            continue
        if a.episode not in allowed_eps or b.episode not in allowed_eps:
            err("id_outside_allowed_episodes", f"segment {i + 1}: uses E{a.episode}/E{b.episode}; allowed {sorted(allowed_eps)}")
            continue
        if a.episode != b.episode:
            err("segment_spans_episodes", f"segment {i + 1}: {sid}..{eid} crosses episodes; split it into two segments")
            continue
        if b.idx < a.idx:
            err("segment_reversed", f"segment {i + 1}: end {eid} comes before start {sid}")
            continue
        ep = index.episodes[a.episode]
        start = _snap_start(a.start, ep, rules.snap_tolerance_s)
        end = _snap_end(max(b.end, a.end), ep, rules.snap_tolerance_s)
        segments.append({"episode": a.episode, "start_id": sid, "end_id": eid,
                         "start_s": start, "end_s": end, "start": fmt_ts(start), "end": fmt_ts(end),
                         "duration_s": round(end - start, 2), "_a": a, "_b": b})

    if not segments or any(i.severity == "error" for i in issues):
        return None, issues

    for prev, nxt in zip(segments, segments[1:]):
        if (nxt["episode"], nxt["start_s"]) < (prev["episode"], prev["end_s"] - 0.01):
            err("segments_out_of_order_or_overlap",
                f"segment {nxt['start_id']}..{nxt['end_id']} starts before the previous one ends")

    total = round(sum(s["duration_s"] for s in segments), 2)
    if total < rules.min_duration_s:
        err("duration_too_short", f"total {total:.1f}s < {rules.min_duration_s:.0f}s - extend end_id or add a segment")
    if total > rules.max_duration_s:
        err("duration_too_long", f"total {total:.1f}s > {rules.max_duration_s:.0f}s - move end_id earlier or start_id later")

    # Hook: must sit in the first segment and start within the hook window.
    first, last = segments[0], segments[-1]
    hook = index.get(str(raw.get("hook_id", "")))
    hook_offset = None
    if not hook:
        err("unknown_hook_id", f"hook_id {raw.get('hook_id')!r} does not exist")
    elif not (hook.episode == first["episode"] and first["_a"].idx <= hook.idx <= first["_b"].idx):
        err("hook_not_in_first_segment", f"hook {hook.id} is not inside the first segment")
    else:
        hook_offset = round(hook.start - first["start_s"], 2)
        if hook_offset > rules.hook_window_s:
            err("hook_too_late", f"hook starts {hook_offset:.1f}s into the promo (> {rules.hook_window_s:.0f}s); start the segment at {hook.id}")
        ok, why = quote_ok(str(raw.get("hook_quote", "")), hook, rules)
        if not ok:
            err("hook_quote_not_verbatim", why)
        if hook.kind == "DIALOGUE" and is_fragment(hook.text):
            prev = [b for b in index.episodes[hook.episode].content_blocks if b.idx < hook.idx]
            hint = f" (previous line is {prev[-1].id}: {prev[-1].text[:50]!r})" if prev else ""
            err("hook_is_fragment", f"hook {hook.id} {hook.text[:50]!r} continues or answers an earlier line, so a new "
                                    f"viewer can't follow it. Set start_id and hook_id to the line it answers{hint}, "
                                    f"or pick another hook")

    cliff = index.get(str(raw.get("cliffhanger_id", "")))
    cliff_gap = None
    if not cliff:
        err("unknown_cliffhanger_id", f"cliffhanger_id {raw.get('cliffhanger_id')!r} does not exist")
    elif not (cliff.episode == last["episode"] and last["_a"].idx <= cliff.idx <= last["_b"].idx):
        err("cliffhanger_not_in_last_segment", f"cliffhanger {cliff.id} is not inside the last segment")
    else:
        cliff_gap = round(last["end_s"] - cliff.end, 2)
        if cliff_gap > rules.cliff_window_s:
            err("cliffhanger_too_early",
                f"clip runs {cliff_gap:.1f}s past the cliffhanger (> {rules.cliff_window_s:.0f}s); end the segment at {cliff.id}")
        ok, why = quote_ok(str(raw.get("cliffhanger_quote", "")), cliff, rules)
        if not ok:
            err("cliffhanger_quote_not_verbatim", why)

    scores_raw = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    scores = {k: _as_int_score(scores_raw.get(k)) for k in ("hook", "cliffhanger", "standalone")}
    if any(v is None for v in scores.values()):
        err("bad_scores", f"scores must be integers 1-5, got {scores_raw}")

    if any(i.severity == "error" for i in issues):
        return None, issues

    # ---- warnings: keep the promo but flag it for a human --------------------------------
    hook_type = str(raw.get("hook_type", "other")).strip().lower()
    if hook_type not in HOOK_TYPES:
        warn("unknown_hook_type", f"hook_type {hook_type!r} normalised to 'other'")
        hook_type = "other"
    hs, cs = hook_signal(hook), cliff_signal(cliff, index.episodes[cliff.episode].duration)
    if hs.score < 1:
        warn("weak_rule_hook", "no keyword/emotion signal on the hook line - check the first 3 s really stop a scroll")
    if scores["standalone"] <= 2:
        warn("low_standalone", f"LLM rates standalone {scores['standalone']}/5")
    context = str(raw.get("context_needed", "") or "").strip()
    if context and context.lower() not in ("none", "n/a", "-"):
        warn("context_needed", context[:200])
    else:
        context = ""
    eps = sorted({s["episode"] for s in segments})
    if len(eps) > 1:
        warn("cross_episode", f"joins episodes {eps}")
    if len(segments) > 1:
        warn("multi_segment", f"{len(segments)} segments - needs an editor to cut cleanly")
    why = str(raw.get("why", "")).strip()
    if not why:
        warn("missing_why", "LLM gave no reason")
    elif len(why.split()) > 40:
        warn("why_too_long", "reason longer than one line")

    def line(b: Block, quote: str, offset_key: str, offset_val):
        return {"line_id": b.id, "episode": b.episode, "timestamp": fmt_ts(b.start),
                "speaker": b.speaker or b.kind, "script_text": b.text, "llm_quote": quote, offset_key: offset_val}

    promo = {
        "title": str(raw.get("title", "")).strip()[:80],
        "episodes": eps,
        "segments": [{k: v for k, v in s.items() if not k.startswith("_")} for s in segments],
        "total_duration_s": total,
        "hook": line(hook, str(raw.get("hook_quote", "")), "offset_s", hook_offset),
        "cliffhanger": line(cliff, str(raw.get("cliffhanger_quote", "")), "gap_to_end_s", cliff_gap),
        "hook_type": hook_type,
        "why": why,
        "context_needed": context,
        "llm_scores": scores,
        "rule_signals": {"hook": hs.score, "hook_reasons": hs.reasons, "cliff": cs.score, "cliff_reasons": cs.reasons},
        "warnings": [i.to_dict() for i in issues if i.severity == "warning"],
    }
    return promo, issues
