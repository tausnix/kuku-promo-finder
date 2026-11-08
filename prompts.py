"""Prompt templates (versioned) and the line format the LLM sees.

Key design choice: the model sees timestamps so it can judge length, but it must answer with
line IDs only. Python maps IDs back to timestamps, so a hallucinated timestamp cannot reach the output.
"""
from __future__ import annotations

import json
import re

from parser import Episode, Block, fmt_ts
from signals import hook_signal, cliff_signal, HOOK_TYPES

PROMPT_VERSION = "promo-v2"  # v2: hook must not be a reply/fragment

LINE_RE = re.compile(
    r"^(E\d+-\d+) \[(\d+:\d+\.\d+)-(\d+:\d+\.\d+)\] (DIALOGUE|ACTION|SFX|CAMERA)(?: (.+?))?: (.*)$"
)

SYSTEM_PROMPT = """You are a senior promo editor for vertical microdrama apps (ReelShort / DramaBox style).
You read time-coded screenplays and pick the moments that make the best 20-60 second social ads
(Instagram Reels, TikTok, YouTube Shorts) to pull in NEW viewers.

A great promo:
- HOOK: opens on something that stops a scrolling thumb in the first 3 seconds (a slap, a reveal,
  a humiliating or shocking line, a threat, a forbidden kiss).
- CLIFFHANGER: cuts at an unresolved moment so the viewer must download the app to see what happens.
  Never include the resolution.
- STANDALONE: a stranger understands who is against whom within seconds, without earlier episodes.

HARD RULES (a program checks every one; violations are rejected):
1. Refer to script lines ONLY by the IDs shown, e.g. "E07-031". Never invent IDs. Never write timestamps.
2. A promo has 1-4 segments in chronological order. Each segment is a continuous range
   start_id..end_id inside ONE episode. Segments must not overlap.
3. Total runtime of all segments: 20-60 seconds (use the [start-end] times; aim for 30-45 s).
4. The hook line must start within 3 seconds of the first segment's start. Best: start_id = hook_id.
   The hook must make sense alone: never open on a reply or half-sentence ("...and", "Because...",
   "Worry?"). If the strong line is a reply, start on the line it answers.
5. The cliffhanger line must be in the last segment and end within 6 seconds of its end.
   Best: end_id = cliffhanger_id.
6. hook_quote and cliffhanger_quote: copy 3-15 consecutive words EXACTLY from that line's text.
7. Promos from the same episode must not overlap each other in time.
8. Lines under NEXT_EPISODE_OPENING may be used only as a later segment that continues a
   cliffhanger from the current episode.
9. Quality over quantity. If nothing is strong, return {"promos": []}.

Return ONLY a JSON object, no prose, with this schema:
{"promos": [{
  "title": "short internal name",
  "segments": [{"start_id": "E01-005", "end_id": "E01-040"}],
  "hook_id": "E01-005",
  "hook_quote": "exact words from the hook line",
  "cliffhanger_id": "E01-040",
  "cliffhanger_quote": "exact words from the cliffhanger line",
  "hook_type": "one of: """ + ", ".join(sorted(HOOK_TYPES)) + """",
  "why": "one line on why this works as an ad",
  "scores": {"hook": 1-5, "cliffhanger": 1-5, "standalone": 1-5},
  "context_needed": "what a new viewer would be missing, or empty string"
}]}"""


def render_line(b: Block) -> str:
    who = ""
    if b.kind == "DIALOGUE":
        who = f" {b.speaker}" + (f" ({b.direction})" if b.direction else "")
    return f"{b.id} [{fmt_ts(b.start)}-{fmt_ts(b.end)}] {b.kind}{who}: {b.text}"


def render_lines(blocks: list[Block], include_camera: bool = False) -> str:
    return "\n".join(render_line(b) for b in blocks if include_camera or b.is_content)


def rule_hints(ep: Episode, top_k: int = 6) -> str:
    hooks = sorted(((hook_signal(b), b) for b in ep.content_blocks), key=lambda x: -x[0].score)[:top_k]
    cliffs = sorted(((cliff_signal(b, ep.duration), b) for b in ep.content_blocks), key=lambda x: -x[0].score)[:top_k]
    out = ["Possible hooks:"]
    out += [f"- {b.id}: {', '.join(s.reasons[:4])}" for s, b in hooks if s.score >= 2] or ["- (none)"]
    out.append("Possible cliffhangers:")
    out += [f"- {b.id}: {', '.join(s.reasons[:4])}" for s, b in cliffs if s.score >= 2] or ["- (none)"]
    return "\n".join(out)


def build_user_prompt(ep: Episode, next_ep: Episode | None, show_title: str, max_promos: int,
                      include_camera: bool = False, next_context_s: float = 30.0) -> str:
    parts = [
        f"SHOW: {show_title}",
        f"CURRENT_EPISODE: E{ep.number:02d}",
        f"EPISODE_DURATION: {fmt_ts(ep.duration)}",
        f"LOGLINE: {ep.logline}",
        f"CONTINUITY: {ep.continuity}",
        "",
        "RULE_FLAGGED_MOMENTS (keyword scan; hints only, often wrong):",
        rule_hints(ep),
        "",
        "SCRIPT_LINES" + ("" if include_camera else " (camera directions omitted)") + ":",
        render_lines(ep.blocks, include_camera),
    ]
    if next_ep is not None:
        opening = [b for b in next_ep.blocks if b.start < next_context_s]
        parts += ["", f"NEXT_EPISODE_OPENING (E{next_ep.number:02d}, first {int(next_context_s)} s; continuation only):",
                  render_lines(opening, include_camera)]
    parts += ["", f"Return up to {max_promos} promos for E{ep.number:02d} as JSON."]
    return "\n".join(parts)


def build_repair_prompt(original_user: str, previous_answer: str, failures: list[dict]) -> str:
    lines = []
    for f in failures:
        title = (f.get("raw") or {}).get("title", "")
        errs = "; ".join(f"{i['code']}: {i['message']}" for i in f["issues"] if i["severity"] == "error")
        lines.append(f"- promo #{f['index'] + 1} \"{title}\": {errs}")
    return (
        original_user
        + "\n\nYOUR PREVIOUS ANSWER:\n" + previous_answer.strip()[:6000]
        + "\n\nThe validator REJECTED these promos:\n" + "\n".join(lines)
        + "\n\nFix ONLY the rejected promos (adjust start/end IDs, quotes or segments), or drop any that "
          "cannot satisfy the rules. Do not repeat promos that passed. Return JSON with the same schema."
    )


def build_json_fix_prompt(original_user: str, error: str) -> str:
    return original_user + f"\n\nYour previous reply was not valid JSON ({error}). Return ONLY the JSON object."


# ---------------------------------------------------------------- side-by-side ranking (one call per pass)
RERANK_VERSION = "rerank-v1"

RERANK_SYSTEM = """You are head of paid social for a vertical microdrama app. You choose which promo clips
to run as ads for strangers scrolling TikTok / Reels / Shorts who have never seen the show.

You will see candidate clips from one show. Each card shows what plays in the first seconds, how the clip
ends, and its length. Order ALL of them from best ad to worst ad.

Judge, in this order:
1. Would the first 3 seconds stop a scrolling thumb? Loud conflict, a slap, a shocking or humiliating line
   beats calm talk. Quiet scenes of people talking or plotting rank LOW even if the plot point matters.
2. Does it end on an unresolved moment that makes you need the next scene?
3. Can a stranger follow who is against whom?
4. Does it avoid giving away the show's biggest late twists?

Return ONLY JSON: {"order": ["<id>", "<id>", ...every id exactly once...],
                   "notes": {"<id>": "12 words max on why it is placed there"}}"""


def _card(p: dict, index) -> str:
    first, last = p["segments"][0], p["segments"][-1]
    opening = [b for b in index.range(first["start_id"], first["end_id"])
               if b.is_content and b.start < first["start_s"] + 6][:3]
    ending = [b for b in index.range(last["start_id"], last["end_id"]) if b.is_content][-2:]
    fmt = lambda bs: " / ".join(f"{b.speaker or b.kind}: {b.text[:110]}" for b in bs)
    eps = "+".join(f"E{e:02d}" for e in p["episodes"])
    return (f"[{p['promo_id']}] {p['total_duration_s']:.0f}s, {eps}, {len(p['segments'])} segment(s)\n"
            f"  OPENS: {fmt(opening)}\n  ENDS:  {fmt(ending)}")


def build_rerank_prompt(cands: list[dict], show_title: str, index) -> str:
    return "\n\n".join([f"SHOW: {show_title}", f"CANDIDATE_PROMOS ({len(cands)}):"]
                       + [_card(p, index) for p in cands]
                       + [f"Order all {len(cands)} ids from best to worst ad. JSON only."])


def parse_rendered_lines(prompt: str) -> list[dict]:
    """Inverse of render_line, used by the offline mock provider."""
    out = []
    for line in prompt.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            bid, s, e, kind, who, text = m.groups()
            speaker, direction = who, ""
            if who and "(" in who:
                speaker = who.split("(", 1)[0].strip()
                direction = who.split("(", 1)[1].rstrip(")")
            ms, ss = s.split(":")
            me, se = e.split(":")
            out.append({"id": bid, "start": int(ms) * 60 + float(ss), "end": int(me) * 60 + float(se),
                        "kind": kind, "speaker": speaker, "direction": direction, "text": text})
    return out


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
