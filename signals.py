"""DETERMINISTIC: keyword/punctuation signals for hooks and cliffhangers.

Used three ways:
  1. hints in the LLM prompt ("rule-flagged moments")
  2. part of the final ranking score (a second opinion next to the LLM's)
  3. the offline `mock` provider: a pure-rules baseline that needs no API key
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (regex, weight, label). Stems keep this short: "slap" matches slap/slaps/slapped.
HOOK_TERMS: list[tuple[str, float, str]] = [
    (r"\bslap\w*", 4, "slap"), (r"\bpunch\w*", 3, "violence"), (r"\bstrik\w*|\bstruck\b", 3, "violence"),
    (r"\bshov\w*|\bpush(es|ed)?\b", 2, "violence"), (r"\bgrab\w*|\bseiz\w*|\byank\w*", 2, "physical"),
    (r"\bthrow\w*|\bhurl\w*|\bthrew\b", 2, "physical"), (r"\bslam\w*|\bshatter\w*|\bsmash\w*", 2, "physical"),
    (r"\bkiss\w*", 2, "romance"), (r"\bblood\w*", 2, "violence"), (r"\bknife|\bdagger|\bsword|\bpistol|\bgun\b|\bblade", 3, "weapon"),
    (r"\bcollaps\w*|\bfaint\w*", 2, "physical"), (r"\bscream\w*|\bshriek\w*", 2, "shock"),
    (r"\bpregnan\w*", 3, "reveal"), (r"\bdivorc\w*", 3, "reveal"), (r"\bmurder\w*|\bkill\w*|\bpoison\w*", 3, "threat"),
    (r"\bdead\b|\bdie\b|\bdeath\b", 2, "threat"), (r"\bsecret\w*", 2, "reveal"), (r"\bliar\b|\blied\b|\blie to\b", 2, "accusation"),
    (r"\bhow dare\b", 3, "confrontation"), (r"\bget out\b", 2, "confrontation"), (r"\bkneel\w*|\bon your knees\b", 3, "humiliation"),
    (r"\bbetray\w*", 2, "betrayal"), (r"\bduke\b|\bduchess\b|\bheir\b", 2, "reveal"), (r"\bmarr(y|ying|ied|iage)\b", 1, "romance"),
    (r"\bworthless\b|\bplain\w*\b|\bpathetic\b|\bdisgrac\w*|\bhumiliat\w*", 2, "humiliation"),
    (r"\bnever\b.*\bagain\b|\bthrow you out\b|\bdisown\w*", 2, "confrontation"),
]
EMOTION_TERMS: list[tuple[str, float, str]] = [
    (r"furious|enraged|outraged|livid", 2, "rage"), (r"shock\w*|horrified|stunned|aghast", 2, "shock"),
    (r"scream\w*|shout\w*|yell\w*|roar\w*|crying out", 2, "shouting"),
    (r"menac\w*|threaten\w*|venom\w*|cruel|vicious", 2, "menace"),
    (r"gloat\w*|smug|mocking|contempt\w*|scorn\w*|disdain\w*", 1, "contempt"),
    (r"desperate|sobbing|devastated|broken", 1, "anguish"),
]
CLIFF_TERMS: list[tuple[str, float, str]] = [
    (r"cliffhanger", 4, "scripted cliffhanger"), (r"cut to black|smash cut|freeze[s]? frame", 3, "hard cut"),
    (r"ominous|dread|foreboding", 2, "ominous"), (r"drone|swell\w*|sting\b|thud|boom", 1, "sound sting"),
    (r"\bwho are you\b|\bwhat did you\b|\bwhat have you\b|\bwhat is this\b", 3, "open question"),
    (r"\bfroze\b|\bfreez\w*|\bstunned\b|\bgasp\w*", 2, "shock beat"),
    (r"\bdoor\w* (burst|swing|fly|flies|slam)\w*|\bbursts? in\b|\barriv\w*", 2, "interruption"),
    (r"\byou will\b|\byou'll regret\b|\bthis isn't over\b|\bwait\b", 2, "threat/turn"),
]
_HOOK = [(re.compile(p, re.I), w, l) for p, w, l in HOOK_TERMS]
_EMO = [(re.compile(p, re.I), w, l) for p, w, l in EMOTION_TERMS]
_CLIFF = [(re.compile(p, re.I), w, l) for p, w, l in CLIFF_TERMS]


@dataclass
class Signal:
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    label: str = "other"


# Not "that": a line like "That is a lie." stands on its own (it was a false positive on the first full run).
_FRAGMENT_STARTERS = {"and", "but", "because", "so", "or", "then", "which", "than", "nor", "yet"}


def is_fragment(text: str) -> bool:
    """True if a line can't open a promo: it continues or answers something the viewer didn't hear.
    E.g. '...and then he left town.', 'Because she lied to you.', 'Worry? You fell.'"""
    t = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", (text or "").strip().strip('"“”').strip())
    if not t:
        return False
    if t.startswith(("...", "…", "-", "—")) or t[0].islower():
        return True
    first = re.split(r"[\s,?!.]+", t, maxsplit=1)[0].lower()
    if first in _FRAGMENT_STARTERS:
        return True
    # one-word echo question answering someone else: "Worry? ...", "Dignity?", "Embarrassed you?"
    return bool(re.match(r"^[A-Za-z']+( [a-z']+)?\?", t))


def hook_signal(block) -> Signal:
    sig = Signal()
    text = block.text or ""
    labels: dict[str, float] = {}
    for rx, w, label in _HOOK:
        if rx.search(text):
            sig.score += w
            sig.reasons.append(label)
            labels[label] = labels.get(label, 0) + w
    for rx, w, label in _EMO:
        if rx.search(block.direction or ""):
            sig.score += w
            sig.reasons.append(f"delivery:{label}")
    if block.kind == "DIALOGUE":
        if "!" in text:
            sig.score += 1
            sig.reasons.append("exclamation")
        if len(text.split()) <= 9:  # punchy short line
            sig.score += 0.5
    if block.kind == "ACTION" and labels:
        sig.score += 1  # a visible physical beat is the strongest 3-second stopper
    if labels:
        sig.label = max(labels, key=labels.get)
    elif block.kind == "DIALOGUE" and sig.score >= 2:
        sig.label = "shocking_line"
    return sig


def cliff_signal(block, episode_duration: float | None = None) -> Signal:
    sig = Signal()
    text, direction = block.text or "", block.direction or ""
    for rx, w, label in _CLIFF:
        if rx.search(text) or rx.search(direction):
            sig.score += w
            sig.reasons.append(label)
    stripped = text.rstrip('" ”')
    if block.kind == "DIALOGUE":
        if stripped.endswith("?"):
            sig.score += 1.5
            sig.reasons.append("unanswered question")
        if stripped.endswith(("...", "…", "-", "—")):
            sig.score += 1
            sig.reasons.append("trailing line")
    if episode_duration and block.end >= episode_duration - 6:
        sig.score += 2
        sig.reasons.append("episode ending")
    return sig


def _quote(text: str, max_words: int = 12) -> str:
    words = text.strip().strip('"“”').split()
    return " ".join(words[:max_words])


def _scale(x: float, lo: float = 0, hi: float = 8) -> int:
    return int(max(1, min(5, round(1 + 4 * (x - lo) / (hi - lo)))))


def heuristic_promos(blocks: list, episode_duration: float, max_promos: int = 3,
                     min_dur: float = 20, max_dur: float = 60) -> list[dict]:
    """Pure-rules baseline. Returns promos in the same JSON schema the LLM must return."""
    content = [b for b in blocks if b.kind != "CAMERA"]
    scored = sorted(((hook_signal(b), b) for b in content), key=lambda x: -x[0].score)
    promos: list[dict] = []
    used: list[tuple[float, float]] = []
    for hs, h in scored:
        if len(promos) >= max_promos or hs.score < 2:
            break
        if any(a - 5 <= h.start <= b + 5 for a, b in used):
            continue
        if h.kind == "DIALOGUE" and is_fragment(h.text):
            continue  # a reply or half-sentence can't open a promo for a new viewer
        # Keep a margin inside the hard limits: cut points get snapped to camera shots later.
        ends = [c for c in content if c.idx > h.idx and min_dur + 2 <= c.end - h.start <= max_dur - 4]
        if not ends:
            continue
        best = max(ends, key=lambda c: (cliff_signal(c, episode_duration).score, -abs((c.end - h.start) - 40)))
        cs = cliff_signal(best, episode_duration)
        used.append((h.start, best.end))
        promos.append({
            "title": f"Rule pick: {hs.label} at {h.id}",
            "segments": [{"start_id": h.id, "end_id": best.id}],
            "hook_id": h.id, "hook_quote": _quote(h.text),
            "cliffhanger_id": best.id, "cliffhanger_quote": _quote(best.text),
            "hook_type": hs.label if hs.label in HOOK_TYPES else "other",
            "why": f"Rules only: hook signals [{', '.join(hs.reasons[:4])}], cliff signals [{', '.join(cs.reasons[:3]) or 'none'}].",
            "scores": {"hook": _scale(hs.score), "cliffhanger": _scale(cs.score), "standalone": 3},
            "context_needed": "",
        })
    return promos


HOOK_TYPES = {"slap", "violence", "physical", "weapon", "reveal", "shocking_line", "confrontation",
              "humiliation", "betrayal", "threat", "romance", "accusation", "shock", "other"}
