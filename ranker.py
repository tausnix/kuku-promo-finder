"""DETERMINISTIC: score, de-duplicate and order validated promos.

The LLM gives 1-5 judgements; this module combines them with measured facts (hook speed,
duration, rule signals, episode position) using fixed weights from config.RankWeights,
so the ranking is explainable and identical on every rerun.
"""
from __future__ import annotations

from config import RankWeights


def _overlap_s(a: dict, b: dict) -> float:
    total = 0.0
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["episode"] == sb["episode"]:
                total += max(0.0, min(sa["end_s"], sb["end_s"]) - max(sa["start_s"], sb["start_s"]))
    return total


def overlap_ratio(a: dict, b: dict) -> float:
    shorter = min(a["total_duration_s"], b["total_duration_s"]) or 1.0
    return _overlap_s(a, b) / shorter


def _duration_fit(d: float) -> float:
    if 25 <= d <= 45:
        return 1.0
    if d < 25:
        return max(0.5, 1 - (25 - d) / 10)
    return max(0.5, 1 - (d - 45) / 30)


def _funnel(first_ep: int) -> float:
    # A promo sends viewers to episode 1. Early moments = no spoilers, short path to the payoff.
    return 1.0 if first_ep <= 5 else 0.85 if first_ep <= 15 else 0.7 if first_ep <= 30 else 0.6


def score_promo(p: dict, w: RankWeights) -> dict:
    s = p["llm_scores"]
    parts = {
        "llm_hook": (s["hook"] - 1) / 4,
        "llm_cliffhanger": (s["cliffhanger"] - 1) / 4,
        "llm_standalone": (s["standalone"] - 1) / 4,
        "rule_hook": min(p["rule_signals"]["hook"], 8) / 8,
        "rule_cliff": min(p["rule_signals"]["cliff"], 6) / 6,
        "duration_fit": _duration_fit(p["total_duration_s"]),
        "hook_speed": 1.0 if (p["hook"]["offset_s"] or 0) <= 1.0 else 0.6,
        "funnel": _funnel(min(p["episodes"])),
    }
    total = 100 * sum(getattr(w, k) * v for k, v in parts.items()) / sum(getattr(w, k) for k in parts)
    total -= w.warning_penalty * len(p["warnings"])
    return {"score": round(max(total, 0), 1), "score_breakdown": {k: round(v, 2) for k, v in parts.items()}}


def hook_key(p: dict) -> tuple:
    """How good the opening is: complete line first, then LLM hook score, rule signal, total score."""
    fragment = any(w["code"] == "hook_is_fragment" for w in p["warnings"])
    return (not fragment, p["llm_scores"]["hook"], p["rule_signals"]["hook"], p["score"])


def rank_reason(p: dict) -> str:
    s, b = p["llm_scores"], p["score_breakdown"]
    bits = [f"side-by-side #{p['rerank_pos']:g}" + (f" ({p['rerank_note']})" if p.get("rerank_note") else "")] \
        if p.get("rerank_pos") is not None else []
    bits += [f"hook {s['hook']}/5 ({p['hook_type']}, at +{p['hook']['offset_s']:.1f}s)",
            f"cliffhanger {s['cliffhanger']}/5", f"standalone {s['standalone']}/5",
            f"{p['total_duration_s']:.0f}s"]
    if b["funnel"] >= 1.0:
        bits.append(f"early episode E{min(p['episodes']):02d}, no spoilers")
    if p["warnings"]:
        bits.append(f"{len(p['warnings'])} flag(s): " + ", ".join(w["code"] for w in p["warnings"]))
    return "; ".join(bits)


def rank(promos: list[dict], w: RankWeights, dup_threshold: float = 0.5,
         rerank: dict | None = None) -> list[dict]:
    """rerank: output of reranker.rerank(). Its average positions order the clips it saw;
    everything else follows by score."""
    pos = (rerank or {}).get("avg_pos") or {}
    notes = (rerank or {}).get("notes") or {}
    for p in promos:
        p.update(score_promo(p, w))
        p["duplicate_of"] = None
        p["rerank_pos"] = pos.get(p["promo_id"])
        p["rerank_note"] = notes.get(p["promo_id"], "")
        p.pop("blend_pos", None)

    # De-duplicate: if two promos share >50% of the shorter one's runtime they are the same moment.
    # Keep the one with the stronger HOOK (the first 3 s decide whether anyone watches), then the
    # higher total score. A longer clip with a weaker opening must not push out a better opener.
    ordered = sorted(promos, key=lambda p: (-p["score"], p["promo_id"]))
    kept: list[dict] = []
    for p in ordered:
        clashes = [k for k in kept if overlap_ratio(p, k) > dup_threshold]
        if not clashes:
            kept.append(p)
        elif all(hook_key(p) > hook_key(k) for k in clashes):
            for k in clashes:
                kept.remove(k)
                k["duplicate_of"] = p["promo_id"]
            kept.append(p)
        else:
            p["duplicate_of"] = max(clashes, key=hook_key)["promo_id"]
    kept.sort(key=lambda p: (-p["score"], p["promo_id"]))
    if pos:  # blend: clips the side-by-side call saw are ordered by a mix of both places
        score_place = {p["promo_id"]: i for i, p in enumerate([k for k in kept if k["promo_id"] in pos], 1)}
        bw = w.rerank_weight
        for p in kept:
            if p["promo_id"] in pos:
                p["blend_pos"] = round((1 - bw) * score_place[p["promo_id"]] + bw * pos[p["promo_id"]], 2)
        kept.sort(key=lambda p: (p.get("blend_pos") is None, p.get("blend_pos") or 0, -p["score"], p["promo_id"]))
    # A displaced promo may itself have absorbed duplicates: point those at the final winner.
    by_id = {p["promo_id"]: p for p in ordered}
    for p in ordered:
        seen = set()
        while p["duplicate_of"] and by_id[p["duplicate_of"]]["duplicate_of"] and p["duplicate_of"] not in seen:
            seen.add(p["duplicate_of"])
            p["duplicate_of"] = by_id[p["duplicate_of"]]["duplicate_of"]

    # "Post first" list: best scores, but at most N per episode so the top slots aren't one scene.
    top, rest, per_ep = [], [], {}
    for p in kept:
        ep = min(p["episodes"])
        if len(top) < w.top_n and per_ep.get(ep, 0) < w.max_per_episode_in_top:
            top.append(p)
            per_ep[ep] = per_ep.get(ep, 0) + 1
        else:
            rest.append(p)
    final = top + rest
    for i, p in enumerate(final, 1):
        p["rank"] = i
        p["rank_reason"] = rank_reason(p)
        p["review_status"] = "needs_review" if (p["warnings"] or p["llm_scores"]["standalone"] <= 2) else "ready_for_review"
    dups = [p for p in ordered if p["duplicate_of"]]
    for p in dups:
        p["rank"] = None
        p["rank_reason"] = f"duplicate of {p['duplicate_of']}"
        p["review_status"] = "duplicate"
    return final + dups
