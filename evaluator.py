"""Quality report for a run, plus optional comparison against an editor's picks.

    python evaluator.py --run outputs/gemini_run                    # rebuild the report
    python evaluator.py --run outputs/gemini_run --gold gold.csv    # precision/recall vs editor picks

gold.csv columns: episode,start,end   (e.g. 1,00:11.30,00:45.00)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from signals import hook_signal


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def possible_misses(episodes, kept: list[dict], min_signal: float = 5.0, limit: int = 12) -> list[tuple]:
    """Lines the keyword scan rates as very strong hooks that no accepted promo covers."""
    covered = [(s["episode"], s["start_s"], s["end_s"]) for p in kept for s in p["segments"]]
    out = []
    for ep in episodes:
        for b in ep.content_blocks:
            sig = hook_signal(b)
            if sig.score >= min_signal and not any(e == b.episode and a <= b.start <= z for e, a, z in covered):
                out.append((sig.score, b, sig.reasons))
    return sorted(out, key=lambda x: -x[0])[:limit]


def build_quality_report(meta: dict, episodes, ranked: list[dict], rejected: list[dict], calls: list[dict]) -> str:
    kept = [p for p in ranked if p["review_status"] != "duplicate"]
    dups = [p for p in ranked if p["review_status"] == "duplicate"]
    final_rej = [r for r in rejected if r["final"]]
    repaired_away = [r for r in rejected if not r["final"]]
    n_repaired = sum(1 for p in ranked if p.get("repaired"))
    proposed = sum(c.get("n_proposed", 0) for c in calls if c["attempt"] == 1)
    live = [c for c in calls if "latency_s" in c and not c["cached"]]
    reason_counts = Counter(i["code"] for r in rejected for i in r["issues"] if i["severity"] == "error")
    flag_counts = Counter(w["code"] for p in kept for w in p["warnings"])
    eps_with = {e for p in kept for e in p["episodes"]}
    no_promo = [e.number for e in episodes if e.number not in eps_with]
    parse_warn = [(e.number, w) for e in episodes for w in e.warnings]

    L = [f"# Quality report: {meta['show_title']}", "",
         f"`{meta['provider']}/{meta['model']}`, prompt `{meta['prompt_version']}`, {meta['run_at_utc']}, {meta['elapsed_s']}s", "",
         *([f"**Incomplete run:** {meta['stopped_early']}. Rerun the same command to resume (finished episodes come from the cache).", ""]
           if meta.get("stopped_early") else []),
         "## 1. Input (deterministic parser)",
         f"- Episodes parsed: {len(episodes)}; blocks: {sum(len(e.blocks) for e in episodes)} "
         f"({sum(len(e.content_blocks) for e in episodes)} dialogue/action/SFX)",
         f"- Parse warnings: {len(parse_warn)}"]
    L += [f"  - E{n:02d}: {w}" for n, w in parse_warn[:10]]
    L += ["", "## 2. LLM calls",
          f"- Calls: {len(calls)} ({len(live)} live, {sum(1 for c in calls if c.get('cached'))} from cache)",
          f"- Failed calls (network/API): {sum(1 for c in calls if 'latency_s' not in c)}",
          f"- Invalid JSON replies: {sum(1 for r in rejected if r['stage'] == 'json_parse')}"]
    if live:
        L.append(f"- Latency: mean {mean(c['latency_s'] for c in live):.1f}s, max {max(c['latency_s'] for c in live):.1f}s")
    L += ["", "## 3. Validation funnel",
          f"- Proposed on first attempt: {proposed}",
          f"- Rejected on first attempt and sent back with errors: {len(repaired_away)}",
          f"- Accepted: {len(kept) + len(dups)} (of which repaired on retry: {n_repaired})",
          f"- Finally rejected: {len(final_rej) + max(0, len(repaired_away) - n_repaired)} "
          f"({max(0, len(repaired_away) - n_repaired)} not fixed by the repair round)",
          f"- Removed as duplicates: {len(dups)}",
          f"- Kept for review: {len(kept)}  (`needs_review`: {sum(1 for p in kept if p['review_status'] == 'needs_review')})",
          f"- Hallucinated line IDs caught: {reason_counts.get('unknown_line_id', 0) + reason_counts.get('unknown_hook_id', 0) + reason_counts.get('unknown_cliffhanger_id', 0)}",
          f"- Non-verbatim quotes caught: {reason_counts.get('hook_quote_not_verbatim', 0) + reason_counts.get('cliffhanger_quote_not_verbatim', 0)}",
          "", "Rejection reasons (all attempts):"]
    L += [f"- `{k}`: {v}" for k, v in reason_counts.most_common()] or ["- none"]
    if kept:
        durs = [p["total_duration_s"] for p in kept]
        offs = [p["hook"]["offset_s"] or 0 for p in kept]
        L += ["", "## 4. Accepted promos",
              f"- Duration: min {min(durs):.0f}s, median {median(durs):.0f}s, max {max(durs):.0f}s",
              f"- Hook offset: median +{median(offs):.1f}s, {_pct(sum(o <= 1 for o in offs), len(offs))} within 1 s",
              f"- Multi-segment: {sum(len(p['segments']) > 1 for p in kept)}; cross-episode: {sum(len(p['episodes']) > 1 for p in kept)}",
              f"- Hook types: " + ", ".join(f"{k} {v}" for k, v in Counter(p['hook_type'] for p in kept).most_common()),
              f"- Episodes with no promo: {no_promo or 'none'}",
              "", "Review flags:"]
        L += [f"- `{k}`: {v}" for k, v in flag_counts.most_common()] or ["- none"]

    rr = meta.get("rerank") or {}
    L += ["", "## 4b. Side-by-side ranking (LLM orders the top clips, two passes, second reversed)"]
    if rr.get("applied"):
        after = {p["promo_id"]: p["rank"] for p in kept}
        L += [f"- Passes usable: {sum(1 for l in rr['log'] if l.get('ok'))}/{len(rr['log'])}; "
              f"average shift between passes: {rr.get('pass_disagreement')} places "
              f"(0 = identical; high = the model is unsure)", "",
              "| Before | After | Promo | Avg side-by-side position |", "|---|---|---|---|"]
        for i, pid in enumerate(rr["before"], 1):
            L.append(f"| {i} | {after.get(pid, '-')} | {pid} | {rr['avg_pos'].get(pid, '-')} |")
    else:
        L.append(f"- Not applied: {rr.get('skipped') or '; '.join(str(l.get('error')) for l in rr.get('log', [])) or 'no data'}")

    # Where the tool picks badly: places a human should look first.
    L += ["", "## 5. Where it may pick badly (check these first)", "",
          "**LLM picks with no rule signal on the hook** (the model may be over-rating a quiet opening):"]
    weak = [p for p in kept if p["rule_signals"]["hook"] < 1]
    L += [f"- {p['promo_id']} hook {p['hook']['line_id']}: {p['hook']['script_text'][:90]}" for p in weak[:10]] or ["- none"]
    L += ["", "**Strong rule-flagged hooks no promo covers** (possible misses):"]
    L += [f"- {b.id} ({', '.join(r[:3])}): {b.text[:90]}" for s, b, r in possible_misses(episodes, kept)] or ["- none"]
    L += ["", "**Final rejections** (first 15):"]
    for r in final_rej[:15]:
        msg = "; ".join(i["message"] for i in r["issues"] if i["severity"] == "error")
        L.append(f"- E{r['episode']:02d} attempt {r['attempt']} [{r['stage']}]: {msg[:200]}")
    if not final_rej:
        L.append("- none")
    L += ["", "## 6. Known blind spots (not checkable by rules)",
          "- Works from the script, not the video: a line that reads well may be badly acted or framed.",
          "- 'Standalone' is the model's opinion; the rules only flag low scores and stated missing context.",
          "- Keyword signals are English and genre-specific (regency/romance); new genres need new lexicons.",
          "- Multi-segment and cross-episode promos need an editor to check the cut lands cleanly."]
    return "\n".join(L) + "\n"


def _ts(s: str) -> float:
    m, sec = s.strip().split(":")
    return int(m) * 60 + float(sec)


def compare_with_gold(ranked: list[dict], gold_csv: Path, min_overlap: float = 0.3, k: int = 10) -> dict:
    gold = []
    with open(gold_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gold.append((int(r["episode"]), _ts(r["start"]), _ts(r["end"])))
    kept = [p for p in ranked if p["review_status"] != "duplicate"]

    def hits(p):
        for s in p["segments"]:
            for e, a, z in gold:
                inter = max(0.0, min(z, s["end_s"]) - max(a, s["start_s"]))
                if e == s["episode"] and inter / max(1e-6, min(z - a, s["duration_s"])) >= min_overlap:
                    return (e, a, z)
        return None

    matched = [hits(p) for p in kept]
    found = {m for m in matched if m}
    return {"gold": len(gold), "predicted": len(kept),
            "precision": round(sum(1 for m in matched if m) / len(kept), 3) if kept else 0,
            f"precision_at_{k}": round(sum(1 for m in matched[:k] if m) / min(k, len(kept)), 3) if kept else 0,
            "recall": round(len(found) / len(gold), 3) if gold else 0}


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    ranked = json.loads((run / "promos_ranked.json").read_text(encoding="utf-8"))
    if a.gold:
        print(json.dumps(compare_with_gold(ranked, Path(a.gold)), indent=2))
        return
    from parser import load_episodes
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    eps = [e for e in load_episodes(meta["input"]) if e.number in set(meta["episodes_processed"])]
    report = build_quality_report(meta, eps, ranked,
                                  json.loads((run / "rejected.json").read_text(encoding="utf-8")),
                                  json.loads((run / "llm_calls.json").read_text(encoding="utf-8")))
    (run / "quality_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
