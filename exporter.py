"""Writes every run to disk: JSON (full detail), CSV (spreadsheet), Markdown (human-readable)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _w(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def seg_str(p: dict) -> str:
    return " + ".join(f"E{s['episode']:02d} {s['start']}-{s['end']}" for s in p["segments"])


def _cell(s: str, n: int = 90) -> str:
    s = (s or "").replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


CSV_FIELDS = ["rank", "promo_id", "score", "review_status", "episodes", "segments", "total_duration_s",
              "hook_type", "hook_line_id", "hook_offset_s", "hook_text", "cliffhanger_line_id", "cliffhanger_text",
              "why", "rank_reason", "context_needed", "llm_hook", "llm_cliffhanger", "llm_standalone",
              "rule_hook", "rule_cliff", "rerank_pos", "rerank_note", "flags", "repaired", "duplicate_of",
              "provider", "model"]


def to_row(p: dict) -> dict:
    return {
        "rank": p["rank"], "promo_id": p["promo_id"], "score": p["score"], "review_status": p["review_status"],
        "episodes": " ".join(f"E{e:02d}" for e in p["episodes"]), "segments": seg_str(p),
        "total_duration_s": p["total_duration_s"], "hook_type": p["hook_type"],
        "hook_line_id": p["hook"]["line_id"], "hook_offset_s": p["hook"]["offset_s"], "hook_text": p["hook"]["script_text"],
        "cliffhanger_line_id": p["cliffhanger"]["line_id"], "cliffhanger_text": p["cliffhanger"]["script_text"],
        "why": p["why"], "rank_reason": p["rank_reason"], "context_needed": p["context_needed"],
        "llm_hook": p["llm_scores"]["hook"], "llm_cliffhanger": p["llm_scores"]["cliffhanger"],
        "llm_standalone": p["llm_scores"]["standalone"], "rule_hook": p["rule_signals"]["hook"],
        "rule_cliff": p["rule_signals"]["cliff"], "rerank_pos": p.get("rerank_pos") if p.get("rerank_pos") is not None else "",
        "rerank_note": p.get("rerank_note", ""), "flags": "; ".join(w["code"] for w in p["warnings"]),
        "repaired": p.get("repaired", False), "duplicate_of": p.get("duplicate_of") or "",
        "provider": p["source"]["provider"], "model": p["source"]["model"],
    }


def to_markdown(ranked: list[dict], meta: dict, top_n: int) -> str:
    kept = [p for p in ranked if p["review_status"] != "duplicate"]
    L = [f"# Promo picks: {meta['show_title']}", "",
         f"Provider `{meta['provider']}` / model `{meta['model']}` / prompt `{meta['prompt_version']}` / "
         f"run {meta['run_at_utc']}. {len(kept)} promos from {len(meta['episodes_processed'])} episodes. "
         "All timestamps come from the script, not the model.", "",
         f"## Post first (top {min(top_n, len(kept))})", "",
         "| # | Score | Promo | Clip | Length | Hook (first 3 s) | Cuts on |", "|---|---|---|---|---|---|---|"]
    for p in kept[:top_n]:
        L.append(f"| {p['rank']} | {p['score']} | {p['promo_id']} | {seg_str(p)} | {p['total_duration_s']:.0f}s | "
                 f"{_cell(p['hook']['script_text'], 70)} | {_cell(p['cliffhanger']['script_text'], 70)} |")
    L += ["", "## All promos", ""]
    for p in kept:
        L += [f"### {p['rank']}. {p['promo_id']}: {p['title'] or p['hook_type']} (score {p['score']})", ""]
        for s in p["segments"]:
            L.append(f"- Segment: **E{s['episode']:02d} {s['start']} to {s['end']}** ({s['duration_s']:.1f}s, lines {s['start_id']}..{s['end_id']})")
        h, c = p["hook"], p["cliffhanger"]
        L += [f"- Total: {p['total_duration_s']:.1f}s",
              f"- Hook (+{h['offset_s']:.1f}s, {h['line_id']}, {h['speaker']}): {h['script_text']}",
              f"- Cliffhanger ({c['line_id']}, {c['speaker']}, cut {c['gap_to_end_s']:.1f}s after): {c['script_text']}",
              f"- Why it works: {p['why']}",
              f"- Why this rank: {p['rank_reason']}"]
        if p["context_needed"]:
            L.append(f"- Context a new viewer lacks: {p['context_needed']}")
        if p["warnings"]:
            L.append("- Flags: " + "; ".join(f"`{w['code']}` {w['message']}" for w in p["warnings"]))
        L += [f"- Review status: {p['review_status']}", ""]
    dups = [p for p in ranked if p["review_status"] == "duplicate"]
    if dups:
        L += ["## Removed as duplicates", ""] + [f"- {p['promo_id']} ({seg_str(p)}) overlaps {p['duplicate_of']}" for p in dups]
    return "\n".join(L) + "\n"


def export_all(out_dir: Path, ranked: list[dict], rejected: list[dict], calls: list[dict], episodes, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _w(out_dir / "promos_ranked.json", ranked)
    _w(out_dir / "rejected.json", rejected)
    _w(out_dir / "llm_calls.json", calls)
    _w(out_dir / "run_meta.json", meta)
    _w(out_dir / "episodes_parsed.json", [
        {"episode": e.number, "file": e.file, "duration": e.duration, "scenes": len(e.scenes),
         "blocks": len(e.blocks), "content_blocks": len(e.content_blocks), "parse_warnings": e.warnings}
        for e in episodes])
    with open(out_dir / "promos_ranked.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for p in ranked:
            wr.writerow(to_row(p))
    top_n = meta["config"]["weights"]["top_n"]
    (out_dir / "promos_ranked.md").write_text(to_markdown(ranked, meta, top_n), encoding="utf-8")
