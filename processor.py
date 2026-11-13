"""Pipeline orchestrator + CLI.

    python processor.py --input "scripts.zip" --provider gemini --out outputs/gemini_run
    python processor.py --input "scripts.zip" --provider mock          # offline, no key
    python processor.py --input data/ --provider groq --episodes 1-10

Flow per episode:
  parse (rules) -> prompt with line IDs (rules) -> LLM proposes promos (AI)
  -> validate every promo (rules) -> one repair round with the validator's errors (AI)
  -> rank + de-duplicate (rules) -> export -> human review in app.py (human)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import PipelineConfig, ROOT, load_dotenv
from evaluator import build_quality_report
from exporter import export_all
from llm_client import get_client, LLMError, LLMClient
from parser import load_episodes, BlockIndex, Episode
from prompts import SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt, build_repair_prompt, build_json_fix_prompt, dumps
from ranker import rank
from reranker import rerank
from validator import validate_promo, parse_llm_json


def parse_episode_range(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def infer_show_title(path: Path) -> str:
    name = path.stem if path.suffix else path.name
    return re.sub(r"-scripts.*$", "", name).strip() or "Untitled show"


def process_episode(ep: Episode, next_ep: Episode | None, index: BlockIndex, client: LLMClient,
                    cfg: PipelineConfig, show_title: str, raw_dir: Path) -> dict:
    allowed = {ep.number} | ({next_ep.number} if next_ep else set())
    user = build_user_prompt(ep, next_ep, show_title, cfg.max_promos_per_episode,
                             cfg.include_camera_lines, cfg.rules.next_episode_context_s)
    prompt = user
    accepted: list[dict] = []
    rejected: list[dict] = []
    calls: list[dict] = []

    for attempt in range(1, cfg.max_attempts + 1):
        tag = f"E{ep.number:02d}_attempt{attempt}"
        (raw_dir / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            resp = client.generate(SYSTEM_PROMPT, prompt)
        except LLMError as e:
            if getattr(e, "fatal", False):
                raise
            calls.append({"episode": ep.number, "attempt": attempt, "ok": False, "error": str(e)})
            rejected.append({"episode": ep.number, "attempt": attempt, "final": True, "stage": "llm_call", "raw": None,
                             "issues": [{"code": "llm_call_failed", "message": str(e)[:500], "severity": "error"}]})
            break
        (raw_dir / f"{tag}_response.json").write_text(dumps(resp.to_dict()), encoding="utf-8")

        promos, perr = parse_llm_json(resp.text)
        calls.append({"episode": ep.number, "attempt": attempt, "ok": perr is None, "error": perr,
                      "cached": resp.cached, "latency_s": resp.latency_s, "usage": resp.usage,
                      "n_proposed": len(promos or [])})
        if perr:
            final = attempt == cfg.max_attempts
            rejected.append({"episode": ep.number, "attempt": attempt, "final": final, "stage": "json_parse", "raw": resp.text[:1000],
                             "issues": [{"code": "invalid_json", "message": perr, "severity": "error"}]})
            if not final:
                prompt = build_json_fix_prompt(user, perr)
                continue
            break

        if len(promos) > cfg.max_promos_per_episode:
            promos = promos[: cfg.max_promos_per_episode]  # obey the cap; extra ones are ignored

        failures = []
        for i, raw in enumerate(promos):
            promo, issues = validate_promo(raw, index, allowed, cfg.rules)
            if promo:
                promo["repaired"] = attempt > 1
                promo["source"] = {"provider": client.provider, "model": client.model,
                                   "prompt_version": PROMPT_VERSION, "attempt": attempt}
                accepted.append(promo)
            else:
                failures.append({"index": i, "raw": raw, "issues": [x.to_dict() for x in issues]})

        final = not failures or attempt == cfg.max_attempts
        for f in failures:
            rejected.append({"episode": ep.number, "attempt": attempt, "final": final, "stage": "validation", **f})
        if final:
            break
        prompt = build_repair_prompt(user, resp.text, failures)

    for k, p in enumerate(sorted(accepted, key=lambda p: p["segments"][0]["start_s"]), 1):
        p["promo_id"] = f"E{ep.number:02d}-P{k}"
    return {"accepted": accepted, "rejected": rejected, "calls": calls}


def run(input_path: str | Path, out_dir: str | Path, cfg: PipelineConfig, episodes: set[int] | None = None,
        show_title: str | None = None, progress=None, max_total: int | None = None) -> dict:
    t0 = time.time()
    input_path, out_dir = Path(input_path), Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_eps = load_episodes(input_path)
    if not all_eps:
        raise SystemExit(f"No .txt screenplays found in {input_path}")
    index = BlockIndex(all_eps)
    by_num = {e.number: e for e in all_eps}
    selected = [e for e in all_eps if episodes is None or e.number in episodes]
    show_title = show_title or infer_show_title(input_path)
    client = get_client(cfg.provider, cfg.model, use_cache=cfg.use_cache, min_interval_s=cfg.min_interval_s)

    accepted, rejected, calls = [], [], []
    stopped_early = None
    done_eps = []
    for n, ep in enumerate(selected, 1):
        try:
            res = process_episode(ep, by_num.get(ep.number + 1), index, client, cfg, show_title, raw_dir)
        except LLMError as e:
            if not getattr(e, "fatal", False) or n == 1:
                raise  # bad key / unknown model on the first call: nothing to save
            # Quota ran out mid-run: keep what we have. Rerunning later resumes for free from the cache.
            stopped_early = f"stopped at E{ep.number:02d}: {str(e)[:300]}"
            print(stopped_early, flush=True)
            break
        done_eps.append(ep)
        accepted += res["accepted"]
        rejected += res["rejected"]
        calls += res["calls"]
        msg = (f"[{n}/{len(selected)}] E{ep.number:02d}: {len(res['accepted'])} accepted, "
               f"{sum(1 for r in res['rejected'] if r['final'])} rejected")
        print(msg, flush=True)
        if progress:
            progress(n, len(selected), msg)

    ranked = rank(accepted, cfg.weights)
    rr = {"applied": False, "log": [], "skipped": "disabled"}
    if cfg.rerank_top_k > 1 and not stopped_early:
        cands = [p for p in ranked if p["review_status"] != "duplicate"][: cfg.rerank_top_k]
        if len(cands) > 1:
            rr = rerank(cands, index, client, show_title, raw_dir, cfg.rerank_passes)
            if rr["applied"]:
                ranked = rank(accepted, cfg.weights, rerank=rr)
            print(f"Side-by-side ranking: {'applied' if rr['applied'] else 'not applied'} "
                  f"({sum(1 for l in rr['log'] if l.get('ok'))}/{cfg.rerank_passes} passes ok)", flush=True)
    if max_total:
        ranked = ranked[:max_total]
    meta = {
        "show_title": show_title,
        "input": str(input_path.resolve()),
        "provider": client.provider,
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "episodes_processed": [e.number for e in done_eps],
        "stopped_early": stopped_early,
        "rerank": {k: v for k, v in rr.items() if k != "notes"},
        "config": cfg.to_dict(),
    }
    export_all(out_dir, ranked, rejected, calls, done_eps, meta)
    report = build_quality_report(meta, done_eps, ranked, rejected, calls)
    (out_dir / "quality_report.md").write_text(report, encoding="utf-8")
    kept = [p for p in ranked if p["review_status"] != "duplicate"]
    print(f"Done in {meta['elapsed_s']}s -> {out_dir}  ({len(kept)} promos, {len(ranked) - len(kept)} duplicates)")
    return {"out_dir": str(out_dir), "promos": len(kept), "meta": meta}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    ap = argparse.ArgumentParser(description="Find promo clips in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of .txt scripts, a single .txt, or a .zip")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<provider>_<timestamp>)")
    ap.add_argument("--provider", default=None, help="gemini | groq | huggingface | openai | anthropic | ollama | mock")
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--episodes", default=None, help="e.g. 1-5,9")
    ap.add_argument("--max-promos", type=int, default=3, help="max promos per episode")
    ap.add_argument("--attempts", type=int, default=2, help="LLM calls per episode incl. repair rounds")
    ap.add_argument("--min-interval", type=float, default=None, help="seconds between live LLM calls")
    ap.add_argument("--include-camera", action="store_true", help="send camera lines to the LLM too")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses")
    ap.add_argument("--rerank-top", type=int, default=20, help="side-by-side order the best N clips (0 = off)")
    ap.add_argument("--max-total", type=int, default=None, help="cap the total number of ranked promos")
    ap.add_argument("--show-title", default=None)
    a = ap.parse_args(argv)

    import os
    from llm_client import normalize_provider
    provider = normalize_provider(a.provider or os.getenv("LLM_PROVIDER")) or "mock"
    cfg = PipelineConfig(provider=provider, model=a.model, max_promos_per_episode=a.max_promos,
                         max_attempts=max(1, a.attempts), include_camera_lines=a.include_camera,
                         use_cache=not a.no_cache, min_interval_s=a.min_interval, rerank_top_k=a.rerank_top)
    out = a.out or str(ROOT / "outputs" / f"{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        run(a.input, out, cfg, parse_episode_range(a.episodes), a.show_title, max_total=a.max_total)
    except LLMError as e:
        sys.exit(f"LLM setup error: {e}")


if __name__ == "__main__":
    main()
