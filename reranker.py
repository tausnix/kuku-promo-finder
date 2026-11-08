"""LLM, checked by code: order the top candidates side by side.

Why: when scoring one clip at a time, the model gave most promos 5/5/5, so its scores couldn't separate
a loud confrontation from a quiet plotting scene. Comparing clips side by side is more reliable.

Controls:
  * two passes, the second with the list reversed; positions are averaged (cancels position bias)
  * the reply must be a permutation of the given IDs: unknown IDs are dropped, duplicates ignored,
    missing IDs keep their old relative place; under half valid -> that pass is discarded
  * if no pass is usable (or the quota is gone) the deterministic order stands
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from llm_client import LLMClient, LLMError
from prompts import RERANK_SYSTEM, build_rerank_prompt, dumps


def parse_rerank(text: str, ids: list[str]) -> tuple[list[str] | None, dict, str | None, int]:
    """Return (full order, notes, error, n_missing)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        data = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return None, {}, "invalid JSON", 0
    order_raw = data.get("order") if isinstance(data, dict) else None
    if not isinstance(order_raw, list):
        return None, {}, "no 'order' list", 0
    valid, seen = [], set()
    for x in order_raw:
        x = str(x).strip().strip("[]")
        if x in ids and x not in seen:
            valid.append(x)
            seen.add(x)
    if len(valid) < len(ids) / 2:
        return None, {}, f"only {len(valid)} of {len(ids)} ids were valid", 0
    missing = [i for i in ids if i not in seen]
    notes = {k: str(v)[:120] for k, v in (data.get("notes") or {}).items() if k in ids} if isinstance(data.get("notes"), dict) else {}
    return valid + missing, notes, None, len(missing)


def rerank(cands: list[dict], index, client: LLMClient, show_title: str, raw_dir: Path, passes: int = 2) -> dict:
    ids = [p["promo_id"] for p in cands]
    by_id = {p["promo_id"]: p for p in cands}
    positions: dict[str, list[int]] = {i: [] for i in ids}
    notes: dict[str, str] = {}
    log, orders = [], []
    for k in range(passes):
        shown = ids if k % 2 == 0 else ids[::-1]
        prompt = build_rerank_prompt([by_id[i] for i in shown], show_title, index)
        (raw_dir / f"rerank_pass{k + 1}_prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            resp = client.generate(RERANK_SYSTEM, prompt)
        except LLMError as e:  # ranking is an extra; never let it break the run
            log.append({"pass": k + 1, "ok": False, "error": str(e)[:300]})
            if getattr(e, "fatal", False):
                break
            continue
        (raw_dir / f"rerank_pass{k + 1}_response.json").write_text(dumps(resp.to_dict()), encoding="utf-8")
        order, nts, err, n_missing = parse_rerank(resp.text, ids)
        log.append({"pass": k + 1, "ok": err is None, "error": err, "missing_ids": n_missing,
                    "cached": resp.cached, "latency_s": resp.latency_s})
        if err:
            continue
        orders.append(order)
        for pos, i in enumerate(order, 1):
            positions[i].append(pos)
        for i, n in nts.items():
            notes.setdefault(i, n)
    if not orders:
        return {"applied": False, "log": log, "before": ids}
    agreement = None
    if len(orders) >= 2:  # mean absolute rank shift between the two passes: 0 = identical
        p1 = {i: n for n, i in enumerate(orders[0])}
        agreement = round(mean(abs(p1[i] - n) for n, i in enumerate(orders[1])), 2)
    return {"applied": True, "avg_pos": {i: round(mean(v), 2) for i, v in positions.items() if v},
            "notes": notes, "log": log, "before": ids, "orders": orders, "pass_disagreement": agreement}
