"""HUMAN-IN-THE-LOOP: an editor reviews ranked promos, approves/rejects, and exports the final list.

    streamlit run app.py

Decisions are saved to <run>/review.json. Approved clips can be exported as gold.csv, which
evaluator.py uses to measure future runs (precision/recall vs the editor), closing the loop.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import PipelineConfig, ROOT, load_dotenv
from llm_client import PROVIDERS, LLMError
from parser import load_episodes, BlockIndex
from processor import run, parse_episode_range

# Streamlit Community Cloud injects secrets via st.secrets, not os.environ.
# Bridge them before the pipeline reads its API keys (llm_client reads env vars).
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:  # no secrets.toml locally, or an older Streamlit
    pass

load_dotenv()
OUT = ROOT / "outputs"
DEMO = ROOT / "demo"
REJECT_REASONS = ["weak hook", "no real cliffhanger", "needs too much context", "spoils a big twist",
                  "bad cut point", "duplicate", "other"]
st.set_page_config(page_title="Promo Finder", layout="wide")


@st.cache_data(show_spinner=False)
def load_run(run_dir: str):
    d = Path(run_dir)
    ranked = json.loads((d / "promos_ranked.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    return ranked, meta


@st.cache_resource(show_spinner=False)
def load_index(input_path: str):
    try:
        p = Path(input_path)
        return BlockIndex(load_episodes(p if p.is_absolute() else ROOT / p))
    except Exception:
        return None


def load_reviews(d: Path) -> dict:
    f = d / "review.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def runs() -> list[Path]:
    found = list(OUT.glob("*/promos_ranked.json")) + list(DEMO.glob("*/promos_ranked.json"))
    return sorted({p.parent for p in found}, key=lambda p: p.stat().st_mtime, reverse=True)


# The deployed app has no API key until one is added in Streamlit secrets, so pick a
# provider that can actually run instead of defaulting to a provider that will fail.
KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "huggingface": "HUGGING_FACE_API_KEY",
           "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def default_provider() -> str:
    return next((name for name, env in KEY_ENV.items() if os.getenv(env)), "mock")


# ---------------------------------------------------------------- sidebar: run the pipeline
with st.sidebar:
    st.header("Run on new scripts")
    up = st.file_uploader("Upload a .zip or .txt screenplay", type=["zip", "txt"])
    path_in = st.text_input("...or a local folder / .zip path (local runs only)")
    providers = list(PROVIDERS)
    _default = default_provider()
    provider = st.selectbox("Provider", providers, index=providers.index(_default))
    if _default == "mock":
        st.caption("No API key found, so the offline `mock` rules baseline is selected. Add "
                   "`GEMINI_API_KEY` (or another provider key) in the app's Streamlit secrets to use an LLM.")
    model = st.text_input("Model (blank = provider default)")
    eps = st.text_input("Episodes (blank = all, e.g. 1-5)")
    if st.button("Find promos", type="primary"):
        src = path_in.strip()
        if up is not None:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getvalue())
            src = str(tmp)
        if not src:
            st.error("Upload a file or give a path.")
        else:
            cfg = PipelineConfig(provider=provider, model=model or None)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = OUT / f"{provider}_{Path(src).stem[:30].replace(' ', '_')}_{stamp}"
            bar = st.progress(0.0, "starting")
            try:
                run(src, out, cfg, parse_episode_range(eps), progress=lambda n, t, m: bar.progress(n / t, m))
                st.success(f"Saved to {out.name}")
                st.cache_data.clear()
            except (LLMError, SystemExit, FileNotFoundError) as e:
                st.error(str(e))

# ---------------------------------------------------------------- main: review a run
st.title("Promo Finder: editor review")
all_runs = runs()
if not all_runs:
    st.info("No runs yet. Run `python processor.py --input <scripts> --provider mock` or use the sidebar.")
    st.stop()

run_dir = st.selectbox("Run", all_runs, format_func=lambda p: p.name)
ranked, meta = load_run(str(run_dir))
index = load_index(meta["input"])
reviews = load_reviews(run_dir)
st.caption(f"{meta['show_title']} | {meta['provider']}/{meta['model']} | prompt {meta['prompt_version']} | "
           f"{meta['run_at_utc']} | timestamps come from the script; the model only picks line IDs")

c1, c2, c3, c4 = st.columns(4)
status_f = c1.multiselect("Status", ["ready_for_review", "needs_review", "duplicate"], default=["ready_for_review", "needs_review"])
decision_f = c2.selectbox("Decision", ["all", "undecided", "approve", "reject"])
min_score = c3.slider("Min score", 0, 100, 0)
ep_options = sorted({e for p in ranked for e in p["episodes"]})
ep_f = c4.multiselect("Episodes", ep_options)

shown = [p for p in ranked if p["review_status"] in status_f and p["score"] >= min_score
         and (not ep_f or set(p["episodes"]) & set(ep_f))
         and (decision_f == "all" or reviews.get(p["promo_id"], {}).get("decision", "undecided") == decision_f)]
n_dec = sum(1 for v in reviews.values() if v.get("decision") in ("approve", "reject"))
st.write(f"**{len(shown)}** shown | {n_dec} decided | "
         f"{sum(1 for v in reviews.values() if v.get('decision') == 'approve')} approved")

for p in shown:
    pid = p["promo_id"]
    rv = reviews.get(pid, {})
    segs = " + ".join(f"E{s['episode']:02d} {s['start']}-{s['end']}" for s in p["segments"])
    badge = {"approve": "✅", "reject": "❌"}.get(rv.get("decision"), "⬜")
    with st.expander(f"{badge} #{p['rank'] or '-'}  {pid}  |  {p['score']}  |  {segs}  ({p['total_duration_s']:.0f}s)  |  {p['review_status']}"):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Hook** (+{p['hook']['offset_s']:.1f}s, {p['hook']['speaker']}): {p['hook']['script_text']}")
            st.markdown(f"**Cuts on** ({p['cliffhanger']['speaker']}): {p['cliffhanger']['script_text']}")
            st.markdown(f"**Why it works:** {p['why']}")
            st.markdown(f"**Why this rank:** {p['rank_reason']}")
            if p["context_needed"]:
                st.warning(f"Context a new viewer lacks: {p['context_needed']}")
            for w in p["warnings"]:
                st.caption(f"⚠ {w['code']}: {w['message']}")
            if index:
                with st.popover("Show script for this clip"):
                    for s in p["segments"]:
                        st.markdown(f"**E{s['episode']:02d} {s['start']} to {s['end']}**")
                        for b in index.range(s["start_id"], s["end_id"]):
                            who = b.speaker or b.kind
                            st.text(f"{b.id} {b.start:6.1f}s  {who}: {b.text}")
        with right:
            opts = ["undecided", "approve", "reject"]
            dec = st.radio("Decision", opts, index=opts.index(rv.get("decision", "undecided")), key=f"d_{pid}", horizontal=True)
            reason = st.selectbox("Reject reason", [""] + REJECT_REASONS,
                                  index=([""] + REJECT_REASONS).index(rv.get("reason", "")), key=f"r_{pid}",
                                  disabled=dec != "reject")
            note = st.text_input("Note / edited cut points", rv.get("note", ""), key=f"n_{pid}")
            prio = st.number_input("Post order (0 = use rank)", 0, 999, int(rv.get("post_order", 0)), key=f"o_{pid}")
            new = {"decision": dec, "reason": reason if dec == "reject" else "", "note": note, "post_order": prio}
            if new != {k: rv.get(k, d) for k, d in (("decision", "undecided"), ("reason", ""), ("note", ""), ("post_order", 0))}:
                reviews[pid] = new
                (run_dir / "review.json").write_text(json.dumps(reviews, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- exports
st.divider()
approved = [p for p in ranked if reviews.get(p["promo_id"], {}).get("decision") == "approve"]
approved.sort(key=lambda p: (reviews[p["promo_id"]].get("post_order") or 10_000, p["rank"] or 10_000))
buf = io.StringIO()
wr = csv.writer(buf)
wr.writerow(["post_order", "promo_id", "segments", "duration_s", "hook", "cliffhanger", "why", "note"])
for i, p in enumerate(approved, 1):
    wr.writerow([i, p["promo_id"], " + ".join(f"E{s['episode']:02d} {s['start']}-{s['end']}" for s in p["segments"]),
                 p["total_duration_s"], p["hook"]["script_text"], p["cliffhanger"]["script_text"], p["why"],
                 reviews[p["promo_id"]].get("note", "")])
gold = io.StringIO()
gw = csv.writer(gold)
gw.writerow(["episode", "start", "end"])
for p in approved:
    for s in p["segments"]:
        gw.writerow([s["episode"], s["start"], s["end"]])
a, b = st.columns(2)
a.download_button(f"Download approved list ({len(approved)})", buf.getvalue(), "approved_promos.csv", "text/csv")
b.download_button("Download as gold.csv (for evaluator.py)", gold.getvalue(), "gold.csv", "text/csv")
