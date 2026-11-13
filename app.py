"""Promo Finder — Kuku AI Generalist, Problem 1.

    streamlit run app.py

Human-in-the-loop review: ranked promos on the left, the full clip on the right.
Approving/rejecting writes <run>/review.json; approved clips export as a post list
and as gold.csv for evaluator.py.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from config import PipelineConfig, ROOT, load_dotenv
from llm_client import PROVIDERS, LLMError
from parser import BlockIndex, load_episodes
from processor import parse_episode_range, run

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
MODE_HELP = {
    "Fast": "One AI pass. The quickest useful run.",
    "Rules only": "Offline rules baseline. Takes seconds and needs no API key.",
    "Full": "Adds the second side-by-side judging pass. Slower, better ordering.",
}
MODEL_CHOICES = [(name, cls.default_model) for name, cls in PROVIDERS.items()]
KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "huggingface": "HUGGING_FACE_API_KEY",
           "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

st.set_page_config(page_title="PromoFinder · Kuku AI", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
.stApp { background: radial-gradient(1100px 520px at 18% -12%, #1a2030 0%, #0e1117 48%) fixed; }
section[data-testid="stSidebar"] { background: #0a0d13; border-right: 1px solid #1d2330; }
section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
.brand { font-size: 2.0rem; font-weight: 800; letter-spacing: -.045em; line-height: 1.1; }
.brand .dot { color: #FF4B4B; }
.brand .slash { color: #5b6472; font-weight: 500; margin: 0 .5rem; }
.brand .sub { color: #f3f4f6; font-weight: 700; letter-spacing: -.025em; }
.stats { color: #9ca3af; font-size: .8rem; margin: .3rem 0 1.0rem; }
.side-title { font-size: 1.05rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: .35rem; }
.side-copy { color: #9ca3af; font-size: .8rem; line-height: 1.45; margin-bottom: .9rem; }
.section-label { color: #6b7280; font-size: .66rem; letter-spacing: .15em; text-transform: uppercase;
                 font-weight: 800; margin: .1rem 0 .55rem; }
.promo-kicker { color: #FF4B4B; font-size: .7rem; letter-spacing: .16em; font-weight: 800;
                text-transform: uppercase; }
.promo-title { font-size: 1.6rem; font-weight: 800; letter-spacing: -.035em; margin: .1rem 0 .4rem; }
.badge { display: inline-block; border: 1px solid #FF4B4B; color: #ff9a9a; border-radius: 999px;
         padding: .1rem .55rem; font-size: .68rem; font-weight: 700; }
.badge.gray { border-color: #374151; color: #9ca3af; }
.label { color: #6b7280; font-size: .64rem; letter-spacing: .15em; text-transform: uppercase;
         font-weight: 800; margin-top: .95rem; }
.quote { font-size: 1.04rem; line-height: 1.45; color: #f9fafb; font-weight: 600; margin: .18rem 0 .12rem; }
.ts { color: #9ca3af; font-size: .74rem; margin-bottom: .55rem; }
.body { color: #cbd5e1; font-size: .85rem; line-height: 1.5; }
.transcript { max-height: 260px; overflow-y: auto; border: 1px solid #1f2531; border-radius: .5rem;
              padding: .55rem .7rem; background: #0a0d13; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: .74rem; }
.transcript .line { padding: .1rem 0; color: #cbd5e1; }
.transcript .tsm { color: #6b7280; }
.transcript .hl-hook { background: #3a1b1b; border-left: 2px solid #FF4B4B; padding-left: .4rem; }
.transcript .hl-cliff { background: #172538; border-left: 2px solid #60a5fa; padding-left: .4rem; }
.transcript .seg { color: #6b7280; margin-top: .45rem; font-weight: 700; }
.tl { position: relative; height: 7px; background: #1f2531; border-radius: 999px; margin: .3rem 0 .15rem;
      overflow: hidden; }
.tl-fill { position: absolute; top: 0; bottom: 0; background: #4b5563; }
.tl-hook { position: absolute; top: 0; bottom: 0; background: #FF4B4B; }
.step { display: flex; gap: .7rem; margin-bottom: 1rem; }
.step-num { flex: 0 0 1.35rem; width: 1.35rem; height: 1.35rem; border: 1px solid #FF4B4B; color: #FF4B4B;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: .7rem; font-weight: 800; }
.step-title { font-weight: 700; color: #f3f4f6; font-size: .88rem; margin-bottom: .12rem; }
.step-body { color: #9ca3af; font-size: .79rem; line-height: 1.45; }
div[data-testid="stButton"] > button[kind="primary"],
button[data-testid="stBaseButton-primary"] { background: #FF4B4B; border: 0; color: #fff; font-weight: 700;
    border-radius: .55rem; }
div[data-testid="stButton"] > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover { background: #ff6464; border: 0; color: #fff; }
[data-testid="stFileUploaderDropzone"] { background: #11151d; border: 1px dashed #2b3242; border-radius: .7rem; }
[data-testid="stDataFrame"] { border: 1px solid #1f2531; border-radius: .7rem; overflow: hidden; }
[data-testid="stMetricValue"] { font-size: 1.25rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
BRAND = ('<div class="brand"><span>kuku<span class="dot">.</span></span>'
         '<span class="slash">/</span><span class="sub">PromoFinder</span></div>')


# --------------------------------------------------------------------------- data helpers
@st.cache_data(show_spinner=False)
def load_run(run_dir: str):
    d = Path(run_dir)
    ranked = json.loads((d / "promos_ranked.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    return ranked, meta


@st.cache_data(show_spinner=False)
def load_run_meta(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "run_meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_episodes_meta(run_dir: str) -> list[dict]:
    f = Path(run_dir) / "episodes_parsed.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


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
    """Newest run first, by the run's own timestamp rather than file mtime.

    Streamlit Cloud checks out files in an arbitrary order, so mtime would pick a
    random run on startup. run_at_utc is written by the pipeline and is stable.
    """
    found = list(OUT.glob("*/promos_ranked.json")) + list(DEMO.glob("*/promos_ranked.json"))

    def newest_first(p: Path) -> float:
        try:
            return datetime.fromisoformat(load_run_meta(str(p))["run_at_utc"]).timestamp()
        except Exception:
            return p.stat().st_mtime

    return sorted({p.parent for p in found}, key=newest_first, reverse=True)


def run_label(p: Path) -> str:
    try:
        return f"Submitted run · {load_run_meta(str(p)).get('show_title', p.name)}"
    except Exception:
        return p.name


def default_provider() -> str:
    return next((name for name, env in KEY_ENV.items() if os.getenv(env)), "mock")


def clean_quote(t: str | None) -> str:
    return (t or "").strip().strip('"“”')


def use_label(p: dict) -> str:
    r = p.get("rank")
    if r == 1:
        return "Post first"
    if r and r <= 5:
        return "Test next"
    return "Retargeting"


def step(n: int, title: str, body: str) -> None:
    st.markdown(f'<div class="step"><span class="step-num">{n}</span><div>'
                f'<div class="step-title">{title}</div><div class="step-body">{body}</div></div></div>',
                unsafe_allow_html=True)


def render_promo(p: dict, index, reviews: dict, run_dir: Path,
                 episode_durations: dict[int, float]) -> None:
    pid = p["promo_id"]
    seg0 = p["segments"][0]
    rank = p["rank"]
    st.markdown(f'<div class="promo-kicker">{"Promo #" + str(rank) if rank else "Duplicate"}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="promo-title">Ep {seg0["episode"]}, {seg0["start"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge">{use_label(p)}</span> '
                f'<span class="badge gray">{p["review_status"].replace("_", " ")}</span>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="body" style="margin-top:.5rem">Episode {seg0["episode"]} · '
                f'{p["total_duration_s"]:.0f}s · {len(p["segments"])} segment'
                f'{"s" if len(p["segments"]) > 1 else ""} · score {p["score"]:.1f}/100</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="label">Opens on</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="quote">“{escape(clean_quote(p["hook"]["script_text"]))}”</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="ts">Ep {p["hook"]["episode"]}, {p["hook"]["timestamp"]} · '
                f'{escape(p["hook"]["speaker"] or "—")} · hook +{p["hook"]["offset_s"]:.1f}s</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="label">Cuts on</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="quote">“{escape(clean_quote(p["cliffhanger"]["script_text"]))}”</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="ts">Ep {p["cliffhanger"]["episode"]}, {p["cliffhanger"]["timestamp"]} · '
                f'{escape(p["cliffhanger"]["speaker"] or "—")} · '
                f'{p["cliffhanger"]["gap_to_end_s"]:.1f}s before the cut</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="label">Where it sits in the episode</div>', unsafe_allow_html=True)
    for s in p["segments"]:
        dur = max(float(episode_durations.get(s["episode"]) or s["end_s"] or 1.0), 0.1)
        left = max(0.0, min(100.0, s["start_s"] / dur * 100))
        width = max(0.5, min(100.0 - left, s["duration_s"] / dur * 100))
        hook_w = max(0.5, min(width, 3.0 / dur * 100))
        st.markdown(
            f'<div class="tl"><div class="tl-fill" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
            f'<div class="tl-hook" style="left:{left:.2f}%;width:{hook_w:.2f}%"></div></div>'
            f'<div class="ts" style="margin-top:-.1rem">Ep {s["episode"]:02d} · {s["start"]} – {s["end"]} '
            f'({s["duration_s"]:.0f}s of {dur:.0f}s)</div>', unsafe_allow_html=True)

    st.markdown('<div class="label">Why it works</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="body">{escape(p["why"] or "—")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="label">Why this rank</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="body">{escape(p["rank_reason"] or "—")}</div>', unsafe_allow_html=True)

    if p["context_needed"]:
        st.warning(f"Context a new viewer lacks: {p['context_needed']}")
    for w in p["warnings"]:
        st.caption(f"⚠ {w['code']}: {w['message']}")

    with st.expander("Full transcript · hook and cliffhanger marked"):
        parts: list[str] = []
        for s in p["segments"]:
            parts.append(f'<div class="seg">Ep {s["episode"]:02d} · {s["start"]} – {s["end"]}</div>')
            if index is None:
                parts.append('<div class="line">Transcript unavailable for this run.</div>')
                continue
            for b in index.range(s["start_id"], s["end_id"]):
                cls = ("hl-hook" if b.id == p["hook"]["line_id"]
                       else "hl-cliff" if b.id == p["cliffhanger"]["line_id"] else "")
                who = escape(b.speaker or b.kind)
                parts.append(f'<div class="line {cls}"><span class="tsm">{b.start:6.1f}s</span> '
                             f'<b>{who}</b> {escape(b.text)}</div>')
        st.markdown('<div class="transcript">' + "".join(parts) + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="label">Scores</div>', unsafe_allow_html=True)
    s = p["llm_scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Hook", f"{s['hook']}/5")
    c2.metric("Cliffhanger", f"{s['cliffhanger']}/5")
    c3.metric("Standalone", f"{s['standalone']}/5")
    st.progress(min(max(p["score"] / 100, 0.0), 1.0), text=f"Blended score {p['score']:.1f}/100")
    rs = p.get("rule_signals", {})
    st.caption(f"Rules: hook {rs.get('hook', 0):g} · cliffhanger {rs.get('cliff', 0):g}")

    st.markdown('<div class="label">Editor decision</div>', unsafe_allow_html=True)
    rv = reviews.get(pid, {})
    opts = ["undecided", "approve", "reject"]
    dec = st.radio("Decision", opts, index=opts.index(rv.get("decision", "undecided")),
                   horizontal=True, key=f"dec_{pid}", label_visibility="collapsed")
    reason = st.selectbox("Reject reason", [""] + REJECT_REASONS,
                          index=([""] + REJECT_REASONS).index(rv.get("reason", "")),
                          key=f"rej_{pid}", disabled=dec != "reject")
    note = st.text_input("Note / edited cut points", rv.get("note", ""), key=f"note_{pid}")
    prio = st.number_input("Post order (0 = use rank)", 0, 999, int(rv.get("post_order", 0)),
                           key=f"ord_{pid}")
    new = {"decision": dec, "reason": reason if dec == "reject" else "", "note": note,
           "post_order": prio}
    old = {"decision": rv.get("decision", "undecided"), "reason": rv.get("reason", ""),
           "note": rv.get("note", ""), "post_order": int(rv.get("post_order", 0))}
    if new != old:
        reviews[pid] = new
        (run_dir / "review.json").write_text(json.dumps(reviews, indent=2), encoding="utf-8")
        st.caption("Decision saved.")
    else:
        st.caption("Decisions save automatically.")


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-title">Run it on new scripts</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">Upload one script file per episode, or a single .zip of them. '
                'A whole series in one file also works.</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Upload", type=["zip", "txt"], accept_multiple_files=True,
                               label_visibility="collapsed")
    sample = st.checkbox("Use the bundled demo show instead", value=False)

    st.markdown("**Mode**")
    mode = st.radio("Mode", list(MODE_HELP), index=0, label_visibility="collapsed")
    st.caption(MODE_HELP[mode])

    st.markdown("**Model**")
    default_idx = next((i for i, (name, _) in enumerate(MODEL_CHOICES) if name == default_provider()),
                       len(MODEL_CHOICES) - 1)
    choice = st.selectbox("Model", range(len(MODEL_CHOICES)),
                          format_func=lambda i: f"{MODEL_CHOICES[i][0]} · {MODEL_CHOICES[i][1]}",
                          index=default_idx, label_visibility="collapsed",
                          disabled=(mode == "Rules only"))
    if default_provider() == "mock" and mode != "Rules only":
        st.caption("No API key found. Add `GEMINI_API_KEY` in the app's Streamlit secrets, "
                   "or use Rules only.")

    st.markdown("**How many promos**")
    target = st.slider("How many promos", 5, 50, 15, 5, label_visibility="collapsed")

    if st.button("Find promos", type="primary", width="stretch"):
        src = ""
        if sample:
            src = str(DEMO / "scripts")
        elif uploads:
            tmpdir = Path(tempfile.mkdtemp())
            for u in uploads:
                (tmpdir / u.name).write_bytes(u.getvalue())
            src = str(tmpdir)
        if not src:
            st.error("Upload a script (or tick the demo show) first.")
        else:
            provider, model = ("mock", None) if mode == "Rules only" else MODEL_CHOICES[choice]
            try:
                n_eps = len(load_episodes(src))
            except Exception:
                n_eps = 0
            per_ep = max(1, min(5, round(target / n_eps))) if n_eps else 3
            cfg = PipelineConfig(provider=provider, model=model, max_promos_per_episode=per_ep,
                                 max_attempts=1 if mode != "Full" else 2,
                                 rerank_top_k=20 if mode == "Full" else 0)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = OUT / f"{provider}_{Path(src).stem[:30].replace(' ', '_')}_{stamp}"
            bar = st.progress(0.0, "starting")
            try:
                run(src, out, cfg, parse_episode_range(""),
                    progress=lambda n, t, m: bar.progress(n / t, m), max_total=target)
                st.session_state["flash"] = f"Run saved: {out.name}"
                st.cache_data.clear()
                st.rerun()
            except (LLMError, SystemExit, FileNotFoundError) as e:
                st.error(str(e))

# --------------------------------------------------------------------------- header
if flash := st.session_state.pop("flash", None):
    st.success(flash)

all_runs = runs()
if not all_runs:
    st.markdown(BRAND, unsafe_allow_html=True)
    st.info("No runs yet. Start one from the sidebar, or tick the demo show.")
    st.stop()

head_l, head_r = st.columns([2.0, 1.35], vertical_alignment="center")
with head_l:
    st.markdown(BRAND, unsafe_allow_html=True)
with head_r:
    run_dir = st.selectbox("Run", all_runs, format_func=run_label, label_visibility="collapsed")

ranked, meta = load_run(str(run_dir))
if st.session_state.get("last_run") != str(run_dir):
    st.session_state["last_run"] = str(run_dir)
    st.session_state.pop("selected_promo", None)

eps_meta = load_episodes_meta(str(run_dir))
episode_durations = {e["episode"]: e.get("duration", 0.0) for e in eps_meta}
possible = sum(e.get("content_blocks", 0) for e in eps_meta)
n_promos = sum(1 for p in ranked if p["review_status"] != "duplicate")
judge = "rules baseline" if meta["provider"] == "mock" else f'{meta["provider"]}/{meta["model"]}'
double = " · two-pass side-by-side check" if meta.get("rerank", {}).get("applied") else ""
st.markdown(f'<div class="stats">{escape(meta.get("show_title", "Untitled show"))} · '
            f'{len(meta.get("episodes_processed", []))} episodes · {possible:,} script moments scored · '
            f'{n_promos} promos · {escape(judge)} picks and judges{double}</div>', unsafe_allow_html=True)

with st.expander("How to use this app", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        step(1, "Pick a promo on the left",
             "The list is already ranked. <b>Post first</b> (#1) is the safest strong clip to run first. "
             "<b>Test next</b> are the ones to try against it. <b>Retargeting</b> clips give away late twists, "
             "so they are for people already watching. Tick the checkbox next to a row to open it.")
        step(3, "Approve or reject",
             "At the bottom of each promo. Add a note if you like. Decisions are saved with the run so the "
             "ranking can be tuned later.")
    with c2:
        step(2, "Read it on the right",
             "The exact cut, a bar showing where it sits in the episode (bright red is the first 3 seconds, "
             "where the hook must land), the hook and cliffhanger lines, why it works and why it ranks there, "
             "anything to check before posting, the full transcript with the hook and cliffhanger marked, "
             "and the scores behind the ranking.")
        step(4, "Run it on new scripts",
             "Use the panel on the left. Upload one script file per episode, a .zip of them, or a whole series "
             "in one file, or tick the bundled demo. <b>Fast</b> uses the AI. <b>Rules only</b> takes seconds "
             "and needs no key. <b>Full</b> adds the second judge pass. The new run appears in the dropdown above.")

# --------------------------------------------------------------------------- table + detail
left, right = st.columns([1.08, 1], gap="large")
with left:
    label_col, dup_col = st.columns([3, 1], vertical_alignment="center")
    with label_col:
        st.markdown('<div class="section-label">Ranked promos</div>', unsafe_allow_html=True)
    with dup_col:
        show_dups = st.toggle("Duplicates", value=False, help="Include clips the dedupe step dropped.")

    statuses = ["ready_for_review", "needs_review"] + (["duplicate"] if show_dups else [])
    shown = [p for p in ranked if p["review_status"] in statuses]
    shown.sort(key=lambda p: (p["rank"] is None, p["rank"] or 10 ** 9))

    if not shown:
        st.info("No promos match this filter.")
    else:
        rows = []
        for p in shown:
            s0 = p["segments"][0]
            rows.append({"#": str(p["rank"]) if p["rank"] else "—",
                         "Clip": f"Ep {s0['episode']}  {s0['start']}",
                         "Length": f"{p['total_duration_s']:.0f}s",
                         "Use": use_label(p)})
        event = st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            height=min(600, 42 + 35 * len(rows)),
            on_select="rerun", selection_mode="single-row", key="promo_table",
            column_config={"#": st.column_config.TextColumn(width="small"),
                           "Clip": st.column_config.TextColumn(width="medium"),
                           "Length": st.column_config.TextColumn(width="small"),
                           "Use": st.column_config.TextColumn(width="small")})
        picked = getattr(getattr(event, "selection", None), "rows", []) or []
        if picked and picked[0] < len(shown):
            st.session_state["selected_promo"] = shown[picked[0]]["promo_id"]
        st.caption("Tick a row to open it on the right.")

with right:
    if shown:
        promo = next((p for p in shown
                      if p["promo_id"] == st.session_state.get("selected_promo")), shown[0])
        render_promo(promo, load_index(meta["input"]), load_reviews(run_dir), run_dir,
                     episode_durations)

# --------------------------------------------------------------------------- exports
st.divider()
reviews = load_reviews(run_dir)
approved = [p for p in ranked if reviews.get(p["promo_id"], {}).get("decision") == "approve"]
approved.sort(key=lambda p: (reviews[p["promo_id"]].get("post_order") or 10_000, p["rank"] or 10_000))
buf = io.StringIO()
wr = csv.writer(buf)
wr.writerow(["post_order", "promo_id", "segments", "duration_s", "hook", "cliffhanger", "why", "note"])
for i, p in enumerate(approved, 1):
    wr.writerow([i, p["promo_id"],
                 " + ".join(f"E{s['episode']:02d} {s['start']}-{s['end']}" for s in p["segments"]),
                 p["total_duration_s"], p["hook"]["script_text"], p["cliffhanger"]["script_text"],
                 p["why"], reviews[p["promo_id"]].get("note", "")])
gold = io.StringIO()
gw = csv.writer(gold)
gw.writerow(["episode", "start", "end"])
for p in approved:
    for s in p["segments"]:
        gw.writerow([s["episode"], s["start"], s["end"]])
a, b = st.columns(2)
a.download_button(f"Download approved list ({len(approved)})", buf.getvalue(),
                  "approved_promos.csv", "text/csv", width="stretch")
b.download_button("Download as gold.csv (for evaluator.py)", gold.getvalue(),
                  "gold.csv", "text/csv", width="stretch")
