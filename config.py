"""All tunable knobs in one place. Every number here is a deterministic rule, not a model choice."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass
class Rules:
    """Hard + soft rules the validator enforces on every LLM-proposed promo."""
    min_duration_s: float = 20.0        # brief: promo is 20-60 s
    max_duration_s: float = 60.0
    hook_window_s: float = 3.0          # brief: hook in the first 3 seconds
    cliff_window_s: float = 6.0         # cliffhanger line must end within N s of the clip end
    max_segments: int = 4               # promos may join a few moments
    snap_tolerance_s: float = 1.5       # snap cut points to camera-shot boundaries if this close
    max_quote_words: int = 25
    min_quote_chars: int = 3
    next_episode_context_s: float = 30.0  # how much of episode N+1 the LLM may use as a continuation


@dataclass
class RankWeights:
    """Final score = weighted sum of LLM judgement and deterministic signals (0-100).

    LLM scores are only 40%: on the first full run the model gave most promos 5/5/5, so its
    absolute scores barely separate good from weak. Measured signals break the ties.
    """
    llm_hook: float = 0.15
    llm_cliffhanger: float = 0.12
    llm_standalone: float = 0.13
    rule_hook: float = 0.20
    rule_cliff: float = 0.10
    duration_fit: float = 0.10
    hook_speed: float = 0.08
    funnel: float = 0.12               # earlier episodes = fewer spoilers, shorter path for a new viewer
    warning_penalty: float = 3.0       # points off per warning
    max_per_episode_in_top: int = 2    # diversity rule for the "post first" list
    # Final place for the top clips = blend of score place and side-by-side place.
    # 0 = scores only, 1 = side-by-side only. Alone, side-by-side favoured late spoiler episodes and
    # its two passes disagreed by ~2 places; alone, scores let quiet scenes through. 0.5 averages them.
    rerank_weight: float = 0.5
    top_n: int = 10


@dataclass
class PipelineConfig:
    provider: str = "mock"
    model: str | None = None
    max_promos_per_episode: int = 3
    max_attempts: int = 2               # 1 call + 1 repair round with validator feedback
    include_camera_lines: bool = False  # camera lines double the tokens; we use them only for snapping
    use_cache: bool = True
    min_interval_s: float | None = None # rate limit between non-cached calls (None = provider default)
    rerank_top_k: int = 20              # side-by-side ordering of the best N clips (0 = off)
    rerank_passes: int = 2              # second pass sees the list reversed, to cancel position bias
    rules: Rules = field(default_factory=Rules)
    weights: RankWeights = field(default_factory=RankWeights)

    def to_dict(self) -> dict:
        return asdict(self)
