"""DETERMINISTIC: screenplay text -> structured, time-stamped blocks.

Every block gets a stable ID like ``E07-031`` (episode 7, block 31). The LLM only ever
refers to these IDs; all timestamps in the output come from this parser, never from the model.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
TAG_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(r"^Start:\s*\(\s*" + TS + r"\s*\)\s*\|\s*End:\s*\(\s*" + TS + r"\s*\)\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$")
SCENE_HEADING_RE = re.compile(r"^Scene heading:\s*(.*)$")
SCENE_START_RE = re.compile(r"^Scene start:\s*" + TS)
SCENE_END_RE = re.compile(r"^Scene end:\s*" + TS)
EP_NUM_RE = re.compile(r"^EPISODE NUMBER:\s*Episode\s+(\d+)", re.I)
LOGLINE_RE = re.compile(r"^EPISODE LOGLINE:\s*(.*)$")
CONT_RE = re.compile(r"^CONTINUITY NOTES:\s*(.*)$")
FILE_EP_RE = re.compile(r"episode-(\d+)", re.I)
NON_DIALOGUE = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX"}


def parse_ts(minutes: str, seconds: str) -> float:
    return round(int(minutes) * 60 + float(seconds), 2)


def fmt_ts(t: float) -> str:
    t = round(max(t, 0.0), 2)
    m = int(t // 60)
    s = round(t - 60 * m, 2)
    if s >= 60:
        m, s = m + 1, s - 60
    return f"{m:02d}:{s:05.2f}"


def block_id(episode: int, idx: int) -> str:
    return f"E{episode:02d}-{idx:03d}"


def episode_of(bid: str) -> int | None:
    m = re.match(r"^E(\d+)-\d+$", bid or "")
    return int(m.group(1)) if m else None


@dataclass
class Block:
    id: str
    episode: int
    idx: int
    scene: int
    kind: str              # DIALOGUE | ACTION | SFX | CAMERA
    speaker: str | None
    direction: str         # parentheticals, e.g. "O.S., Firm"
    start: float
    end: float
    text: str
    source_line: int

    @property
    def is_content(self) -> bool:
        return self.kind != "CAMERA"


@dataclass
class Episode:
    number: int
    file: str
    logline: str
    continuity: str
    duration: float
    scenes: list[dict]
    blocks: list[Block]
    warnings: list[str] = field(default_factory=list)

    @property
    def content_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.is_content]

    @property
    def camera_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "CAMERA"]

    def to_dict(self) -> dict:
        return asdict(self)


def _split_tag(tag: str) -> tuple[str, str | None, str]:
    parens = re.findall(r"\(([^)]*)\)", tag)
    name = re.sub(r"\s*\([^)]*\)", "", tag).strip()
    kind = NON_DIALOGUE.get(name.upper())
    if kind:
        return kind, None, ", ".join(parens)
    return "DIALOGUE", name, ", ".join(parens)


def parse_episode(text: str, filename: str = "") -> Episode:
    text = text.lstrip("﻿")
    lines = text.splitlines()
    warnings: list[str] = []
    number = None
    logline = continuity = ""
    scenes: list[dict] = []
    blocks: list[Block] = []
    scene_no = 0
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        if cur.get("start") is None:
            warnings.append(f"line {cur['lineno']}: block '{cur['tag']}' has no Start/End timestamps - dropped")
        elif not cur["text"]:
            warnings.append(f"line {cur['lineno']}: block '{cur['tag']}' has no text - dropped")
        else:
            kind, speaker, direction = _split_tag(cur["tag"])
            start, end = cur["start"], cur["end"]
            if end < start:
                warnings.append(f"line {cur['lineno']}: end {fmt_ts(end)} < start {fmt_ts(start)} - clamped")
                end = start
            idx = len(blocks) + 1
            blocks.append(Block(
                id=block_id(number or 0, idx), episode=number or 0, idx=idx, scene=scene_no,
                kind=kind, speaker=speaker, direction=direction, start=start, end=end,
                text=" ".join(cur["text"]).strip(), source_line=cur["lineno"],
            ))
        cur = None

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if number is None and (m := EP_NUM_RE.match(line)):
            number = int(m.group(1))
            continue
        if m := LOGLINE_RE.match(line):
            logline = m.group(1).strip()
            continue
        if m := CONT_RE.match(line):
            continuity = m.group(1).strip()
            continue
        if m := SCENE_RE.match(line):
            flush()
            scene_no = int(m.group(1))
            scenes.append({"scene": scene_no, "heading": "", "start": None, "end": None})
            continue
        if scenes and (m := SCENE_HEADING_RE.match(line)):
            scenes[-1]["heading"] = m.group(1).strip()
            continue
        if scenes and (m := SCENE_START_RE.match(line)):
            scenes[-1]["start"] = parse_ts(*m.groups())
            continue
        if scenes and (m := SCENE_END_RE.match(line)):
            scenes[-1]["end"] = parse_ts(*m.groups())
            continue
        if line.startswith("--- SCRIPT"):
            flush()
            continue
        if m := TAG_RE.match(line):
            flush()
            cur = {"tag": m.group(1).strip(), "lineno": i, "start": None, "end": None, "text": []}
            continue
        if cur is not None and cur["start"] is None and (m := TIME_RE.match(line)):
            g = m.groups()
            cur["start"], cur["end"] = parse_ts(g[0], g[1]), parse_ts(g[2], g[3])
            continue
        if not line:
            if cur is not None and cur["text"]:
                flush()
            continue
        if cur is not None:
            cur["text"].append(line)
    flush()

    if number is None:
        m = FILE_EP_RE.search(filename)
        number = int(m.group(1)) if m else 0
        warnings.append(f"no 'EPISODE NUMBER' header; using {number} from filename")
        for b in blocks:  # re-key IDs now that we know the episode
            b.episode = number
            b.id = block_id(number, b.idx)

    scene_ends = [s["end"] for s in scenes if s.get("end") is not None]
    duration = max(scene_ends + [b.end for b in blocks] + [0.0])
    if scene_ends and blocks and max(b.end for b in blocks) > max(scene_ends) + 0.5:
        warnings.append("some blocks end after the last 'Scene end' timestamp")
    if not blocks:
        warnings.append("no blocks parsed - check file format")

    return Episode(number=number, file=filename, logline=logline, continuity=continuity,
                   duration=duration, scenes=scenes, blocks=blocks, warnings=warnings)


def load_episodes(path: str | Path) -> list[Episode]:
    """Load every .txt screenplay from a folder, a single file, or a .zip."""
    path = Path(path)
    items: list[tuple[str, str]] = []
    if path.is_dir():
        for f in sorted(path.rglob("*.txt")):
            items.append((f.name, f.read_text(encoding="utf-8-sig", errors="replace")))
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith(".txt") and not name.startswith("__MACOSX"):
                    items.append((Path(name).name, z.read(name).decode("utf-8-sig", errors="replace")))
    elif path.is_file():
        items.append((path.name, path.read_text(encoding="utf-8-sig", errors="replace")))
    else:
        raise FileNotFoundError(f"Input not found: {path}")

    episodes = [parse_episode(text, name) for name, text in items]
    episodes.sort(key=lambda e: e.number)
    seen: set[int] = set()
    for ep in episodes:
        if ep.number in seen:
            ep.warnings.append(f"duplicate episode number {ep.number}")
        seen.add(ep.number)
    return episodes


class BlockIndex:
    """Lookup of every block by ID across all loaded episodes."""

    def __init__(self, episodes: list[Episode]):
        self.episodes = {e.number: e for e in episodes}
        self.by_id: dict[str, Block] = {b.id: b for e in episodes for b in e.blocks}

    def get(self, bid: str) -> Block | None:
        return self.by_id.get((bid or "").strip())

    def range(self, start_id: str, end_id: str) -> list[Block]:
        a, b = self.get(start_id), self.get(end_id)
        if not a or not b or a.episode != b.episode:
            return []
        return [x for x in self.episodes[a.episode].blocks if a.idx <= x.idx <= b.idx]
