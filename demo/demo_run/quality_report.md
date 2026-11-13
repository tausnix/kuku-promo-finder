# Quality report: The Heiress (demo)

`mock/rules-baseline`, prompt `promo-v2`, 2026-11-13T14:12:07+00:00, 0.0s

## 1. Input (deterministic parser)
- Episodes parsed: 3; blocks: 62 (59 dialogue/action/SFX)
- Parse warnings: 0

## 2. LLM calls
- Calls: 3 (3 live, 0 from cache)
- Failed calls (network/API): 0
- Invalid JSON replies: 0
- Latency: mean 0.0s, max 0.0s

## 3. Validation funnel
- Proposed on first attempt: 8
- Rejected on first attempt and sent back with errors: 0
- Accepted: 8 (of which repaired on retry: 0)
- Finally rejected: 0 (0 not fixed by the repair round)
- Removed as duplicates: 2
- Kept for review: 6  (`needs_review`: 0)
- Hallucinated line IDs caught: 0
- Non-verbatim quotes caught: 0

Rejection reasons (all attempts):
- none

## 4. Accepted promos
- Duration: min 26s, median 36s, max 56s
- Hook offset: median +0.0s, 100% within 1 s
- Multi-segment: 0; cross-episode: 0
- Hook types: physical 3, confrontation 2, slap 1
- Episodes with no promo: none

Review flags:
- none

## 4b. Side-by-side ranking (LLM orders the top clips, two passes, second reversed)
- Passes usable: 2/2; average shift between passes: 3 places (0 = identical; high = the model is unsure)

| Before | After | Promo | Avg side-by-side position |
|---|---|---|---|
| 1 | 1 | E01-P2 | 3.5 |
| 2 | 2 | E03-P3 | 3.5 |
| 3 | 3 | E03-P2 | 3.5 |
| 4 | 4 | E02-P2 | 3.5 |
| 5 | 5 | E01-P3 | 3.5 |
| 6 | 6 | E02-P1 | 3.5 |

## 5. Where it may pick badly (check these first)

**LLM picks with no rule signal on the hook** (the model may be over-rating a quiet opening):
- none

**Strong rule-flagged hooks no promo covers** (possible misses):
- E01-003 (confrontation, delivery:shock, exclamation): "How dare you!"
- E01-004 (accusation, humiliation, delivery:rage): "You are a worthless liar, and this house is done with you."
- E01-002 (slap): Vera slaps Mara across the face.

**Final rejections** (first 15):
- none

## 6. Known blind spots (not checkable by rules)
- Works from the script, not the video: a line that reads well may be badly acted or framed.
- 'Standalone' is the model's opinion; the rules only flag low scores and stated missing context.
- Keyword signals are English and genre-specific (regency/romance); new genres need new lexicons.
- Multi-segment and cross-episode promos need an editor to check the cut lands cleanly.
