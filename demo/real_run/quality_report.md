# Quality report: I Married a Broken Soldier. He was a Duke

`mock/rules-baseline`, prompt `promo-v2`, 2026-11-13T14:58:38+00:00, 1.5s

## 1. Input (deterministic parser)
- Episodes parsed: 50; blocks: 3727 (1973 dialogue/action/SFX)
- Parse warnings: 0

## 2. LLM calls
- Calls: 50 (50 live, 0 from cache)
- Failed calls (network/API): 0
- Invalid JSON replies: 0
- Latency: mean 0.0s, max 0.0s

## 3. Validation funnel
- Proposed on first attempt: 46
- Rejected on first attempt and sent back with errors: 0
- Accepted: 15 (of which repaired on retry: 0)
- Finally rejected: 0 (0 not fixed by the repair round)
- Removed as duplicates: 0
- Kept for review: 15  (`needs_review`: 0)
- Hallucinated line IDs caught: 0
- Non-verbatim quotes caught: 0

Rejection reasons (all attempts):
- none

## 4. Accepted promos
- Duration: min 24s, median 34s, max 53s
- Hook offset: median +0.2s, 100% within 1 s
- Multi-segment: 0; cross-episode: 0
- Hook types: shocking_line 4, threat 4, physical 2, violence 2, confrontation 1, reveal 1, humiliation 1
- Episodes with no promo: [1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 30, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50]

Review flags:
- none

## 4b. Side-by-side ranking (LLM orders the top clips, two passes, second reversed)
- Not applied: disabled

## 5. Where it may pick badly (check these first)

**LLM picks with no rule signal on the hook** (the model may be over-rating a quiet opening):
- none

**Strong rule-flagged hooks no promo covers** (possible misses):
- E25-081 (violence, threat, delivery:rage): "Worry? They struck you. They could have killed you."
- E43-074 (threat, delivery:shouting, delivery:anguish): "I was told he was dead!"
- E39-022 (violence, physical): The driver leaps off the cart, striking the sentry across the chest, seizing his neck, and
- E16-007 (delivery:rage, delivery:shouting, exclamation): "I said, move!"
- E23-043 (threat, delivery:shock): "Poison?"
- E28-013 (threat, delivery:menace): "She looks exactly like a poisoner."
- E29-067 (reveal, delivery:shouting, exclamation): "I saved the Duchess!"
- E30-013 (threat, reveal): "She knows the Duchess was poisoned."
- E32-034 (delivery:rage, delivery:shouting, exclamation): "You still think you are better?!"
- E32-042 (delivery:rage, delivery:shouting, exclamation): "You are lying!"
- E32-081 (delivery:rage, delivery:shouting, exclamation): "And what is wrong with that?!"
- E32-111 (delivery:rage, delivery:shouting, exclamation): "You think you can lecture me?!"

**Final rejections** (first 15):
- none

## 6. Known blind spots (not checkable by rules)
- Works from the script, not the video: a line that reads well may be badly acted or framed.
- 'Standalone' is the model's opinion; the rules only flag low scores and stated missing context.
- Keyword signals are English and genre-specific (regency/romance); new genres need new lexicons.
- Multi-segment and cross-episode promos need an editor to check the cut lands cleanly.
