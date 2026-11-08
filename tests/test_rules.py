"""Deterministic rule tests. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Rules, RankWeights  # noqa: E402
from parser import parse_episode, BlockIndex, fmt_ts  # noqa: E402
from ranker import rank  # noqa: E402
from validator import validate_promo, parse_llm_json  # noqa: E402


def make_script(ep: int, lines: list[tuple[str, float, float, str]]) -> str:
    body = [f"EPISODE NUMBER: Episode {ep}", "EPISODE LOGLINE: test", "CONTINUITY NOTES: test", "",
            "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. HALL - DAY",
            "Scene start: 00:00.00", f"Scene end: {fmt_ts(lines[-1][2])}", "", "Actions and dialogue:", ""]
    for tag, s, e, text in lines:
        body += [f"**{tag}:**", f"Start: ({fmt_ts(s)}) | End: ({fmt_ts(e)})", text, ""]
    return "\n".join(body + ["--- SCRIPT END ---"])


# Episode 1: camera shot 0-5, slap at 1.0, dialogue every 5 s up to 70 s.
LINES = [("CAMERA", 0.0, 5.0, "Wide shot of the hall."),
         ("ACTION", 1.0, 3.0, "Vera slaps Mara across the face."),
         ("MARA (Shocked)", 3.5, 5.0, '"How dare you!"')]
LINES += [(f"TOM (Cold)", t, t + 4.0, f'"Line number {i} of the argument."') for i, t in enumerate(range(5, 70, 5), 1)]
LINES += [("STRANGER (O.S.) (Menacing)", 70.0, 72.0, '"Who are you?"')]


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = parse_episode(make_script(1, LINES), "episode-001.txt")
        cls.index = BlockIndex([cls.ep])
        cls.rules = Rules()
        cls.ids = {b.text: b.id for b in cls.ep.blocks}
        cls.slap = cls.ids["Vera slaps Mara across the face."]
        cls.line6 = cls.ids['"Line number 6 of the argument."']   # ends at 34 s

    def promo(self, **over):
        p = {"title": "t", "segments": [{"start_id": self.slap, "end_id": self.line6}],
             "hook_id": self.slap, "hook_quote": "slaps Mara across the face",
             "cliffhanger_id": self.line6, "cliffhanger_quote": "Line number 6",
             "hook_type": "slap", "why": "a slap", "scores": {"hook": 5, "cliffhanger": 4, "standalone": 4},
             "context_needed": ""}
        p.update(over)
        return validate_promo(p, self.index, {1}, self.rules)

    def codes(self, issues):
        return {i.code for i in issues if i.severity == "error"}

    def test_parser(self):
        self.assertEqual(self.ep.number, 1)
        self.assertEqual(self.ep.blocks[0].kind, "CAMERA")
        last = self.ep.blocks[-1]
        self.assertEqual((last.speaker, last.direction), ("STRANGER", "O.S., Menacing"))
        self.assertEqual(self.ep.duration, 72.0)

    def test_valid_promo_passes_and_timestamps_come_from_script(self):
        p, issues = self.promo()
        self.assertIsNotNone(p, issues)
        seg = p["segments"][0]
        self.assertEqual(seg["start"], "00:00.00")   # snapped from 1.0 s to the camera cut at 0.0 s
        self.assertEqual(seg["end"], "00:34.00")
        self.assertEqual(p["hook"]["script_text"], "Vera slaps Mara across the face.")

    def test_hallucinated_id_rejected(self):
        p, issues = self.promo(segments=[{"start_id": self.slap, "end_id": "E01-999"}])
        self.assertIsNone(p)
        self.assertIn("unknown_line_id", self.codes(issues))

    def test_invented_quote_rejected(self):
        p, issues = self.promo(hook_quote="punches the king in the face")
        self.assertIsNone(p)
        self.assertIn("hook_quote_not_verbatim", self.codes(issues))

    def test_short_whole_line_quote_ok(self):
        who = self.ids['"Who are you?"']
        start = self.ids['"Line number 8 of the argument."']   # 40 s -> 72 s
        p, issues = self.promo(segments=[{"start_id": start, "end_id": who}], hook_id=start,
                               hook_quote="Line number 8", cliffhanger_id=who, cliffhanger_quote="Who are you?")
        self.assertIsNotNone(p, issues)

    def test_too_long_rejected(self):
        end = self.ids['"Line number 13 of the argument."']   # ends at 69 s
        p, issues = self.promo(segments=[{"start_id": self.slap, "end_id": end}], cliffhanger_id=end,
                               cliffhanger_quote="Line number 13")
        self.assertIn("duration_too_long", self.codes(issues))

    def test_too_short_rejected(self):
        end = self.ids['"Line number 2 of the argument."']    # ends at 14 s
        p, issues = self.promo(segments=[{"start_id": self.slap, "end_id": end}], cliffhanger_id=end,
                               cliffhanger_quote="Line number 2")
        self.assertIn("duration_too_short", self.codes(issues))

    def test_late_hook_rejected(self):
        hook = self.ids['"Line number 2 of the argument."']   # starts 10 s into the segment
        p, issues = self.promo(hook_id=hook, hook_quote="Line number 2")
        self.assertIn("hook_too_late", self.codes(issues))

    def test_cliffhanger_must_be_at_end(self):
        early = self.ids['"Line number 3 of the argument."']  # ends 19 s, clip ends 34 s
        p, issues = self.promo(cliffhanger_id=early, cliffhanger_quote="Line number 3")
        self.assertIn("cliffhanger_too_early", self.codes(issues))

    def test_episode_not_allowed(self):
        p, issues = validate_promo({"segments": [{"start_id": self.slap, "end_id": self.line6}]},
                                   self.index, {2}, self.rules)
        self.assertIn("id_outside_allowed_episodes", self.codes(issues))

    def test_bad_scores_rejected(self):
        p, issues = self.promo(scores={"hook": 9, "cliffhanger": "high", "standalone": 3})
        self.assertIn("bad_scores", self.codes(issues))

    def test_json_parsing(self):
        self.assertEqual(parse_llm_json('```json\n{"promos": []}\n```'), ([], None))
        self.assertEqual(parse_llm_json('Sure! {"promos": [{"a": 1}]} hope it helps')[0], [{"a": 1}])
        self.assertIsNotNone(parse_llm_json("not json")[1])
        self.assertIsNotNone(parse_llm_json('{"clips": []}')[1])

    def test_ranker_removes_duplicates(self):
        a, _ = self.promo()
        b, _ = self.promo(scores={"hook": 3, "cliffhanger": 3, "standalone": 3})
        a["promo_id"], b["promo_id"] = "E01-P1", "E01-P2"
        for p in (a, b):
            p["source"] = {"provider": "t", "model": "t"}
        ranked = rank([a, b], RankWeights())
        self.assertEqual(ranked[0]["promo_id"], "E01-P1")
        self.assertEqual(ranked[1]["duplicate_of"], "E01-P1")


class FragmentAndDedupTests(unittest.TestCase):
    def test_fragment_detection(self):
        from signals import is_fragment
        for t in ['"...and then he left town."', '"Because she lied to you."',
                  '"Worry? You fell."', '"[gasping] but why?"', '"Tired already?"']:
            self.assertTrue(is_fragment(t), t)
        for t in ['"You should leave now, Mara."', '"Who are you?"', '"How dare you!"',
                  '"Open the door!"', '"[gasping] Give it back!"', '"That is a lie."']:
            self.assertFalse(is_fragment(t), t)

    def test_fragment_hook_is_rejected(self):
        lines = [("TOM", 0.0, 3.0, '"Why are you leaving?"'),
                 ("VERA", 3.5, 6.0, '"Because she lied to you."')]
        lines += [("TOM", t, t + 4.0, f'"Filler line {i} here."') for i, t in enumerate(range(6, 40, 5), 1)]
        ep = parse_episode(make_script(1, lines), "episode-001.txt")
        ids = {b.text: b.id for b in ep.blocks}
        frag, end = ids['"Because she lied to you."'], ids['"Filler line 6 here."']
        p = {"segments": [{"start_id": frag, "end_id": end}], "hook_id": frag, "hook_quote": "she lied to you",
             "cliffhanger_id": end, "cliffhanger_quote": "Filler line 6", "scores": {"hook": 5, "cliffhanger": 5, "standalone": 5}}
        promo, issues = validate_promo(p, BlockIndex([ep]), {1}, Rules())
        self.assertIsNone(promo)
        msg = next(i.message for i in issues if i.code == "hook_is_fragment")
        self.assertIn(ids['"Why are you leaving?"'], msg)   # tells the model which line to start on

    def _p(self, pid, start, end, llm_hook, score, fragment=False):
        return {"promo_id": pid, "episodes": [1], "total_duration_s": end - start,
                "segments": [{"episode": 1, "start_s": start, "end_s": end}],
                "llm_scores": {"hook": llm_hook, "cliffhanger": 3, "standalone": 3},
                "rule_signals": {"hook": 1, "cliff": 1}, "score": score, "hook_type": "other",
                "hook": {"offset_s": 0}, "source": {},
                "warnings": [{"code": "hook_is_fragment", "message": ""}] if fragment else []}

    def test_dedup_keeps_stronger_hook_not_longer_clip(self):
        from ranker import rank
        import ranker
        a = self._p("A", 40, 100, llm_hook=3, score=0)   # long clip, weaker opening
        b = self._p("B", 39, 75, llm_hook=5, score=0)    # same scene, stronger opening
        c = self._p("C", 45, 95, llm_hook=5, score=0, fragment=True)  # strong but opens on a fragment
        orig = ranker.score_promo
        ranker.score_promo = lambda p, w: {"score": {"A": 80, "C": 75, "B": 70}[p["promo_id"]], "score_breakdown": {"funnel": 1}}
        try:
            out = {p["promo_id"]: p for p in rank([a, b, c], RankWeights())}
        finally:
            ranker.score_promo = orig
        self.assertIsNone(out["B"]["duplicate_of"])
        self.assertEqual(out["A"]["duplicate_of"], "B")
        self.assertEqual(out["C"]["duplicate_of"], "B")


class RerankTests(unittest.TestCase):
    def test_parse_rerank_checks_ids(self):
        from reranker import parse_rerank
        ids = ["A", "B", "C", "D"]
        order, notes, err, missing = parse_rerank('{"order": ["C", "X", "C", "[A]", "D"], "notes": {"C": "loud", "X": "?"}}', ids)
        self.assertIsNone(err)
        self.assertEqual(order, ["C", "A", "D", "B"])   # unknown X dropped, duplicate C ignored, missing B appended
        self.assertEqual((notes, missing), ({"C": "loud"}, 1))
        self.assertIsNotNone(parse_rerank('{"order": ["A"]}', ids)[2])        # under half valid -> rejected
        self.assertIsNotNone(parse_rerank("sorry, I can't", ids)[2])

    def test_two_passes_cancel_position_bias(self):
        import tempfile
        from llm_client import LLMClient
        from reranker import rerank

        class AlwaysFirstShown(LLMClient):  # a maximally position-biased model
            provider, default_model = "fake", "fake"

            def _call(self, system, user):
                import re as _re
                return json.dumps({"order": _re.findall(r"^\[(\w+)\]", user, _re.M)}), {}

        import json
        cands = [{"promo_id": i} for i in ("A", "B", "C")]
        import reranker
        orig = reranker.build_rerank_prompt
        reranker.build_rerank_prompt = lambda cs, t, ix: "\n".join(f"[{c['promo_id']}]" for c in cs)
        try:
            with tempfile.TemporaryDirectory() as d:
                rr = rerank(cands, None, AlwaysFirstShown(use_cache=False, cache_dir=None), "T", Path(d))
        finally:
            reranker.build_rerank_prompt = orig
        self.assertTrue(rr["applied"])
        self.assertEqual(rr["avg_pos"], {"A": 2, "B": 2, "C": 2})   # bias averages out to a tie
        self.assertEqual(rr["pass_disagreement"], round((2 + 0 + 2) / 3, 2))

    def test_rank_follows_rerank_order(self):
        from ranker import rank
        mk = FragmentAndDedupTests._p
        a, b = mk(None, "A", 0, 30, 5, 0), mk(None, "B", 100, 130, 5, 0)
        a["episodes"], b["episodes"] = [1], [2]
        for p in (a, b):
            p["segments"][0]["episode"] = p["episodes"][0]
        rr = {"avg_pos": {"A": 2.0, "B": 1.0}, "notes": {"B": "louder"}}
        out = rank([a, b], RankWeights(rerank_weight=1.0), rerank=rr)       # side-by-side only
        self.assertEqual([p["promo_id"] for p in out], ["B", "A"])
        self.assertIn("side-by-side #1", out[0]["rank_reason"])
        out = rank([a, b], RankWeights(rerank_weight=0.0), rerank=rr)       # scores only (tie -> id order)
        self.assertEqual([p["promo_id"] for p in out], ["A", "B"])
        a["llm_scores"]["hook"] = 1                                          # A now clearly worse on score
        out = rank([a, b], RankWeights(rerank_weight=0.5), rerank=rr)       # both agree on B
        self.assertEqual([p["promo_id"] for p in out], ["B", "A"])


class RepairLoopTest(unittest.TestCase):
    def test_bad_answer_is_sent_back_and_fixed(self):
        import json
        import tempfile
        from config import PipelineConfig
        from llm_client import LLMClient
        from processor import process_episode

        ep = parse_episode(make_script(1, LINES), "episode-001.txt")
        ids = {b.text: b.id for b in ep.blocks}
        slap, end = ids["Vera slaps Mara across the face."], ids['"Line number 6 of the argument."']
        good = {"title": "t", "segments": [{"start_id": slap, "end_id": end}], "hook_id": slap,
                "hook_quote": "slaps Mara", "cliffhanger_id": end, "cliffhanger_quote": "Line number 6",
                "hook_type": "slap", "why": "w", "scores": {"hook": 5, "cliffhanger": 4, "standalone": 4}}
        bad = {**good, "hook_quote": "stabs the king", "segments": [{"start_id": slap, "end_id": "E01-777"}]}

        class Fake(LLMClient):
            provider, default_model = "fake", "fake"
            prompts = []

            def _call(self, system, user):
                self.prompts.append(user)
                return json.dumps({"promos": [good if "REJECTED" in user else bad]}), {}

        client = Fake(use_cache=False, cache_dir=None)
        with tempfile.TemporaryDirectory() as d:
            res = process_episode(ep, None, BlockIndex([ep]), client, PipelineConfig(), "T", Path(d))
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("unknown_line_id", client.prompts[1])        # validator errors fed back to the model
        self.assertEqual(len(res["accepted"]), 1)
        self.assertTrue(res["accepted"][0]["repaired"])
        self.assertEqual(res["rejected"][0]["final"], False)


if __name__ == "__main__":
    unittest.main()
