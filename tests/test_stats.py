"""Tests for weightsmith.stats and the `weightsmith stats` command (CPU only).

The expected values are the paper's (https://ali-adeeb.com/weightsmith/paper.pdf): Section 4, Table 3 and
Appendices B, C, F. Inputs come from the repository's result files where they exist.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from weightsmith import cli  # noqa: E402
from weightsmith import stats as S  # noqa: E402

HELDOUT = REPO / "study" / "results" / "opus" / "heldout"
V2MIF = HELDOUT / "cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json"
CONTROL = REPO / "study" / "results" / "opus" / "controls" / "ctl1-90000-64.results.json"
COMPARISONS = REPO / "study" / "results" / "opus" / "comparisons.json"

# Controlled study, cube then banana: the operator's held-out run of the blind editor's p1 on slots 91008-91071
# (per_slot of heldout/p1-RubiksCubeThenBananaTask-91008-64/results.json on the GPU host), base then candidate.
CTRL_BASE = "0000100010000000010001010101000000000000000000001000111000000000"
CTRL_CAND = "1001101000110001010000010000001001011011110100101110001100101010"


def ctrl_results() -> dict:
    per_slot = {str(91008 + i): [b == "1", c == "1"] for i, (b, c) in enumerate(zip(CTRL_BASE, CTRL_CAND, strict=True))}
    summary = {"cand": "p1", "task": "RubiksCubeThenBananaTask", "slots": [91008, 91071],
               "order": "first half: base then cand; second half: cand then base", "paired_n": 64,
               "base_success": 11, "cand_success": 27, "base_wilson95": [0.099, 0.282],
               "cand_wilson95": [0.309, 0.544], "fixed": 22, "broke": 6, "mcnemar_exact_p": 0.0037191659212112427}
    return {"summary": summary, "per_slot": per_slot}


def pts(x: float) -> float:
    return round(100 * x, 1)


# ------------------------------------------------------------------ the paper's numbers
def test_mcnemar_reproduces_the_headline_result():
    p = S.mcnemar_exact(81, 2)  # 27/128 -> 106/128 on held-out seeds
    assert f"{p:.1e}" == "7.2e-22"
    assert p == json.loads(V2MIF.read_text())["summary"]["mcnemar_exact_p"]  # same as heldout.py
    assert S.mcnemar_exact(22, 6) == pytest.approx(0.0037191659, rel=1e-8)  # controlled study, cube then banana
    assert f"{S.mcnemar_exact(86, 3):.1e}" == "3.8e-22"  # the 0.16% variant
    assert round(S.mcnemar_exact(6, 10), 2) == 0.45  # random-direction control
    assert S.mcnemar_exact(0, 0) == 1.0 and S.mcnemar_exact(37, 37) == 1.0


def test_mde_matches_the_paper_table():
    got = [round(100 * S.mde(n, 0.3)) for n in (32, 64, 128, 256)]
    assert got == [27, 19, 14, 10]  # Table 5 and Fig. 4
    assert S.mde(64, 0.3) == pytest.approx((1.959964 + 0.841621) * math.sqrt(0.3 / 64), rel=1e-6)
    # the exact McNemar test needs about the same: the normal approximation is fine at these sizes
    exact = [round(100 * S.mde(n, 0.3, exact=True), 1) for n in (32, 64, 128, 256)]
    assert all(abs(e - a) < 1.5 for e, a in zip(exact, (27.1, 19.2, 13.6, 9.6), strict=True))


def test_holm_over_the_seven_blind_editors():
    p = [S.mcnemar_exact(22, 6), S.mcnemar_exact(18, 9), S.mcnemar_exact(9, 6)] + [1.0] * 4  # 4 submitted base
    adj = S.holm(p)
    assert round(p[0], 4) == 0.0037 and round(adj[0], 3) == 0.026
    assert round(p[1], 2) == 0.12 and round(p[2], 2) == 0.61
    assert adj[1] > 0.05 and adj[2] == 1.0 and adj[3:] == [1.0] * 4


def test_wilson_matches_the_stored_intervals():
    for f in sorted(HELDOUT.glob("*.json")):
        s = json.loads(f.read_text())["summary"]
        for k, key in ((s["base_success"], "base_wilson95"), (s["cand_success"], "cand_wilson95")):
            assert S.wilson(k, s["paired_n"]) == pytest.approx(s[key], abs=6e-4), (f.name, key)
    assert [round(100 * x, 1) for x in S.wilson(11, 64)] == [9.9, 28.2]  # Appendix F
    assert [round(100 * x, 1) for x in S.wilson(27, 64)] == [30.9, 54.4]
    lo, hi = S.wilson(0, 64)
    assert lo == 0.0 and 0.05 < hi < 0.06
    with pytest.raises(ValueError):
        S.wilson(1, 0)


@pytest.mark.parametrize("n, base, cand, fixed, broke, want", [
    (128, 27, 106, 81, 2, (61.7, 52.3, 70.3)),   # distilled edit, held-out
    (64, 22, 44, 31, 9, (34.4, 17.2, 51.6)),     # transfer to eyeglasses
    (64, 8, 36, 33, 5, (43.8, 28.1, 59.4)),      # transfer to cube left of bowl
    (64, 18, 27, 18, 9, (14.1, -1.6, 29.7)),     # blind editor, eyeglasses
    (64, 8, 11, 9, 6, (4.7, -6.2, 17.2)),        # blind editor, mustard
    (64, 13, 9, 6, 10, (-6.2, -18.8, 6.2)),      # random-direction control
    (16, 14, 8, 0, 6, (-37.5, -62.5, -12.5)),    # unedited butter rerun
])
def test_paired_bootstrap_reproduces_the_paper(n, base, cand, fixed, broke, want):
    t = S.PairedTable.from_counts(n, base, cand, fixed, broke)
    lo, hi = S.paired_bootstrap(t)
    assert (pts(t.difference), pts(lo), pts(hi)) == want


def test_pooled_bootstrap_reproduces_the_gripper_row():
    rows = [r for r in json.loads(COMPARISONS.read_text())["rows"] if r["label"].startswith("gthr04")]
    tables = [S.PairedTable.from_counts(r["n"], r["base"], r["cand"], r["fixed"], r["broke"]) for r in rows]
    assert sum(t.n for t in tables) == 240 and sum(t.fixed for t in tables) == sum(t.broke for t in tables) == 37
    lo, hi = S.paired_bootstrap(tables)
    assert (pts(lo), pts(hi)) == (-7.1, 7.1)  # "Seven tasks pooled: gripper bias", Table 3


def test_compare_on_the_heldout_file_and_its_halves():
    arms = S.read_arms(V2MIF)
    c = S.compare(arms["base"], arms["cand"], cand_name="v2mi-f")
    assert (c.n, c.ref_success, c.cand_success, c.fixed, c.broke) == (128, 27, 106, 81, 2)
    assert (pts(c.difference), pts(c.difference_ci[0]), pts(c.difference_ci[1])) == (61.7, 52.3, 70.3)
    assert c.verdict == "better" and c.significant and round(100 * c.mde) == 14
    first, second = c.halves()  # Table 7: base first on 90000-90063, edit first on 90064-90127
    assert (first.ref_success, first.cand_success, first.fixed, first.broke) == (13, 50, 38, 1)
    assert (second.ref_success, second.cand_success, second.fixed, second.broke) == (14, 56, 43, 1)
    text = c.report()
    assert "27/128" in text and "106/128" in text and "7.2e-22" in text and "+52.3 to +70.3" in text


def test_compare_across_files_reproduces_the_random_control():
    base_a = S.read_arms(V2MIF)["base"]  # first unedited run on 90000-90127
    ctl = S.read_arms(CONTROL)  # control arms on 90000-90063
    rand = S.compare(base_a, ctl["rand"], cand_name="rand")
    assert (rand.n, rand.ref_success, rand.cand_success, rand.fixed, rand.broke) == (64, 13, 9, 6, 10)
    assert round(rand.p_value, 2) == 0.45 and rand.verdict == "no detectable difference"
    assert len(rand.dropped) == 64  # 90064-90127 ran in one arm only
    teacher = S.compare(base_a, ctl["teacher"])  # the explicit wording, unedited weights
    assert (teacher.cand_success, teacher.fixed, teacher.broke) == (57, 45, 1)
    assert f"{teacher.p_value:.1e}" == "1.3e-12"


def test_compare_sequences_and_dropping():
    c = S.compare([1, 0, 0, 1], [1, 1, 0, 0])
    assert c.table.cells == (1, 1, 1, 1) and c.p_value == 1.0 and c.difference == 0
    c = S.compare({"s1": True, 2: None, 3: False}, {1: "success", 2: "fail", 4: 1})
    assert c.slots == (1,) and c.dropped == (2, 3, 4)
    with pytest.raises(ValueError):
        S.compare({1: True}, {2: True})
    with pytest.raises(ValueError):
        S.compare([1, 0], [1])


def test_paired_table_and_outcome_parsing():
    t = S.PairedTable.from_counts(64, 11, 27, 22, 6)
    assert t.cells == (5, 22, 6, 31) and t.discordance == 28 / 64 and t.difference == 16 / 64
    with pytest.raises(ValueError):
        S.PairedTable.from_counts(64, 11, 27, 20, 6)  # inconsistent
    with pytest.raises(ValueError):
        S.PairedTable(0, 0, 0, 0)
    assert [S.parse_outcome(v) for v in (True, 0, "1", "Yes", "SUCCESS", "fail", "", None)] == \
        [True, False, True, True, True, False, None, None]
    assert S.parse_outcome({"ok": True, "success": True}) is True
    assert S.parse_outcome({"ok": False, "success": True}) is None
    with pytest.raises(ValueError):
        S.parse_outcome("maybe")
    assert [S.normalize_slot(x) for x in (90000, "90000", "s20003", "abc")] == [90000, 90000, 20003, "abc"]


def test_power_and_pairs_needed():
    assert S.mcnemar_power(64, 0.15, 0.15) < 0.05  # no effect: the exact test keeps its size
    p1, p2 = S.mcnemar_power(64, 0.2, 0.1), S.mcnemar_power(64, 0.25, 0.05)
    assert p1 < p2 < S.mcnemar_power(128, 0.25, 0.05)
    assert S.pairs_needed(S.mde(64, 0.3)) == 64
    assert S.pairs_needed(0.25) == 38
    assert math.isnan(S.mde(8, 0.05, exact=True))  # too few discordant pairs ever to reach p < 0.05


def test_validate_slots():
    assert S.validate_slots(91008, 64) == (range(91008, 91040), range(91040, 91072))
    with pytest.raises(ValueError, match="multiple of 16"):
        S.validate_slots(91000, 64)  # the controlled study's failed first launch
    with pytest.raises(ValueError, match="multiple of 32"):
        S.validate_slots(90000, 48)
    assert S.validate_slots(20000, 48, halves=False)[0] == range(20000, 20024)


def test_read_csv_and_json_arms(tmp_path):
    f = tmp_path / "out.csv"
    f.write_text("slot,base,v2mi-f,teacher\ns20000,0,1,1\ns20001,1,1,\ns20002,false,true,true\n")
    a = S.read_arm_file(f)
    assert a.kind == "csv" and list(a.arms) == ["base", "v2mi-f", "teacher"]
    assert a.arms["teacher"] == {20000: True, 20001: None, 20002: True}
    with pytest.raises(ValueError, match="name the two"):
        a.default_pair()
    g = tmp_path / "arms.json"
    g.write_text(json.dumps({"base": {"1": 1, "2": 0}, "cand": {"1": 1, "2": 1}}))
    b = S.read_arm_file(g)
    assert b.kind == "heldout" and b.default_pair() == ("base", "cand")
    bad = tmp_path / "bad.csv"
    bad.write_text("slot,a,b\n1,1,maybe\n")
    with pytest.raises(ValueError, match="line 2"):
        S.read_arm_file(bad)


# ------------------------------------------------------------------ the command
def run_cli(capsys, *args) -> tuple[int, str, str]:
    code = cli.main(["stats", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_heldout_file_with_halves(tmp_path, capsys):
    f = tmp_path / "results.json"
    f.write_text(json.dumps(ctrl_results()))
    code, out, _ = run_cli(capsys, f, "--halves")
    assert code == 0
    assert "base vs p1: 64 paired slots (91008-91071)" in out
    assert "11/64" in out and "27/64" in out and "9.9% to 28.2%" in out and "30.9% to 54.4%" in out
    assert "both 5, fixed 22, broke 6, neither 31" in out and "exact p = 0.0037" in out
    assert "first half (base ran first): 7/32 -> 10/32" in out  # Appendix F: 7 -> 10 and 4 -> 17
    assert "second half (p1 ran first): 4/32 -> 17/32" in out
    assert "matches the file's own summary" in out and "p1 better than base" in out


def test_cli_holm_family_and_json(tmp_path, capsys):
    f = tmp_path / "results.json"
    f.write_text(json.dumps(ctrl_results()))
    code, out, _ = run_cli(capsys, f, V2MIF, "--family-size", 7)
    assert code == 0 and "Holm correction over 7 comparisons (5 not given here count as p = 1)" in out
    code, out, _ = run_cli(capsys, f, "--family-size", 7, "--json")
    d = json.loads(out)
    assert d["family_size"] == 7
    c = d["comparisons"][0]
    assert (c["fixed"], c["broke"], c["n"]) == (22, 6, 64) and round(c["holm_p"], 3) == 0.026
    assert c["summary_check"].startswith("matches")


def test_cli_across_files_and_csv(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "--ref", f"{V2MIF}:base", "--cand", f"{CONTROL}:rand")
    assert code == 0 and "base vs rand: 64 paired slots (90000-90063)" in out
    assert "fixed 6, broke 10" in out and "exact p = 0.45" in out and "left out 64 slot(s)" in out
    code, out, _ = run_cli(capsys, "--ref", f"{V2MIF}:v2mi-f", "--cand", f"{CONTROL}:teacher")
    assert code == 0 and "fixed 11, broke 4" in out and "exact p = 0.12" in out  # Appendix C: 4 and 11
    csv = tmp_path / "practice.csv"
    csv.write_text("slot,base,edit\n" + "".join(f"{20000 + i},{int(i < 3)},{int(i < 25)}\n" for i in range(32)))
    code, out, _ = run_cli(capsys, csv)
    assert code == 0 and "3/32" in out and "25/32" in out and "fixed 22, broke 0" in out
    assert "exact p = 4.8e-07" in out and "fewer than 64 pairs" in out  # Section 2: practice seeds


def test_cli_errors(tmp_path, capsys):
    code, _, err = run_cli(capsys)
    assert code == 1 and "give at least one results file" in err
    code, _, err = run_cli(capsys, "--ref", V2MIF)
    assert code == 1 and "--ref and --cand go together" in err
    code, _, err = run_cli(capsys, "--ref", f"{V2MIF}:nope", "--cand", f"{CONTROL}:rand")
    assert code == 1 and "no arm 'nope'" in err
    three = tmp_path / "three.csv"
    three.write_text("slot,a,b,c\n1,1,0,1\n")
    code, _, err = run_cli(capsys, three)
    assert code == 1 and "name the two to compare" in err
    code, out, _ = run_cli(capsys, three, "--arms", "a,c")
    assert code == 0 and "a vs c: 1 paired slots" in out
