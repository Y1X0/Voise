import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2 as S  # noqa: E402


def test_turns_never_split_and_sets_disjoint():
    rng = np.random.default_rng(0)
    turns = [[0, 1, 2], [3], [4, 5], [6, 7, 8, 9], [10, 11]]
    for _ in range(200):
        s = S.try_split(turns, 3, rng)
        if s is None:
            continue
        a, b = map(set, s)
        assert not a & b
        for t in turns:
            assert set(t) <= a or set(t) <= b
        sb = S.split_sets(turns, 3, rng)
        assert len(sb[0]) == 3 and len(sb[1]) == 3 and not set(sb[0]) & set(sb[1])


def test_single_turn_person_not_evaluable():
    assert S.evaluable([[0, 1, 2, 3, 4, 5]], 2, np.random.default_rng(0)) == 0


def test_assign_turns_gap():
    rows = [dict(person="a", episode="m1", t=t) for t in (0, 50, 200, 210, 1000)] + [dict(person="o", episode="x", t=None)] * 1
    rows = [dict(r) for r in rows]
    S.assign_turns(rows)
    assert [r["turn"] for r in rows[:5]] == ["a#0", "a#0", "a#1", "a#1", "a#2"]
    assert rows[5]["turn"] == "o#clip0"


def test_icc_pairs_perfect():
    assert abs(S.icc_pairs(np.arange(10.0), np.arange(10.0)) - 1) < 1e-9
