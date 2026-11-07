"""Recipe 10: per-slot outcomes of several arms, as a CSV for ``weightsmith stats``.

When: after closed-loop practice runs (``owsp eval``) of the unedited policy and one or more candidates on the same
slots, to read the comparison properly (paired outcomes, exact McNemar, intervals). Held-out runs already write a
``results.json`` that ``weightsmith stats`` reads directly. See docs/COOKBOOK.md, section 6.

Run (read-only):  bin/owsp py base examples/recipes/10_paired_outcomes.py > practice.csv
                  weightsmith stats practice.csv --arms base,v2mi-f

Arms are the labels in the project's episode index: candidate names, and ``NAME@pXXXXXX`` for runs with a
diagnostic instruction override (``owsp eval NAME --prompt ...``; the label hashes the wording). Lines starting
with ``#`` are comments, which ``weightsmith stats`` skips.

Check it worked: the counts in the comment lines match what ``owsp episodes`` prints. On cube-then-banana,
``weightsmith stats`` on this CSV gives 3/32 -> 25/32 for the distilled edit (22 fixed, 0 broken, exact
p = 4.8e-7) and 3/32 -> 22/32 for the explicit wording with unchanged weights (20 fixed, 1 broken): the numbers in
Section 2 of the paper.
"""
SETTINGS = {
    "arms": ["base", "v2mi-f", "abl-imgl8", "base@p89134f"],  # episode-index labels; the first is the reference
    "slots": None,      # e.g. [20000, 20031]: only this inclusive range; None = every slot any arm ran
    "run": "first",     # a slot evaluated twice for one arm: use the "first" or "last" run
}


def main(cfg=SETTINGS):
    rows = episodes()
    table = {}
    for arm in cfg["arms"]:
        runs = [r for r in rows if r["cand"] == arm]
        if not runs:
            labels = sorted({r["cand"] for r in rows})
            raise SystemExit(f"no recorded episodes for arm {arm!r}; labels in this project: {labels}")
        per_slot = {}
        for r in runs:  # the index is in recording order
            slot = int(str(r["slot"]).lstrip("s"))
            if cfg["run"] == "last" or slot not in per_slot:
                per_slot[slot] = bool(r["success"])
        table[arm] = per_slot
    slots = sorted(set().union(*table.values()))
    if cfg["slots"]:
        lo, hi = cfg["slots"]
        slots = [s for s in slots if lo <= s <= hi]
    ref = cfg["arms"][0]
    for arm in cfg["arms"]:
        both = [s for s in slots if s in table[arm] and s in table[ref]]
        fixed = sum(table[arm][s] and not table[ref][s] for s in both)
        broke = sum(table[ref][s] and not table[arm][s] for s in both)
        print(f"# {arm}: {sum(table[arm].get(s, False) for s in slots)}/{sum(s in table[arm] for s in slots)} "
              + (f"(vs {ref} on {len(both)} shared slots: fixed {fixed}, broke {broke})" if arm != ref else ""))
    print("slot," + ",".join(cfg["arms"]))
    for s in slots:
        print(f"{s}," + ",".join("" if s not in table[a] else str(int(table[a][s])) for a in cfg["arms"]))
    return table


if __name__ == "__main__":
    main()
