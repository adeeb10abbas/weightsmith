# ruff: noqa: F821  (episodes: helper of the owsp py sandbox prelude)
# Per-slot outcomes of base and one or more candidates on their shared practice slots, as CSV for `weightsmith stats`:
#   bin/owsp py base examples/cube_then_banana/paired_csv.py > paired.csv
#   weightsmith stats paired.csv --arms base,ctb-imgl8
# A slot run more than once keeps its first episode, as a paired comparison should.
CANDS = ["ctb-imgl8", "ctb-first"]

first = {}
for r in episodes():
    first.setdefault((r["cand"], r["slot"]), r["success"])
arms = ["base"] + [c for c in CANDS if any(k[0] == c for k in first)]
slots = sorted({s for c, s in first if c == "base"}, key=lambda s: int(s.lstrip("s")))
print("slot," + ",".join(arms))
for s in slots:
    print(s + "," + ",".join("" if (a, s) not in first else str(int(first[(a, s)])) for a in arms))
