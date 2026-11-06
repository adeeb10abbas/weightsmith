# Proposal mode: nonadaptive

Create p1, p2, and p3 independently from base, using only the common initial base evidence and permitted base activations. Commit all three exact candidates and their hypotheses before evaluating any candidate. They remain immutable after commitment. Only after all three commitments may you call wsp_eval for each. You may inspect returned results, but cannot create or alter another proposal. The same operator gate chooses the final candidate, or base. Never request held-out results.
