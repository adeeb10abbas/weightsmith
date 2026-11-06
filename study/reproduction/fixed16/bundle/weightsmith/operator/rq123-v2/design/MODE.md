# Proposal mode: adaptive

Create p1 from base. Commit it, evaluate it, and inspect the returned outcomes and gate. Only then create p2, starting from base or the currently accepted incumbent. Repeat for p3. You may inspect newly observed failures before the next proposal. Each candidate still has the same cumulative 128-row/two-layer limit relative to the original base. The gate may keep the earlier candidate or base. Three round-end incumbent states are retained for later operator-only scoring. Never request those held-out results.
