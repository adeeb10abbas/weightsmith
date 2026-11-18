# Frozen repair-comparison analysis

This reporter consumes the frozen protocol and one completed held-out worker root
per case. It does not launch policies, train adapters, choose checkpoints, or
change any outcome. Run from `/home/ali/weightsmith-iros2027`:

```bash
python3 study/iros2027/v2/rss_repair_comparison/analysis/report.py --protocol study/iros2027/v2/rss_repair_comparison/protocol.json --temporal-root /absolute/temporal-heldout-worker --spatial-root /absolute/spatial-heldout-worker --output projects/rss-repair-comparison/final-analysis
```

The output directory must be new. Full analysis requires all **640** registered
policy episodes: two cases, five cells, four conditions, eight layouts, and two
policy-noise replicates. Episodes remain separate even when conditions happen
to execute the same instruction. No automatic deduplication or outcome-based
exclusion occurs.

The reporter verifies:

- The worker froze the exact supplied protocol and completed every planned batch.
- Every completed receipt matches its episode-file hash, identity, count, and
  success count. Referenced metric files also match their recorded hashes.
- The unedited comparators load no workspace, and each adapted workspace's
  manifest and edited tensors appear in the worker's frozen identity manifest.
- Every case/cell/condition/slot is unique, belongs to the registered slot set,
  and uses the expected instruction and strict selected-goal predicate.
- Every cell/condition has eight layouts with two replicates each, with identical
  layout IDs and first policy sampling seeds across paired conditions.

Observation-file hashes remain attached to the episode records but the large raw
observation files are not reread. The worker's frozen identities attest model and
runtime inputs; this reporter does not reload or rehash the large weight files.

For an unfinished run, add `--allow-partial`. This option always produces only
`status.json` and `inventory.csv`, including verified episode counts and integrity
errors. It never creates success tables or intervals. Without that option,
incomplete or invalid inputs produce the same status artifacts and a failing exit
code; there is no partial result presented as a completed comparison.

## Outputs and interpretation

- `primary.csv`: strict selected-goal final success for every case/cell/condition.
- `stock_compatible.csv`: secondary stock-compatible final outcomes, kept apart
  from the primary predicate.
- `paired.csv`: fixed, broken, both-successful, and both-failed counts for each
  comparator against base, plus closed form against saved rewrites and LoRA.
  Rate differences are fractions, not percentage points.
- `collisions.csv`: other goals achieved at the selected endpoint, simultaneous
  selected/competing final predicates, and competing-only final outcomes.
  Ever-achieved predicates are separate columns. Two ever-achieved goals may
  occur at different times, or one may be a necessary intermediate step; these
  columns do not establish a simultaneous collision or a semantic violation.
- `pairing_diagnostics.csv`: initial base/wrist camera-hash equality for every
  compared episode pair. Mismatches are counted, retained, and never silently
  excluded. A mismatch warrants investigation before attributing the entire
  paired difference to the repair.
- `exact_cache_derived.csv` and `exact_cache_rows.json`: an exact-string cache
  comparator derived by reusing the matching base or semantic-rewrite episode
  with the identical effective instruction. Each row identifies its source.
  These 160 reused records add **zero new independent policy episodes** and no
  additional statistical power. They are not 160 additional robot executions.
- `report.json`: all tables, provenance hashes, verification status, statistical
  settings, and evidence accounting.

Intervals resample the eight layout clusters with replacement, retaining both
noise replicates together. Paired intervals resample layout-level paired
differences. Defaults are 20,000 draws, seed 17, and the 2.5th/97.5th percentiles.
These are exploratory intervals from only eight layout clusters. Every row
carries the small-cluster warning, and zero-width intervals receive an explicit
degeneracy warning. The report contains no p values, equivalence tests, or claim
that these eight layouts represent a deployment population. In particular,
zero observed regressions does not establish preservation equivalence.

## Verification

```bash
python3 -m unittest discover -s study/iros2027/v2/rss_repair_comparison/analysis -p 'test_*.py' -v
```

Nine tests generate synthetic 640-episode fixtures. They check exact-cache reuse
accounting, paired improvement/regression counts, incomplete-run refusal, receipt
and metric corruption, duplicate slots, seed mismatches, camera-difference
retention, unfrozen adapters, protocol changes, degenerate intervals, and separation
of final versus ever-achieved predicates, and full versus status-only CLI output. Synthetic fixtures are not policy
results. No real held-out episode has been consumed to develop this reporter.
