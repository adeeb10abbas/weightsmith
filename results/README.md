# Results

This page lists where every published result lives and what a submission must include. The [leaderboard](../LEADERBOARD.md) summarizes the entries. The [report](../study/blog/failure-levels.md) is the source of truth for every number.

## Where the files are

| What | Where |
|---|---|
| Every paired comparison from our cohort, plus Astra reference rows | [`study/results/opus/comparisons.json`](../study/results/opus/comparisons.json) |
| Held-out scores, one file per `heldout.py` run | [`study/results/opus/heldout/`](../study/results/opus/heldout/) |
| Evidence record: protocol, budgets, costs and every proposal | [`study/results/opus/EVIDENCE.md`](../study/results/opus/EVIDENCE.md) |
| Scripts that build the language-binding edit and its variants | [`study/results/opus/ctb/`](../study/results/opus/ctb/README.md) |
| Claude under the Astra protocol: brief, notebook and final answer | [`study/results/opus/claude-astra-protocol-*.md`](../study/results/opus/claude-astra-protocol-FINAL.md) |
| Early Claude runs (banana then cube, cube then banana) | [`docs/results/`](../docs/results/README.md) |
| Astra study: results and limitations (frozen) | [`study/RESULTS.md`](../study/RESULTS.md) |
| Astra measurement records (frozen; do not edit) | [`study/results/data/`](../study/results/data/) |
| Astra study manuscript | [`study/paper/`](../study/paper/README.md) |
| Figures and plotting scripts | `study/results/figures/`, `study/blog/figures/`, [`study/results/scripts/`](../study/results/scripts/) |
| Large downloads (Astra raw archives and videos) | [`study/ARTIFACTS.md`](../study/ARTIFACTS.md) |
| Raw episodes and activation captures from our cohort | Not in Git; they stay on the reference GPU host |

To redraw the summary figure from `comparisons.json`, run `python3 study/results/scripts/plot_opus_forest.py`. It writes to `study/results/figures/` and `study/blog/figures/`.

## File formats

**`comparisons.json`** has one entry in `rows` per paired comparison:

| Field | Meaning |
|---|---|
| `level`, `task`, `label` | Failure level, task, and what was compared |
| `editor`, `type` | Who made the change (Claude Opus, GPT-6 Astra or operator), and its kind: `weight` (practice or development), `heldout`, `prompt`, `runtime` or `identity` |
| `values_changed` | bf16 values changed, or `null` if not recorded |
| `n`, `base`, `cand` | Paired slots, and successes in each arm |
| `fixed`, `broke` | Slots where the base failed and the candidate succeeded, and the reverse |

**`heldout/<project>__<candidate>-<Task>-<first slot>-<n>.json`** is the `results.json` written by `weightsmith/heldout.py`:
- `summary` holds:
  - the candidate name and the fingerprint of its frozen edits;
  - the task, the slot range and the run order;
  - the base and candidate successes, with Wilson 95% intervals;
  - the fixed and broke counts and `mcnemar_exact_p`;
  - the crashed slots and the mean steps to success.
- `per_slot` maps each slot to `[base_success, candidate_success]`.

## Submitting a result

Put each run in `results/submissions/<YYYY-MM-DD>-<editor>-<task>/`:

| File | Contents |
|---|---|
| `README.md` | A summary of the run (see below) |
| `config.json` | The project config |
| `FINAL.md`, `NOTEBOOK.md` | The editor's own files, verbatim |
| `heldout-<Task>-<first slot>-<n>.json` | `heldout.py`'s `results.json`, unmodified. Omit it if the final answer was `base`. |
| `edit/` | The `owsp py` scripts that build the candidate from base, plus the `owsp diff` output. Do not commit weight arrays; link large files with their sha256. |
| `practice.json` (optional) | Every evaluated candidate, as rows with the `comparisons.json` fields |

The run's `README.md` states:
- the editor model and version, its effort setting, and the driver and protocol used;
- the task and any hints given (such as a `PRIOR_FINDINGS.md`);
- the budget used: practice episodes, activation items, wall time and API cost;
- the final answer and the failure level targeted;
- any deviation from [BENCHMARK.md](../BENCHMARK.md);
- who built the edit: an autonomous editor or an operator.

Then add your row to [LEADERBOARD.md](../LEADERBOARD.md). A maintainer may ask for the edit so they can re-score it.

The runtime's directory names (`data/`, `heldout/`, `notes/`, `frames/`, `logs/`, `state/`, `cands/`, `acts/`, `episodes/`) are git-ignored at any depth. So are `*.npy` and `*.npz` files, and images outside a `figures/` folder. Use the flat file names above.
