# Bounded development-only LoRA selection

The first 200-step fit at learning rate 0.001 worsened development validation.
Before any held-out evaluation, the controller authorized two additional rates,
**0.00003 and 0.0001**, with the same checkpoint, shared observations, teacher
targets, rank 32, language layers 8–16, 200 steps, seed 17, zero weight decay, and
native flow-matching objective. The original fit remains immutable.

Run only after the controller confirms that the amended protocol and source
freeze are complete:

```bash
python3 study/iros2027/v2/rss_repair_comparison/lora/sweep.py --adaptation-root projects/rss-repair-comparison-20261117/temporal-adaptation --case temporal --gpu 0 --source-freeze /absolute/sweep-source-freeze.json
python3 study/iros2027/v2/rss_repair_comparison/lora/sweep.py --adaptation-root projects/rss-repair-comparison-20261117/spatial-adaptation --case spatial --gpu 1 --source-freeze /absolute/sweep-source-freeze.json
```

The source freeze is a path-to-SHA256 mapping, directly or under a `sources` key.
It must include this runner, `train_lora.py`, and the amended `protocol.json`.
The runner checks every provided source, the exact dataset and item files, and
the original candidate before and after each fit. Each child training command
uses the existing GPU lock. Each case has a strict fifteen-minute wall-clock
budget for the new sweep; its two fits run sequentially. No GPU work is launched
by importing the module, reading its CLI help, or running its CPU tests.

Outputs are additive under `case-adaptation/lora-sweep/`:

- `lr-3e-5/` and `lr-1e-4/` contain the two independent fits, with separate logs.
- Each completed fit gets an immediate `seal.json` covering its nine merged
  tensors, workspace manifest, training/settings reports, and adapter factors.
- `identity.json` and frozen source copies bind the source/data/original-fit
  identities. `initial-seal.json` records the existing original fit without
  changing any original file.
- `status.json` retains progress and any failed attempt. Failure stops the sweep;
  neither failed trials nor interrupted artifacts are silently discarded.
- `nomination.json` binds all three candidates, data/spec/source hashes, costs,
  validation diagnostics, and the next two closed-loop candidates.

Scope verification uses the actual workspace tensor names, model paths, scan
indices, and edited-tensor lists. The base workspace's inherited `include_action`
exposure flag is broad and makes the legacy validator's convenience field
misleading for these selected VLM-only workspaces; that flag is not treated as
evidence that action-expert weights changed. Original metadata remains preserved.

Stage A nominates only among the **two new rates**, minimizing mean physical
action MSE on the fixed development-validation inputs. Repair and preservation
receive equal weight. An exact metric tie selects the smaller learning rate.
The original 0.001 fit remains the second candidate for closed-loop selection,
even if its action error is lower than both new rates. No training checkpoint
within a fit is selected: each candidate is its final 200-step state.

The controller then evaluates the nominated new rate and the original rate on
the known repair and primary-preservation goals, using sixteen fresh development
policy-noise seeds per goal. This is 64 new policy episodes per case and 128
across both cases. It selects the candidate with the largest sum of strict task
successes across those two goals; ties use lower balanced action MSE and then
lower learning rate. The sweep itself launches no policy episodes and makes no
closed-loop success claim. Held-out layouts and target phrasings do not enter
selection. All three fits and all selection episodes count toward the adaptation
budget; the shared target-generation cost is reported separately.

CPU verification:

```bash
python3 -m unittest discover -s study/iros2027/v2/rss_repair_comparison/lora -p 'test_sweep.py' -v
```

Seven tests cover the new-rate-only nomination rule, tie handling, failed or
missing trial rejection, role-balanced action error, unchanged recipes and
validation counts, source mutation detection, and all selected tensor seals.
They use synthetic metadata and tiny arrays; they are not policy results.
