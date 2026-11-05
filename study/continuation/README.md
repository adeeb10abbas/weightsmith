# Continue the study

## Add the teammate's results to the paper

Edit the single `../paper/` directory. Start with `sections/results.tex`, `sections/discussion.tex` and `sections/conclusion.tex`, then revise the abstract to match the new evidence. Preserve the original completed Astra measurements and distinguish new cohorts and runtimes. Use [RESULT_TEMPLATE.md](RESULT_TEMPLATE.md) to record the evidence behind every new claim.

The current narrative treats state-input/VLA/WAM and motor-versus-perception differences as hypotheses. Preliminary observations are not yet numerical findings in the manuscript. These categories may overlap. A failure classification needs validated evidence, not only a plausible description of a video. Control task training, observation access, editable components, budget and evaluation conditions where possible; report residual confounds.

## Run an independent editor trial

The recommended source is `../reproduction/fixed16/`. From that directory:

```sh
python3 tools/verify_package.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Then read `SETUP.md`, `MODEL_ENROLLMENT.md`, and `PACKETS.md`. Download the common baseline using `../ARTIFACTS.md`, and follow `../reproduction/common-baseline/README.md` and its verification tool. On a different host or runtime, validate it and collect a fresh baseline; report a separate cohort rather than rewriting recorded provenance.

The sealed dispatcher assumes two GPUs. A one-GPU adaptation requires a separately tested implementation. Pin the exact provider/model and effort, confirm the live adapter and tool isolation, and complete native replay and unchanged-policy controls before search. The provider adapter is CPU-tested, not live-Opus-certified. The original pod's low disk headroom also means it should not be reused for new jobs without a fresh resource check.

The proposed annotation correction remains unapplied. Fix and validate it in a new version for future experiments; do not rewrite historical logs or claim that the frozen study used corrected feedback. Preserve all proposals, failures, no-edit abstentions, actual bfloat16 changes and total effort. Freeze search choices before independent final scoring.

For a model comparison, give a fresh editor only its clean brief, authorized common evidence, and its own permitted feedback. Human/operator access to the paper does not authorize leaking Astra outcomes or held-out cases into that editor. Cross-policy studies should use an explicitly new protocol and state any prior exposure.

[Prospective priorities](priorities.md) describe the scientific follow-up without launching jobs.
