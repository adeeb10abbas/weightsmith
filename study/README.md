# Completed Astra study — start here

**Status: experiments complete; current paper ready for coauthor editing.** This handoff contains the completed one-task Astra study, its latest manuscript and review, and a sealed fixed-batch reproduction package. It contains no Opus result and no numerical result from the teammate's proposed policy-family comparison.

## Choose your next step

| Goal | Start here |
|---|---|
| Read or edit the paper | [Current PDF](paper/paper.pdf) and [one LaTeX source directory](paper/README.md) |
| Review what was measured | [Results and limitations](RESULTS.md), [measurement records](results/data/), and [figures](results/figures/) |
| Read the accessible explanation | [Editable blog](blog/draft.md); [full offline review with videos](ARTIFACTS.md#full-review-bundle) |
| Review the recent literature | [Critical evidence table](blog/recent_work_table.md), [survey](review/literature-survey.md), [Guo–Xie assessment](review/guo-xie.md) |
| Run a new model or policy experiment | [Continuation guide](continuation/README.md), then the [sealed fixed16 package](reproduction/fixed16/README.md) |
| Locate large downloads | [Artifact catalog and checksums](ARTIFACTS.md) |
| Audit scope and provenance | [Provenance notes](provenance/README.md) and [file manifest](provenance/copy-manifest.json) |

## What is complete

The study used 1,056 simulated episodes: 128 baseline/calibration, 480 candidate, and 448 final-control episodes. Five Astra searches evaluated 15 proposals: four changed deployed weights and eleven were identities. None of the four edits passed the frozen gate. The separate historical smoke used 16 episodes. Four control comparisons and the fixed-input diagnostics are complete; no study GPU jobs are queued.

The manuscript has six body pages plus references. It distinguishes positive literature reports, our measured outcomes, implementation problems, and hypotheses. Policy family and motor-versus-perception differences remain hypotheses pending the teammate's evidence. All previous drafts and the full raw archives remain on the original laptop; they are not mixed into this current manuscript folder.

## Code boundaries

The repository's top-level code includes the teammate's newer parallel-evaluation and locking work. It is **not the runtime used for the completed Astra measurements**. The exact tested fixed16 implementation and its setup fixes are in `reproduction/fixed16/bundle/weightsmith/`, with source manifests and tests. Keep this distinction when reproducing or extending the study. Porting fixes between these implementations is a new integration change requiring its own tests and runtime identity.

The reproduction package's historical metadata says original-host GPU validation was pending when it was packaged. That historical statement is preserved to keep its seal intact. Later validation and the completed results establish readiness on the original host only. A new host and model provider still require their own checks. The frozen dispatcher expects **two GPUs**; a one-GPU port is not certified by this handoff.

For an independent editor-model comparison, the human coauthor and operator may read this whole study, but a fresh scientific editor context must receive only its clean brief, authorized baseline evidence and its own allowed feedback. Keep Astra outcomes and held-out evidence out of that editor context. For ordinary coauthor writing and broader follow-up studies, identify any prior exposure and use a new protocol.
