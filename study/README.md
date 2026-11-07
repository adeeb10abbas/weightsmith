# Study records: start here

This folder holds the evidence behind weightsmith's results:
- the report and its records;
- a controlled study of seven blind editors;
- the companion GPT-6 Astra study, with its manuscript and a sealed reproduction package.

Every result here is simulated (RoboLab on Isaac Sim) and uses one checkpoint, OpenPI `pi05_droid_jointpos`. New here? Read the [report](blog/failure-levels.md) first, then the [leaderboard](../LEADERBOARD.md), which lists every editor run in one table. The edits that passed held-out scoring can be downloaded and verified from [`edits/`](../edits/README.md).

## Where each study lives

| Study | What it asked, and what it found | Write-up | Records |
|---|---|---|---|
| **Claude Opus cohort**, Nov 4–6, 2026 | Which kinds of failure can closed-form weight edits fix? One clear success: closed-form instruction distillation (`v2mi-f`) on cube then banana took held-out success from 27/128 to 106/128, and the same edit helped two more tasks. Edits aimed at perception, a missing skill, contact physics and the gripper threshold gave no gain. | [Report](blog/failure-levels.md) §4–6 | [`results/opus/`](results/opus/): [`comparisons.json`](results/opus/comparisons.json) (every paired comparison), [`heldout/`](results/opus/heldout/), [`EVIDENCE.md`](results/opus/EVIDENCE.md) (protocol, budgets, costs, every proposal), [`ctb/`](results/opus/ctb/README.md) (edit scripts and the attempts that failed first) |
| **Controls** | Does the content of the edit matter, or only its size? A random-direction edit with the same layers, rank and singular values scored 9/64, like the unedited policy, while the explicit wording scored 57/64. With unchanged weights, about 30% of paired outcomes flip. | Report [§5.5](blog/failure-levels.md#55-the-results) and [§7](blog/failure-levels.md#7-how-not-to-fool-yourself) | [`results/opus/controls/`](results/opus/controls/) (the random-direction and explicit-wording runs on held-out seeds: scripts, plan, results, analysis); the `identity` rows of `comparisons.json` |
| **Claude under the Astra protocol** | The Astra study's exact brief, given to Claude Opus on stack yellow on red. Neither of its two proposals passed the gate, and it submitted the unedited policy. | Report [§6.1](blog/failure-levels.md#61-perception-stack-the-yellow-block-on-the-red-block) | [`results/opus/claude-astra-protocol-*.md`](results/opus/claude-astra-protocol-FINAL.md) (brief, notebook, final answer) |
| **Seven blind editors** (controlled study), Nov 6, 2026 | One fresh Claude Opus editor per failure kind, each with the same tools and budget. One edit was accepted: on cube then banana, 1,024 values took held-out success from 11/64 to 27/64 (p = 0.004). The eyeglasses and mustard edits stayed within noise, and four editors submitted no edit. In a second round with more time, the stack-three-cubes editor's 3,072-value edit scored 35/64 → 39/64 (p = 0.58) and was rejected. | Report [§7a](blog/failure-levels.md#7a-controlled-study-seven-blind-editors); [`results/ctrl/README.md`](results/ctrl/README.md) | [`results/ctrl/`](results/ctrl/README.md): one folder per task (`BRIEF.md`, `NOTEBOOK.md`, `FINAL.md`, `scripts/`, `specs/`), [`heldout/`](results/ctrl/heldout/), [`summary.json`](results/ctrl/summary.json), [`operator/`](results/ctrl/operator/); round 2: [`stack3/FINAL2.md`](results/ctrl/stack3/FINAL2.md) |
| **Companion GPT-6 Astra study** (Esmaeil Seraj), completed Nov 5, 2026 | Can GPT-6 Astra repair stack yellow on red with restricted edits to action-MLP rows? It ran 1,056 episodes and five searches with 15 proposals. Four proposals changed weights, and none passed the frozen gate. Reruns of the unchanged policy moved by −5 to +14 points. | [`RESULTS.md`](RESULTS.md) (results and limitations); [`blog/draft.md`](blog/draft.md) (write-up); [`paper/`](paper/README.md) (six-page manuscript, [PDF](paper/paper.pdf)) | [`results/data/`](results/data/) (measurements), [`results/figures/`](results/figures/) and [`results/scripts/`](results/scripts/), [`reproduction/`](reproduction/fixed16/README.md), [`ARTIFACTS.md`](ARTIFACTS.md) (large downloads) |

The main weightsmith paper is in [`../paper/`](../paper/README.md). `study/paper/` holds the Astra study's own manuscript. The file formats of the result records are described in [`results/README.md`](../results/README.md).

## Folder map

```
study/
├── README.md               this index
├── AGENTS.md               rules for anyone, human or agent, who edits the study
├── RESULTS.md              Astra study: results and limitations
├── ARTIFACTS.md            Astra study: large downloads (raw evidence, review bundle) and checksums
├── blog/
│   ├── failure-levels.md     the report: Claude cohort, controls, seven blind editors
│   ├── draft.md              the Astra study's write-up (dated Nov 5, 2026)
│   ├── costs.md              Astra study: recorded discovery effort, linked from draft.md
│   ├── generated_final_*.md  Astra study: generated final-control tables, linked from draft.md
│   ├── recent_work_table.md  others' reported results, not reproduced
│   ├── social.md             social-media drafts
│   └── figures/              figures for both write-ups
├── results/
│   ├── opus/               Claude cohort: comparisons, held-out records, evidence record, edit scripts (ctb/),
│   │                       controls/, the distill2/ task screen, Claude under the Astra protocol
│   ├── ctrl/               seven blind editors: one folder per task, heldout/, summary.json, operator/
│   ├── data/               Astra measurement records (frozen)
│   ├── figures/            figure exports for the Astra study and the report
│   └── scripts/            the scripts that draw them
├── paper/                  Astra study manuscript: LaTeX sources and paper.pdf
├── reproduction/           Astra study: fixed16 reproduction package and common baseline (sealed)
├── provenance/             Astra handoff: copy manifest, artifact checksums, receipts
├── continuation/           how to extend the Astra study: guide, result template, priorities
└── review/                 literature survey and the Guo–Xie assessment
```

## Sealed, frozen and verbatim

| Path | Status | Rule |
|---|---|---|
| `reproduction/**` | Sealed | Kept byte for byte, including whitespace (`.gitattributes`). `fixed16/tools/verify_package.py` and `common-baseline/SHA256SUMS` check it. Never edit. |
| `results/data/**` | Frozen | The Astra measurement records. Never edit. |
| `RESULTS.md`, `provenance/` | Frozen record of the completed Astra study | Do not reinterpret historical statements; add new evidence as a new cohort. |
| `results/*/heldout/*.json` | Append-only | `weightsmith/heldout.py`'s `results.json`, with its `.plan.json` where kept, unmodified. Held-out runs are never repeated or overwritten. |
| Editor files: `results/ctrl/<task>/*.md`, `results/opus/claude-astra-protocol-*.md` | Verbatim | The editors' own words and the briefs they read. Do not edit. |
| `paper/` | The Astra study's live manuscript | Coauthors edit it under the rules in [AGENTS.md](AGENTS.md): six body pages, compiled with XeLaTeX. |

## Before you reuse the Astra material

- **Two runtimes.** The repository's top-level `weightsmith/` runtime produced the Claude cohort and the controlled study. It is not the runtime that produced the Astra measurements: that one is `reproduction/fixed16/bundle/weightsmith/`, with its own manifests and tests. Porting fixes between the two is a new integration change that needs its own tests and runtime identity.
- **Two GPUs.** The sealed Astra dispatcher expects two GPUs, and a one-GPU port is not certified. Its readiness was established only on the original host. A new host or model provider needs its own checks ([continuation guide](continuation/README.md)).
- **Keep editors blind.** A fresh scientific editor gets only its clean brief, authorized baseline evidence and its own feedback. Never give it Astra outcomes, held-out evidence or other editors' notes. The review bundle in [ARTIFACTS.md](ARTIFACTS.md) contains Astra outcomes.
- **New work, new cohort.** Put new experiments in a separately named folder, such as `results/<cohort>/`, with a record filled from [`continuation/RESULT_TEMPLATE.md`](continuation/RESULT_TEMPLATE.md). Never pool runtimes or baselines.
