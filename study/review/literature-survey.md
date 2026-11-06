# Weight-space programming: evidence, applications, and a robotics research plan

Research snapshot: **November 4, 2026, America/Los_Angeles**. Prepared for the `weight_programming` project.

**Later November 4 update:** the [new Gupta et al. robotics article](https://codnay.github.io/blog/llms-fix-robots-by-editing-weights/) provides substantially more detail than the original cloth-folding X post discussed below. It reports a ten-task direct-neuron-edit study, with improvements on five tasks and held-out counts for that arm. Its separate 51% to 75% folding result uses a difference between trained models; the pasted-update denominator is not stated. Accordingly, the earlier statement that only an unspecified social claim was available is historical. The revised [four-page manuscript sources](../ARTIFACTS.md#full-review-bundle) incorporate this distinction and a 20-row evidence table. The original research audit below is retained as a snapshot.

This report covers the original release, the supplied social links, independent experiments, negative results, and implications for manipulation policies. It is a literature and artifact investigation: **we have not run the reported models or independently reproduced their numerical results in this project**. “Verified” below means that a source or artifact was inspected, not that its experiment was rerun.

## 1. Assessment

**Weight-space programming is worth testing for localized policy repair. Current evidence does not establish a general replacement for policy training.** The useful research question is whether an agent can diagnose a repeatable failure, identify a small intervention, and improve held-out behavior more cheaply than the best ordinary fix.

The public record already contains more than commentary: checkpoint reconstructions, independently discovered edits, toy-network audits, translation experiments, a driving-policy report, and a preliminary cloth-folding claim. However, the phrase now covers materially different methods. Some use only weights and a specification; others use extensive inference, activation analysis, fitted detectors, changes to inference code, or subsequent training. These should not be pooled into one success rate.

For this project, the most promising initial targets are systematic errors such as incorrect gripper timing, startup/history failures, action calibration, or recurrent stalling. Broad improvements in visual grounding, contact-rich dexterity, and unfamiliar-object manipulation remain hypotheses. The proposed experiments in Section 7 are our recommendations, not published findings.

## 2. What the original work actually establishes

Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab published *Sparks of Weight-Space Programming* on November 2. The release calls itself a blog post and supplies experiment logs and model artifacts. I did not find a separately released original arXiv paper or evidence of peer review. Its central proposal is to have a general-purpose LLM interpret or directly construct weight values from a task specification, without an optimizer training the target network. [Original article](https://rikiyat.github.io/blog/weight-space-programming/)

### Experiment ledger

Each row links to the authors’ detailed experiment record rather than relying on the launch headline.

| Experiment | What the editor received and did | Reported result | Practical interpretation |
|---|---|---|---|
| 1. Identify a policy’s game | Architecture, semantic input/action layout, and 17,253 weights as text; no tools or network execution | Four classifications accepted. The Freeway network was called Frogger, which the scorer explicitly accepts | Evidence for interpreting small, structured policies; not arbitrary semantic recovery from anonymous tensors. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game) |
| 2. Identify diffusion outputs | Three 23,824-parameter image MLPs; raw weights, no forward passes | Correct shape class for circle, square, triangle; incorrectly described filled shapes as outlines | Partial structural understanding, with an observable semantic error. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp2_diffusion) |
| 3. Interpret transformer interiors | Middle blocks of 7.16M-parameter character models; embeddings/output removed; Python matrix analysis allowed, forward passes forbidden | Python/romaji/English correctly selected in six conversations; copying heads matched the stated 0.2 induction-score cutoff | Stronger than reading an input heatmap, but a three-option task. Released score JSON uses 0.3; README explains the discrepancy. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp3_language) |
| 4. Repair physical control | 171,664-weight Kinetix car actor, scene/observation specification, aggregate failure result, and one trajectory; one reply without tools | 0/256 baseline successes; 1,024/1,024 held-out episodes after changing 6,073 values, 3.54% | Closest original analogue to policy repair. A small circuit backs up, accelerates, then hands control back. Evaluation is one level, not demonstrated transfer across scenes. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car) |
| 5. Write policies from zero | MinAtar rules and exact source, semantic observations/actions, fixed 20,486-parameter MLP; no execution feedback | Beats 100M-step PPO in three games. Against 500M-step PPO: Breakout 517.4 vs 25.3; Seaquest 53.1 vs 88.9; Asterix 53.2 vs 55.2 | Neural implementation of task-specific algorithms. The stronger PPO comparison is one win and two losses. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch) |
| 6. Write a Go policy | 1.48M-parameter ReLU MLP; Python and 0–50 feedback calls; no fitting or tree search | Zero-feedback policy wins 54.1%/54.9% against small PPO opponents, but loses to all three same-size PPO seeds. At 50 calls it wins 70.0–78.5% against same-size PPO | Feedback materially changes the result. No comparison to strong modern Go engines. A call can contain 128 games or 256 positions. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch) |
| 7. Detect sycophancy | Paired Gemma 3 270M fine-tunes; initially only weights and Python, later attention observations | Weight-only rounds were uninformative; with attention access, all three pairs classified correctly | A negative result for general behavioral auditing from weights alone. Attention observations supply behavior-dependent evidence. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy) |
| 8. Repair Qwen reasoning loops | Qwen3.5-0.8B, Python tensor access, labeled practice problems and batched generation/activation probes | MATH-500 7.9→48.1%; GSM8K 13.8→67.1%; IFEval 21.0→62.3%; token-limit hits 93.0→33.1% | Targeted failure repair under fixed inference settings. The successful 50-call run took 158 turns, 80 Python executions and 20.3 hours including downtime. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math) |
| 8b. Improve a non-looping model | Gemma 3 270M with the same style of editing harness and 5–50 feedback calls | MATH-500 6.8% baseline; edited results 7.0–7.7%, without an established gain | The procedure did not demonstrate adding missing mathematical competence. These scores use format-agnostic grading, unlike the main Qwen result. [Record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/gemma_followup) |

### Three distinctions that affect reproduction

**No gradients does not mean no data or cheap discovery.** The Qwen prompt includes labeled practice sets, recommends causal interventions and dose-response tests, and explicitly mentions repetition as a possible failure. One feedback call can generate 512 answers or read 1,024 texts. “50 forward passes” therefore does not mean 50 individual examples or token-level model evaluations. The prompt gives substantial methodological guidance, even though the editor still has to discover an effective intervention. [Actual initial prompt](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt)

**Small edit count does not imply small behavioral impact.** The Qwen patch targets 64 output columns in each of two MLPs; 131,072 entries are addressed and 130,188 differ after bfloat16 rounding. The released update removes six times a selected projection, reversing and amplifying that component rather than merely zeroing it. This is a substantive intervention despite its sparse support. [Edit record](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/runs/p50c/edit), [independent byte-level audit](https://carlostoledo.co/reports/weight-space-programming.html)

**A reconstructable final edit is different from a completely packaged discovery run.** The repository has prompts, outputs, evaluations, networks and exact diffs. It excludes earlier prompt trials, some intermediate files and the main math experiment’s complete agent/forward-pass harness. It also excludes the large sycophancy checkpoints, though reconstruction materials are supplied. This is a valuable release, with remaining reproduction work. [Release README](https://github.com/RikiyaT/lm_understand-logs), [math release contents](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)

## 3. Supplied links and social evidence

All four supplied social links were ultimately resolved. The three X posts were read directly in the browser after ordinary web retrieval failed. **Two are adjacent robot-use research rather than weight-space programming experiments.** This distinction changes how we should use them.

| Supplied link | Actual topic | Classification |
|---|---|---|
| [Yuxin Zuo](https://x.com/zuo_yuxin/status/2117635294115795134), November 3 | `uni-robot-agent`: unified harness for robot-use agents and local open models | Relevant robotics infrastructure/baseline; the post does not report direct policy-weight programming |
| [Andreas Look via LinkedIn](https://lnkd.in/p/gBjdRsQs), November 4 | Claude edits a UniAD driving checkpoint | Direct weight-editing application; open-loop author-reported results |
| [Ben / bentropy_ai](https://x.com/bentropy_ai/status/2117702901493837981), November 3 | ZAP: simulation-derived training for a small VLM robot agent | Conventional fine-tuning and RL; not gradient-free WSP |
| [Pranay Gupta](https://x.com/PranayGupt97167/status/2117708521689223234), edited November 3 | Direct π0.5 weight manipulation for cloth folding | Most directly relevant manipulation claim; protocol and artifacts not supplied in the post |

### Pranay Gupta: π0.5 cloth folding

The exact post reports RoboDojo cloth-folding results of **50% base**, **68% fine-tuned on Astra corrections**, and **75% direct weight manipulation**. The last comparison is a reported seven-percentage-point improvement over the fine-tuned arm. The post promises a future breakdown.

No trial count, confidence interval, editor procedure, target tensors, checkpoint diff, data split or budget comparison was available in the retrieved post. Consequently, statistical significance and comparability of the arms cannot be assessed. It should not be described as validated physical-robot improvement. This is a high-priority reproduction lead, not a basis for selecting WSP over fine-tuning. [Exact original post](https://x.com/PranayGupt97167/status/2117708521689223234)

### Yuxin Zuo and Ben: useful adjacent robotics work

**OpenRUA:** the primary project report attributes Qwen3.8-Flash-Next’s RoboDojo **Score 1.90→31.39** to a changed harness; weights are unchanged. The latter is **25% success**, not 31.39% success. Astra’s often-quoted 28.97 uses the older RoboProbe harness; Astra scores **51.90 in the new harness**. On 19 physical-robot tasks, Qwen’s **53.50/100** and π0.5’s **47.75/100** are aggregate capability/progress scores. Qwen receives depth sensing while π0.5/Dream-1 use RGB, so the comparison does not isolate model quality. The harness combines shared tools, execution feedback and pipelining; trajectory-based training is future work. These are directly useful lessons in controlling our baseline and observation interface. [Original thread](https://x.com/zuo_yuxin/status/2117635294115795134), [primary project report](https://openrua.ai/uni-robot-agent)

**ZAP:** the paper starts from Qwen3.6-27B, collects successful simulated episodes through a GRPO warm-up and search, then applies supervised fine-tuning and longer **GRPO with dense rewards**. It uses 40 simple simulation tasks. In a shared real-robot harness, ZAP and Astra both achieve **29/30 successes across ten tasks**; mean time is **146 vs 397 seconds**, and reported cost **$0.29 vs $4.87**. This is a small evaluation despite the high success percentage. Transfer to an unfamiliar Franka is evaluated in simulation. The paper was initially submitted October 26 and revised November 3, so it is not evidence of a response to the November 2 WSP release. “No real robot data” does not mean “no training.” [Original thread](https://x.com/bentropy_ai/status/2117702901493837981), [primary paper v3](https://arxiv.org/html/2610.35817v3), [version history](https://arxiv.org/abs/2610.35817)

### Andreas Look: UniAD autonomous driving

The supplied LinkedIn short link resolves to a first-person experiment by **Andreas Look**, published November 4 at 14:56 UTC. Claude Opus produced ten checkpoint versions over approximately 12 hours, reportedly without retraining or code/configuration changes. It changed 74,374 of 131.9M values, or 0.056%.

On nuScenes validation using UniAD’s own metric, average L2 fell **0.942→0.845 m**, and collision rate **0.29→0.17%**. Much of the benefit came from first frames without history: a nearly invariant long trajectory was corrected by routing an existing auxiliary signal through unused neurons. Other interventions modified LayerNorm and occupancy outputs; several attempted fixes failed.

The author explicitly cautions that most improvement was concentrated at startup and that open-loop scores need closed-loop validation. No code, checkpoint or evaluation-log link appeared in the retrieved post. This is a relevant quantitative report, with insufficient public evidence to infer better deployed driving. [Original LinkedIn post](https://www.linkedin.com/posts/andreas-look-6b46a7149_autonomousdriving-endtoend-llm-activity-7523760157894365185-nAym)

## 4. Independent experiments and applications

Evidence labels used here: **artifact-backed** means public code, weights, logs or detailed evaluations were available; **reported** means a first-person result with insufficient artifacts inspected; **proposal** means no demonstrated result. None implies our independent reproduction.

### A. Reconstructing and evaluating the original Qwen patch

**Ahsen Khaliq / ML-Intern-lab, November 2 — artifact-backed checkpoint reproduction.** The model card reports reconstructing the published edit and verifying tensor hashes, then evaluating 500 MATH-500 problems with four samples each. It obtained **51.0%**, with **31.6%** reaching the token limit. The baseline check covered only 25 problems. This reproduces the edited checkpoint’s performance; it does not independently reproduce the process that discovered the patch. The exported model includes the vision tower, while the original used a text-only export; language tensors were hash-verified. [Model card and evaluation links](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)

**Carlos Toledo, November 2–3 — artifact-backed audit and baseline rerun.** The audit verifies the tensor edit and agrees with released grading on the answers its exact grader can decide. More consequentially, rerunning the unedited model produced **7.2%** under the original settings, **22.3%** with the model card’s thinking settings, and **44.0%** with its non-thinking settings, counting undecidable answers as wrong. The non-thinking decided-only score was **49.7%**. These denominators and grading differences matter: neither number is a clean matched comparison against the authors’ 48.1%. Nevertheless, much of the headline multiplier disappears against a better configured unedited baseline. This does not refute the fixed-settings patch effect. [Audit and pinned records](https://carlostoledo.co/reports/weight-space-programming.html), [issue filed with the authors](https://github.com/RikiyaT/lm_understand-logs/issues/4)

### B. Independently discovering a different Qwen edit

**`leofmarciano/weightspace` — artifact-backed independent editing and useful controls.** The repository reports an Opus 5.5 edit scaling four MLP neurons, affecting 4,096 weights. In thinking mode on the first 250 examples per benchmark, MATH/GSM8K changed from **8.4/12.4%** to **40.0/53.2%**; the original published patch scored **52.4/61.6%** in that evaluation.

With thinking disabled, on 500 examples per benchmark and a different token budget, the base scored **46.4/56.6%**, the original patch **42.0/57.2%**, and the new patch **41.8/54.4%**. Thus it supports independently finding a loop repair, without showing consistent gains over the non-thinking baseline. Its small development-set result also exceeded performance on fresh practice data. The library exposes tensor inspection, probes, patches and rollback; it is a candidate harness to inspect, not a dependency we have validated. [Code and results](https://github.com/leofmarciano/weightspace)

### C. Weight reading and policy construction with other editors

**Simon Willison, November 3 — artifact-backed narrow reading replication.** A local quantized Qwen3.8-27B correctly identified Pac-Man, Pong and Breakout from the original prompts, but misidentified Freeway. Successful runs took roughly 46–65 minutes each; context and timeout settings needed adjustment. Kimi K3.1 Pro also identified Breakout in a separate run. Public transcripts make this useful evidence that the small-policy reading task is not exclusive to the original editor. It does not establish policy repair or understanding of a large pretrained model. [Article and transcripts](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)

**Alex Krentsel, November 2 — reported cross-model policy-writing replication.** His thread describes repeating the original Breakout construction task with multiple model generations, using the same prompt/scoring and a single reply without tools or training. The target was the 20,486-parameter policy from Experiment 5. Treat this as Breakout-specific evidence; it is not a Qwen math-edit replication. Exact plotted performance values were not verified from the primary graph in this review. [Initial post](https://x.com/AlexKrentsel/status/2117376683724660898), [scope clarification](https://x.com/AlexKrentsel/status/2117379728063799532)

**Federico Galatolo, November 3 — artifact-backed control-policy reading.** DeepSeek V4.1 Flash classified three of four anonymized 51,203-parameter actor-critic checkpoints correctly from supplied choices: CartPole, Flappy, BallBeam and Catch. It confused BallBeam with CartPole at 88% confidence. Four calls consumed about 1.58M tokens. This extends reading evidence to other control tasks, while exposing ambiguity and poor confidence calibration; no policy was repaired. [Experiment report](https://github.com/galatolofederico/wsp/blob/main/reports/dsv4.1-flash-1017.md)

**Chandan Singh, November 2–4 — artifact-backed construction and predictive controls.** Claude generated 959 programs across 13 tasks; ten algorithmic tasks reached 100% on 200 evaluation cases within three attempts. Architecture changes were permitted, and some purported zero-parameter solutions moved constants into buffers. Language-task evaluations had train-pool leakage; an SNLI attempt used forbidden logistic fitting. Those results should not be counted as clean WSP successes. A separate 420-edit study found only a modest correlation advantage from giving the editor raw weights, concentrated in single-scalar edits and disappearing for lower-ranked target tokens. [Construction report](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html), [repository and edit-prediction study](https://github.com/csinva/weight-space-programming-autoresearch)

### D. Italian translation: a localized behavior edit

**Valerio Dolci — artifact-backed, gradient-free but fitted intervention.** An agent edited 4–8 MLP neurons in Gemma 4 12B to steer Italian tense preference. The fraction of held-out target sentences using the nonpreferred tense fell **26→5%** and **32→16%**; two control examples broke, while 182/200 general outputs stayed identical. Both tense choices are usually valid: this is preference steering, not correction of translation errors. The workflow used approximately 27–30 forward passes and a closed-form ridge fit on activations. It belongs to data-assisted editing, not strict no-fitting weight programming.

A LoRA baseline matched the earlier edit’s aggregate target rates and performed better on paragraphs and protected controls; it did not match the best edit on every metric. Two contaminated prompt tests were excluded. The separate Qwen brand-implant experiment trains selected neurons on examples: sparse fine-tuning, not gradient-free replication. [Repository](https://github.com/ValerioDolci/frontier-weight-edit), [separate trained-neuron experiment](https://github.com/ValerioDolci/frontier-weight-edit/tree/main/experiments/qwen27b-brand-implant)

### E. English–Hindi translation and story repetition: success has boundaries

**nutanc, November 3 — detailed first-person study; the successful edit is an activation hook.** In a fine-tuned 4B translator, neuron ablations and generic directions failed to reliably recover lost sentence meaning without damage. Providing sentence-specific phrase notes worked better.

For a TinyStories-tuned Gemma 3 270M, a layer-8 intervention reduced repeated sentences **32→2%** and loops **9→0%** on 100 held-out openings. Shuffled-label controls worsened repetition. However, markup rose **35→44%**, stronger interventions broke generation, and the critic that supplied labels also scored the result.

Crucially, the author states that this is **a constant inference-time activation shift, not a saved-weight edit**, and did not use an LLM to read raw tensors. It supports a mechanism hypothesis—shared repetitive failures may be easier to steer than heterogeneous content errors—rather than directly replicating strict WSP. [Study and limitations](https://gpt3experiments.substack.com/p/experiments-with-weight-space-programming)

### F. Clinical decision-network auditing

**Humza Nusrat and Omar Nusrat, November 3 — artifact-backed synthetic study.** Opus 5.5 audited 433–4,801-parameter screening networks from a protocol and weights, without execution. It identified **16/18** planted defects and repaired those with a median of **three parameter changes**; evaluation covered approximately 420,000 requests.

The controls sharply limit the conclusion: only **2/5 healthy networks** were cleared without forcing an answer, and **1/3 hidden triggers** was detected. Simple weight statistics found the same localized confounds/threshold shifts; black-box testing was faster; conventional fine-tuning repaired all networks. Some large-magnitude repairs failed in FP16.

This is evidence for localization and repair in small, interpretable networks, not general auditing reliability or clinical readiness. The useful robotics lesson is to measure false diagnoses on healthy policies and validate the actual deployment precision. [Preprint](https://arxiv.org/abs/2611.05341), [code and manuscript](https://github.com/hnusrat1/weightspace-rt)

### G. Neural quantum-state transfer

**Paul Gauthier, November 4 — artifact-backed hybrid pipeline.** Opus adapted a roughly 435K-parameter quantum-state ViT from a 10×10 spin lattice to sizes through 18×18. It extrapolated attention, changed output computation to overlapping windows, and explicitly implemented a phase rule.

Editing alone remained around **0.002–0.004 energy/site** behind the selected reference results. Subsequent variational Monte Carlo fine-tuning and a Lanczos step matched the cited 14×14 reference within uncertainty and approached the cited 18×18 result. Editing did not improve the original 10×10 task. The author reports roughly two days and several hundred dollars of GPU expenditure.

This is an interesting initialization/transfer application, with code changes and later training essential to interpreting the headline. It is not a fixed-architecture, no-training result. [Author’s article](https://paulg.info/2026/11/04/quantum-weight-space/), [repository](https://github.com/paul-gauthier/quantum-weight-space)

### H. Local quantized coding-agent loops

**nitayrabi — preliminary mixed intervention.** A first-person report describes Gemini editing a 35B quantized model’s late shared-expert projections, alongside a chat-template repair and harness instruction changes. It reports improvement from 16/23 to 25/27 cases and a large reduction in one stalled session’s runtime.

The changing denominator and simultaneous non-weight changes prevent attributing the gain to weight editing. The author explicitly calls it preliminary; the linked model repository was not retrievable in this review. This is a useful lead for ablation studies, not established evidence that direct tensor edits alone fixed the agent. The precise publication date was not verified from the retrieved relative timestamp. [First-person post](https://www.reddit.com/r/ollama/comments/1xmyok4/trying_weightspace_programming_on_a_local_35b/)

### I. Vision restoration and software-engineering infrastructure

**Zhuoran Zheng and Doubao, November 3 — proposal.** Their restoration-diffusion manuscript proposes diagnosing over-smoothing, color shifts or texture artifacts, then applying targeted edits in an evaluation loop. It explicitly describes its claims as conjectural. No measured restoration improvement should be attributed to this paper. [Preprint](https://arxiv.org/abs/2611.04268)

**Voravee Upatising / Claude: WSSE and `wpgit` — artifact-backed infrastructure, inspected at a November 4 commit.** This project represents patches as hash-pinned textual changes with behavioral tests, reconstruction, interaction/order checks and stale-patch detection after upstream changes. Demonstrations involve image generators, restoration and legacy Keras models. Separately passing edits can fail when combined; some combinations require retraining. This is directly relevant to maintaining several policy repairs, but it is not a robot-manipulation result and includes fitted/fine-tuned interventions. [Project and code](https://github.com/v-code-895/wsse)

### J. Relationship to earlier methods

Direct parameter editing and neural construction predate this release. [HyperNetworks](https://arxiv.org/abs/1609.09106) train a network to generate weights; [ROME](https://arxiv.org/abs/2202.05262) localizes and edits factual associations with a prescribed method; [task arithmetic](https://arxiv.org/abs/2212.04089) operates on differences obtained through fine-tuning; [Tracr](https://arxiv.org/abs/2301.05062) compiles formal programs into transformer weights. The distinctive question here is how far a general-purpose reasoning model can select, construct and validate these interventions without task-specific editor training.

## 5. What the debate gets right—and what remains unproven

**The term merges different capabilities.** Matthew Honnibal’s critique distinguishes raw-weight reading, neural algorithm construction and tool-assisted model repair. That distinction is supported by the released protocols and matters more than whether all three deserve one label. [Critique](https://honnibal.dev/blog/astra-read-write-misleading)

**A human-interpretable shortcut is a necessary baseline.** Sean Goedecke demonstrates how input-weight norms expose recognizable structure in the small game networks and relates the Qwen procedure to established steering techniques. This motivates simple statistical and human-engineered controls; it does not itself negate the agent’s ability to carry out the procedure. [Critique and heatmap method](https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/)

**Replication has several meanings.** Rebuilding the published checkpoint, rerunning its evaluation, independently discovering an alternative patch, and transferring the method to a new task answer different questions. The first three now have public evidence. Transfer exists, but its quality varies markedly and many examples relax the original restrictions.

**Do not turn one negative result into an impossibility theorem.** The non-looping Gemma result shows this procedure failed in that experiment. It does not prove weight edits can never add capabilities; small policies constructed from zero already implement new behavior. Conversely, constructing an understandable control algorithm in a small MLP does not establish general skill acquisition in a large VLA.

**Sparse edits do not guarantee preservation.** A handful of weights may dominate output; a low-rank direction can affect many inputs; precision changes can invalidate a repair. Preservation must be tested on protected behavior, not inferred from unchanged parameter count.

**Best discovered patch is not expected performance.** For our work, repeated editor runs and failed attempts must count. A fixed patch’s evaluation uncertainty is different from the variance in finding that patch in the first place. Current evidence does not establish broad discovery reliability or a consistent cost advantage over fine-tuning.

**Open-loop robotics scores are incomplete.** Predicting a recorded action or trajectory better is useful, but a policy changes its future observations when deployed. The driving report itself makes this distinction. A manipulation claim needs closed-loop success, recovery behavior and regressions across initial conditions.

## 6. A useful taxonomy for this project

These are working categories for our experiments, not a claim of standardized terminology.

| Category | Allowed information/operations | What it would show |
|---|---|---|
| Weight-only diagnosis | Weights, architecture and explicit observation/action semantics; no policy execution | Whether the editor can infer a failure from parameters |
| One-shot policy patch | Above plus a fixed failure record; one final edit, no iterative feedback | Direct analogue of the original car repair |
| Feedback-assisted weight repair | Fixed budget of trajectories, activations and tensor analysis; direct edits | Practical agent-assisted diagnosis and repair |
| Fitted model editing | Activation data plus regression or other parameter fitting | A broader editing method; report fitting and data explicitly |
| Activation/runtime intervention | Hooks, prompts, decoding, control logic or wrappers change | A useful baseline, but not a stored-weight-only result |
| Hybrid improvement | Weight/code edits followed by gradient training | Whether editing improves initialization or training efficiency |

The deployed policy artifact and the editing process both need labels. “Only weights changed at deployment” does not reveal whether finding those weights used a thousand rollouts, supervised fitting, or an LLM that already knew the environment’s source code.

## 7. Recommended first manipulation study

### Research question

**Can an LLM repair one repeatable manipulation failure while preserving existing skills, with a lower total discovery cost than practical alternatives?** Start with an interpretable failure and a reproducible simulator. This design is deliberately narrower than asking an editor to make a general manipulation policy better.

### Candidate order

| Priority | Target | Example failure to test | Why it is informative |
|---|---|---|---|
| 1 | Small state-based MLP controlling a simulated reach/push/grasp task | Action-axis sign/scale error, premature gripper closure, stall before a threshold | Known state semantics make diagnosis and exact comparisons tractable |
| 2 | Learned state-based policy with a natural recurring failure | Repeated approach without closing, overshoot followed by ineffective recovery | Tests whether success extends beyond deliberately planted bugs |
| 3 | Visual policy with frozen encoder and a compact action head | Consistent offset or timing problem with otherwise adequate perception | Localizes the editable subsystem while keeping visual evaluation relevant |
| 4 | Existing ACT, diffusion/flow policy or VLA action expert | A tightly specified failure in one skill, such as a cloth-folding stage | Closest to the user’s objective, but harder to localize and evaluate cheaply |

For the first study, select one existing checkpoint and task after inspecting the available policy and infrastructure. [Fetch PickAndPlace](https://robotics.farama.org/envs/fetch/pick_and_place/) is a candidate because it exposes state/goal information and Cartesian/gripper actions. Do not assume that a failure caused by missing perception or state information can be fixed by editing a downstream tensor. In a temporal policy, explicitly test episode-start history, frame/action alignment and chunk boundaries before editing.

Later stages need architecture-specific evaluation: [ACT](https://tonyzhaozh.github.io/aloha/) predicts action chunks, [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) samples action sequences through denoising, and [OpenVLA](https://openvla.github.io/) maps visual/language representations into action tokens. A change to one head may affect an entire rollout. Restricting edits to an output module is a useful experimental simplification, not an established sufficient approach.

### Experimental protocol

1. **Freeze the task and baseline.** Pin checkpoint, code, observation normalization, action scaling, simulator version and evaluation seeds. Verify the unedited policy under its intended inference settings.
2. **Partition evidence before editing.** Use separate diagnosis, development and locked evaluation sets. Split objects, poses and conditions as well as random seeds where possible. Keep successful behaviors in a protected set.
3. **Run distinct editor conditions.** Compare weight-only diagnosis, one-shot patching from a fixed trajectory record, and feedback-assisted editing. Keep their observations and budgets explicit. Ban fitting only in the conditions intended to test strict WSP.
4. **Require a concrete patch.** Save tensor names, indices, before/after hashes, changed-entry counts, relative norms, maximum magnitudes, an explanation and a rollback artifact. Preserve architecture and inference code for the weight-only condition.
5. **Use competitive baselines.** Include no edit, inference/configuration fixes, a hand-written controller or residual correction, an LLM-written external controller given identical evidence, matched-norm random edits, ordinary fine-tuning and suitable low-rank adaptation. The external-controller comparison tests whether embedding a straightforward rule in weights adds value beyond deriving the rule. Give baselines comparable data or account explicitly for differences.
6. **Evaluate closed-loop.** Measure task success, failure type, completion time, action oscillation, saturation and contact/collision outcomes. Include changed object poses, friction, sensor noise and initialization/history states relevant to the task.
7. **Repeat discovery.** Use several independent editor sessions, not just repeated evaluation of one successful patch. Log all proposals and rollbacks. Report editor-run success rate separately from rollout success.
8. **Check implementation fidelity.** Validate the exact saved checkpoint at deployment dtype, including quantization if used. Compare outputs before and after serialization; check finite values and action limits.

### Metrics and decision rule

Report absolute percentage-point improvement, paired success/failure changes and confidence intervals. Treat an episode as the basic rollout unit; correlated steps within one episode do not create independent evidence. Choose rollout counts using an initial variance/power estimate rather than a visually convenient demo count.

Count **all** discovery costs: editor tokens, CPU/GPU time, trajectory generation, activation storage, evaluation calls and human intervention. A small final patch is cheap to apply, but finding it may not be cheap.

Advance to a larger visual/VLA policy only if the effect persists on locked closed-loop evaluation, does not exceed a predeclared regression allowance, and survives comparison with the best inexpensive baseline. If a normalization correction or small residual controller solves the problem equally well, that is useful progress and an essential comparison.

### What would change our confidence most

- A public cloth-folding checkpoint diff, exact task protocol and paired evaluation logs behind the π0.5 claim.
- Closed-loop validation of the UniAD edit, with startup frames reported separately.
- Reliable edits on naturally occurring manipulation failures across multiple policy training seeds.
- A discovery-cost comparison against fine-tuning using the same demonstrations and evaluation budget.
- Evidence that one edit transfers across object geometries, initial states and protected skills without repeated retuning.

## 8. Search coverage and limitations

The investigation followed the supplied blog, its experiment folders, initial math prompt, negative-control documentation, checkpoint/model card and GitHub issue discussions. Searches covered the exact phrase and variations involving replication, experiments, failure, robotics, manipulation, driving, diffusion, vision and independent applications. Dedicated passes traced X and LinkedIn links, then followed first-person posts into blogs, repositories and preprints.

Search results and news summaries were used for discovery. Technical findings above are grounded in original authors’ posts, experiment records or firsthand audits. Reposts and repeated announcements were not counted as separate experiments. Some early articles saying there were no independent replications were already stale relative to the artifacts found here.

This is a broad public-source investigation, not an exhaustive census of X or LinkedIn. Login restrictions, deleted content, unindexed posts, changing repositories and unavailable media limit coverage. In particular, social percentages without episode counts or task definitions remain reported claims. Dates are stated where verified; uncertain dates are identified. New claims can be added without changing the evidence labels of older entries.


## November 5 addendum

The original survey above remains a dated snapshot. A new [Guo–Xie assessment](guo-xie.md) examines author-reported sparse edits across manipulation, world-action and humanoid policies. It separates numerical claims, task-fine-tuning and within-task transfer scope, verification gaps, and proposed next experiments. The updated [critical evidence map](../blog/recent_work_table.md) has 22 rows, including our completed study, with separate benefits and limitations. The V13 paper and blog explicitly qualify both targeted positive demonstrations and our unsuccessful restricted search; no new experiment was run for this addendum.
