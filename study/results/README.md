# Results and reproducible plots

`data/` contains the copied measurement JSON/CSV and camera examples. `figures/` contains current exports. Data are unchanged from the completed study; no new trials are run by these scripts.

Install Matplotlib, NumPy and Pillow in your own environment. From this directory:

```sh
python3 scripts/build_figures.py
python3 scripts/plot_experiment_results.py
python3 scripts/plot_final_controls.py --output-dir figures/final-controls
```

The first script generates the scientific block diagram and baseline figures. The second generates original and fixed16 proposal/incumbent curves and identity-repeat plots; the third emits final-control figures and tables. Output goes to `figures/` and the companion `../blog/figures/`. Copy the selected PDF figures into `../paper/figures/` deliberately after reviewing them; the manuscript is not updated automatically. These are presentation tools for verified derived measurements, not raw-archive validators or simulation launchers. Rendering can vary with library/font versions; the supplied PDF was already checked.

The [portability manifest](../provenance/plot-script-portability.json) records source hashes and output-path adjustments. The full human review contains the fixed-input action-effect figures and other detailed audits. All attempted edits and identities remain visible. New teammate measurements should use a separate cohort file and explicit labels rather than replacing these frozen results.
