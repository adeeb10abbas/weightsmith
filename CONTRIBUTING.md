# Contributing

Thanks for helping. The most useful contributions are new editor runs (including negative results), new tasks, and integrations.

## Submit a benchmark result

1. Run an editor under [BENCHMARK.md](BENCHMARK.md) and keep held-out slots and results away from it.
2. Score its one frozen candidate, or `base`, with `weightsmith/heldout.py` on at least 64 unseen slots.
3. Open a pull request that adds `results/submissions/<date>-<editor>-<task>/` with the files listed in [results/README.md](results/README.md), plus your row in [LEADERBOARD.md](LEADERBOARD.md). Entries are marked self-reported until a maintainer re-scores them.

Rejected edits and `base` submissions are welcome: they map what weight edits cannot fix. You can also open a [leaderboard submission](https://github.com/adeeb10abbas/weightsmith/issues/new?template=leaderboard_submission.yml) issue first.

## Change the code

```bash
pip install -e '.[dev,mcp,hub]'
ruff check .                 # lint (the research runtime and frozen study files are excluded)
python -m pytest tests -q    # CPU only: no GPU, simulator or network
```

- **Keep the runtime's behaviour.** `weightsmith/owsp.py`, the eval pool, `heldout.py` and the drivers produced the published results; changes there need a reason, a test and a note in the pull request.
- **Do not edit frozen material.** `study/reproduction/` is sealed and `study/results/data/` is frozen. Add new results next to them instead.
- **Test what you advertise.** New commands, integrations and documented numbers need a test or a pointer to the result file they come from. Anything not built yet is labelled Roadmap.
- Report bugs with the [bug report](https://github.com/adeeb10abbas/weightsmith/issues/new?template=bug_report.yml) form.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
