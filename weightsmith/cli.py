"""The ``weightsmith`` command.

    weightsmith edit export CANDIDATE_DIR -o EDIT.safetensors [--rank auto|N] [--lossy]
    weightsmith edit info EDIT [--json]
    weightsmith edit apply EDIT --workspace BASE_EXPORT --out NEW_CANDIDATE_DIR
    weightsmith edit apply EDIT --arrays IN.npz --out OUT.npz
    weightsmith mcp [--root WSP_ROOT] [--host WSP_HOST]      (needs: pip install 'weightsmith[mcp]')
    weightsmith distill run CONFIG.json [--dry-run]          (Experimental; see docs/DISTILL.md)
    weightsmith distill render CONFIG.json [--script build|compare|select]
    weightsmith stats RESULTS.json [RESULTS.json ...] [--halves] [--family-size M] [--json]
    weightsmith stats --ref FILE[:ARM] --cand FILE[:ARM]     (paired statistics; see docs/COOKBOOK.md)

These wrap the research runtime without changing it: ``bin/owsp``, ``weightsmith/heldout.py`` and
the drivers keep their own entry points (see README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("weightsmith")
    except Exception:  # noqa: BLE001 - running from a source checkout
        return "unknown (source checkout)"


def _rank(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        rank = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("rank must be 'auto' or a positive integer") from None
    if rank < 1:
        raise argparse.ArgumentTypeError("rank must be 'auto' or a positive integer")
    return rank


def _cmd_export(a: argparse.Namespace) -> int:
    from weightsmith import edits

    def progress(name: str) -> None:
        print(f"  encoding {name}", file=sys.stderr)

    edit = edits.from_candidate(a.candidate, rank=a.rank, lossy=a.lossy, progress=None if a.quiet else progress)
    if not edit.tensors:
        print(f"weightsmith: {a.candidate} changes no values; nothing to export", file=sys.stderr)
        return 1
    path = edit.save(a.output, container=a.format)
    print(edits.describe(edits.load(path)))  # read back what was written
    print(f"\nwrote {path} ({edits._human_bytes(path.stat().st_size)})")
    return 0


def _cmd_info(a: argparse.Namespace) -> int:
    from weightsmith import edits

    edit = edits.load(a.edit)
    if a.json:
        print(json.dumps(edit.metadata(), indent=1, sort_keys=True))
    else:
        print(edits.describe(edit))
    return 0


def _cmd_apply(a: argparse.Namespace) -> int:
    import numpy as np

    from weightsmith import edits

    edit = edits.load(a.edit)
    if a.workspace:
        report = edits.apply_to_workspace(edit, a.workspace, a.out, verify=not a.no_verify)
        print(f"wrote candidate {report['candidate']}: {report['changed_values']:,} values in "
              f"{report['tensors']} tensors")
        if report["fingerprint_matches"]:
            print(f"fingerprint {report['fingerprint']}: identical to the source candidate")
        else:
            print(f"fingerprint {report['fingerprint']} (source: {report['source_fingerprint'] or 'unknown'}); "
                  + ("every tensor matched its recorded sha256" if report["exact"] and not a.no_verify
                     else "values not verified against the source"))
        name = Path(report["candidate"]).name
        print(f"next, with WSP_ROOT set to the project that holds it: owsp diff {name} && owsp eval {name}")
        return 0
    with np.load(a.arrays, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    out = edit.apply(arrays, verify=not a.no_verify)
    with open(a.out, "wb") as f:  # a file object: np.savez must not append ".npz"
        np.savez(f, **out)
    print(f"wrote {a.out}: {len(edit.tensors)} of {len(out)} arrays edited ({edit.changed_values:,} values)")
    return 0


def _split_source(text: str) -> tuple[Path, str | None]:
    """``FILE`` or ``FILE:ARM`` -> (path, arm); a path that exists as given wins over the split."""
    p = Path(text)
    if p.exists() or ":" not in text:
        return p, None
    head, arm = text.rsplit(":", 1)
    if not arm or "/" in arm:
        return p, None
    return Path(head), arm


def _pick_arm(f, arm: str | None, default: str) -> dict:
    if arm is None:
        arm = default
    if arm not in f.arms and f.kind == "heldout" and f.summary and arm == f.summary.get("cand"):
        arm = "cand"  # a held-out file's candidate may be named by its candidate name, e.g. v2mi-f
    if arm not in f.arms:
        raise KeyError(f"{f.path} has no arm {arm!r}; it holds {', '.join(f.arms)}")
    return f.arms[arm]


def _cmd_stats(a: argparse.Namespace) -> int:
    from weightsmith import stats

    opts = dict(alpha=a.alpha, confidence=a.confidence, noise_discordance=a.noise_discordance, draws=a.draws,
                seed=a.seed)
    jobs = []  # (label, Comparison, file summary or None)
    if (a.ref is None) != (a.cand is None):
        raise ValueError("--ref and --cand go together")
    if a.ref is not None:
        if a.files:
            raise ValueError("give either result files or --ref/--cand, not both")
        (rp, ra), (cp, ca) = _split_source(a.ref), _split_source(a.cand)
        rf, cf = stats.read_arm_file(rp), stats.read_arm_file(cp)
        rdef, cdef = rf.default_pair()[0], cf.default_pair()[1]
        rname, cname = ra or rdef, ca or cdef
        if cname == "cand" and cf.summary and cf.summary.get("cand"):
            cname = str(cf.summary["cand"])
        if rname == cname:
            rname, cname = f"{rname} ({rp.stem})", f"{cname} ({cp.stem})"
        c = stats.compare(_pick_arm(rf, ra, rdef), _pick_arm(cf, ca, cdef), ref_name=rname, cand_name=cname, **opts)
        jobs.append((f"{a.ref} vs {a.cand}", c, None))
    else:
        if not a.files:
            raise ValueError("give at least one results file, or --ref FILE[:ARM] --cand FILE[:ARM]")
        names = a.arms.split(",") if a.arms else None
        if names is not None and len(names) != 2:
            raise ValueError("--arms takes two names: REF,CAND")
        for path in a.files:
            f = stats.read_arm_file(path)
            ra, ca = names or f.default_pair()
            cname = str(f.summary["cand"]) if f.kind == "heldout" and ca == "cand" and f.summary and \
                f.summary.get("cand") else ca
            c = stats.compare(_pick_arm(f, ra, ra), _pick_arm(f, ca, ca), ref_name=ra, cand_name=cname, **opts)
            jobs.append((str(path), c, f.summary if (ra, ca) == ("base", "cand") else None))
    family = max(len(jobs), a.family_size or 0)
    holm = stats.holm([c.p_value for _, c, _ in jobs] + [1.0] * (family - len(jobs)))[: len(jobs)]
    checks = [_summary_check(c, s) for _, c, s in jobs]
    if a.json:
        out = []
        for (label, c, _), h, chk in zip(jobs, holm, checks, strict=True):
            d = {"source": label, **c.to_dict(), "holm_p": h if family > 1 else None, "summary_check": chk}
            if a.halves:
                d["halves"] = [x.to_dict() for x in c.halves()]
            out.append(d)
        print(json.dumps({"family_size": family, "comparisons": out}, indent=1))
        return 0
    for i, ((label, c, summary), chk) in enumerate(zip(jobs, checks, strict=True)):
        if i:
            print()
        print(f"[{label}]")
        print(c.report())
        if a.halves:
            first, second = c.halves()
            order = bool(summary and summary.get("order"))  # heldout.py: base first, then cand first
            for tag, h in (("first half", first), ("second half", second)):
                when = (" (base ran first)" if tag == "first half" else f" ({c.cand_name} ran first)") if order else ""
                print(f"  {tag + when}: {h.ref_success}/{h.n} -> {h.cand_success}/{h.n}, fixed {h.fixed}, "
                      f"broke {h.broke}, exact p = {stats.format_p(h.p_value)}")
        if chk is not None:
            print(f"  check: {chk}")
    if family > 1:
        pad = family - len(jobs)
        print(f"\nHolm correction over {family} comparisons"
              + (f" ({pad} not given here count as p = 1)" if pad else "") + ":")
        for (label, c, _), h in zip(jobs, holm, strict=True):
            print(f"  p = {stats.format_p(c.p_value):>8}  Holm {stats.format_p(h):>8}  {'*' if h < a.alpha else ' '} "
                  f"{label}")
    return 0


def _summary_check(c, summary: dict | None) -> str | None:
    """Compare a recomputed comparison with the summary a heldout.py results.json stores."""
    if not summary:
        return None
    want = {"paired_n": c.n, "base_success": c.ref_success, "cand_success": c.cand_success, "fixed": c.fixed,
            "broke": c.broke}
    bad = {k: (summary[k], v) for k, v in want.items() if k in summary and summary[k] != v}
    p = summary.get("mcnemar_exact_p")
    if p is not None and abs(float(p) - c.p_value) > 1e-9 * max(1.0, abs(c.p_value)) + 1e-15:
        bad["mcnemar_exact_p"] = (p, c.p_value)
    if bad:
        diffs = ", ".join(f"{k} {s} (file) vs {v}" for k, (s, v) in bad.items())
        return f"DIFFERS from the file's own summary: {diffs}"
    return "matches the file's own summary (n, successes, fixed, broke, p)"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="weightsmith",
        description="Tools for LLM weight editing of robot policies: portable edit files, an MCP server and "
                    "(experimental) closed-form instruction distillation.")
    ap.add_argument("--version", action="version", version=f"weightsmith {_version()}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{edit,mcp,distill,stats}")

    edit = sub.add_parser("edit", help="export, inspect and apply portable weight edits",
                          description="Portable weight edits: one self-describing file per candidate.")
    esub = edit.add_subparsers(dest="edit_command", required=True, metavar="{export,info,apply}")

    p = esub.add_parser("export", help="write an owsp candidate as one edit file",
                        description="Encode a candidate's edits/ against its base export. Run it on the GPU host.")
    p.add_argument("candidate", help="candidate directory, e.g. $WSP_ROOT/cands/v2mi-f")
    p.add_argument("-o", "--output", required=True, type=Path, help="output file (.safetensors, or .npz)")
    p.add_argument("--rank", type=_rank, default=None,
                   help="store each 2-D+ delta as low-rank factors: 'auto' picks the smallest exact encoding, "
                        "N forces rank N (default: dense or sparse, whichever is smaller)")
    p.add_argument("--lossy", action="store_true",
                   help="with --rank N: drop the exact fix-ups and report the values that are then off")
    p.add_argument("--format", choices=("safetensors", "npz"), default=None,
                   help="container (default: from the output suffix; safetensors unless .npz)")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-tensor progress")
    p.set_defaults(func=_cmd_export)

    p = esub.add_parser("info", help="summarize an edit file")
    p.add_argument("edit", type=Path)
    p.add_argument("--json", action="store_true", help="print the full metadata as JSON")
    p.set_defaults(func=_cmd_info)

    p = esub.add_parser("apply", help="apply an edit to a base export or to arrays",
                        description="Apply an edit. --workspace writes a new owsp candidate directory on top of "
                                    "a base export; --arrays edits an .npz of arrays keyed by tensor name.")
    p.add_argument("edit", type=Path)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--workspace", type=Path, help="base export ($BASE_WORKSPACE) or any candidate made from it")
    target.add_argument("--arrays", type=Path, help=".npz of base arrays keyed by logical tensor name")
    p.add_argument("--out", required=True, type=Path, help="new candidate directory, or output .npz")
    p.add_argument("--no-verify", action="store_true", help="skip the sha256 checks of base and result")
    p.set_defaults(func=_cmd_apply)

    # `weightsmith mcp ...` is dispatched in main() so that its own options reach the server's parser.
    sub.add_parser("mcp", help="run the MCP server (stdio) over the owsp tools; see `weightsmith mcp --help`")
    sub.add_parser("distill", help="EXPERIMENTAL: distil a known-good rewording into the weights in closed form "
                                   "(build step proven bit-exact; select/capture/check only dry-run); "
                                   "see `weightsmith distill --help`")

    p = sub.add_parser(
        "stats", help="paired statistics for two arms run on the same slots (exact McNemar, Wilson, bootstrap)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compare two arms run on the same simulator slots: success rates with Wilson intervals, the\n"
                    "four paired outcome types, the paired difference with a bootstrap interval, the exact McNemar\n"
                    "test, and the smallest difference that many pairs can detect.",
        epilog="inputs:\n"
               "  weightsmith/heldout.py results.json or partial.json   arms base and cand\n"
               "  control-arm results {\"per_arm\": {arm: {slot: {ok, success}}}}   arms by name\n"
               "  CSV with a slot column and one column per arm (1/0, true/false or success/fail)\n\n"
               "examples:\n"
               "  weightsmith stats heldout/p1-RubiksCubeThenBananaTask-91008-64/results.json --halves\n"
               "  weightsmith stats run_task1.json run_task2.json --family-size 7      # Holm over 7 tasks\n"
               "  weightsmith stats outcomes.csv --arms base,v2mi-f\n"
               "  weightsmith stats --ref v2mi-f-results.json:base --cand ctl1-results.json:rand\n\n"
               "Reading the output: docs/COOKBOOK.md (paired evaluation).")
    p.add_argument("files", nargs="*", type=Path, help="results files, one comparison each")
    p.add_argument("--arms", metavar="REF,CAND", help="the two arms to compare in each file (default: base,cand "
                   "for heldout.py files, else the file's two arms)")
    p.add_argument("--ref", metavar="FILE[:ARM]", help="reference arm, for a comparison across two files")
    p.add_argument("--cand", metavar="FILE[:ARM]", help="candidate arm, for a comparison across two files")
    p.add_argument("--halves", action="store_true",
                   help="also report each half of the slot range (heldout.py alternates run order by half)")
    p.add_argument("--family-size", type=int, metavar="M", default=None,
                   help="Holm-correct over M tests; tests not given count as p = 1 (default: the files given)")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    p.add_argument("--confidence", type=float, default=0.95, help="interval coverage (default 0.95)")
    p.add_argument("--noise-discordance", type=float, default=0.3, metavar="Q",
                   help="share of pairs that flip between identical reruns, for the detectable-change line (0.3)")
    p.add_argument("--draws", type=int, default=20000, help="bootstrap draws (default 20000)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    p.set_defaults(func=_cmd_stats)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["mcp"]:
        from weightsmith import mcp_server

        return mcp_server.main(argv[1:])
    if argv[:1] == ["distill"]:
        from weightsmith import distill

        return distill.main(argv[1:], prog="weightsmith distill")
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (OSError, ValueError, KeyError) as e:
        print(f"weightsmith: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
