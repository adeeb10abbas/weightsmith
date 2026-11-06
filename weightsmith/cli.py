"""The ``weightsmith`` command.

    weightsmith edit export CANDIDATE_DIR -o EDIT.safetensors [--rank auto|N] [--lossy]
    weightsmith edit info EDIT [--json]
    weightsmith edit apply EDIT --workspace BASE_EXPORT --out NEW_CANDIDATE_DIR
    weightsmith edit apply EDIT --arrays IN.npz --out OUT.npz
    weightsmith mcp [--root WSP_ROOT] [--host WSP_HOST]      (needs: pip install 'weightsmith[mcp]')
    weightsmith distill run CONFIG.json [--dry-run]          (Experimental; see docs/DISTILL.md)
    weightsmith distill render CONFIG.json [--script build|compare|select]

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="weightsmith",
        description="Tools for LLM weight editing of robot policies: portable edit files, an MCP server and "
                    "(experimental) closed-form instruction distillation.")
    ap.add_argument("--version", action="version", version=f"weightsmith {_version()}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{edit,mcp,distill}")

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
