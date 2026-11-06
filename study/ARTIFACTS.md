# Downloadable artifacts

All assets belong to the same private repository and require repository access. The [Astra study handoff release](https://github.com/adeeb10abbas/weightsmith/releases/tag/astra-study-2026-11-05) supplies the files below. Checksums and byte counts are in [artifact-checksums.json](provenance/artifact-checksums.json). Large baselines and videos are outside Git history.

| Asset | Use |
|---|---|
| `overleaf.zip` | Current compilation sources only, with all required figures and bibliography. Select XeLaTeX and `main.tex`. |
| `study-review.zip` | Full offline human review: current paper, blog, plots, videos, detailed technical evidence and historical handoff provenance. |
| `common-fixed16-baseline-evidence.tar.gz` | Sealed original-policy baseline and recorded packets, approximately 365 MB. Import only under the documented compatible-runtime rules. |
| `fixed16-source.tar.gz` | Exact sealed source package, also expanded in this checkout. |
| `SHA256SUMS` | SHA-256 of the four assets above. |

## Full review bundle

Download `study-review.zip`, extract it into an empty folder, and open `index.html`. Its internal links work offline. The compact Git blog sends large/raw evidence links here; the offline version retains the original file-level links. Historical package metadata in that bundle is provenance, not an alternative current manuscript. Use this checkout's start page for the current continuation route.

The review bundle is for human review and coauthor work. It includes Astra outcomes; keep it out of a fresh editor context for an independent editor-model comparison.

## Download commands

With GitHub CLI authenticated to your own account:

```sh
gh release download astra-study-2026-11-05 --repo adeeb10abbas/weightsmith --dir /your/download/folder
cd /your/download/folder
shasum -a 256 -c SHA256SUMS
```

Do not transfer credentials. Download the foundation checkpoint separately using the included pinned setup and verify all 26 checkpoint files. Full original raw experiment archives, failed attempts and the residual pod snapshot are retained on the owner's laptop; they are not silently replaced by the compact review.
