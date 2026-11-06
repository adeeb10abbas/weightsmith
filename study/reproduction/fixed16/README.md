# Fixed16 teammate handoff — version 1

This is a **separate, model-neutral source extension** for an explicitly identified editor deployment, including an Opus deployment if available to the teammate. It carries the batch-aware manipulation runtime, not experiment outcomes. It supersedes the original handoff's activation-capture path for a newly enrolled fixed16 study; it does not modify that package or any original experiment.

**Status: source and CPU packaging only.** No native GPU validation, teammate baseline, provider readiness, or editor search is certified by this package. The original fixed16 validation was still pending when this handoff was prepared. Do not interpret source hashes or CPU tests as proof of numerical parity. A later validation receipt must identify its actual host, binding, and source revision.

| Included | Excluded |
|---|---|
| All 77 sealed fixed16 sources/design files, byte for byte | Checkpoint weights and credentials |
| Original protocol, resource/precision/closure addenda, native-binding reference | Other editors' outcomes, notebooks, traces, selections or activations |
| Dependency commits, patches, locks, 26-file checkpoint checksum manifest | Development or held-out observations, videos or result tables |
| Exact-model CLI adapter derived from the existing generic handoff | Live provider validation or claimed Opus results |
| Packet/schema instructions and CPU integrity checks | Automatic model/GPU/remote execution during packaging |

Start with [SETUP.md](SETUP.md), [MODEL_ENROLLMENT.md](MODEL_ENROLLMENT.md), and [PACKETS.md](PACKETS.md). Only the exact sealed `BRIEF.md` and adaptive `MODE.md`, plus the separately verified same-runtime base evidence, belong in a scientific editor context. Keep this package, operator documents, protocol test slots, binding files and provider configuration outside the editor workspace.

The source seal is `017e7ab14bdfd330bb11b7addb23f6a04fa4a07374442e41aa253617e4def754`. The original protocol remains an archival reference with its originally assigned model. A teammate's different model receives a **new enrollment record**, not a rewritten historical protocol. The new adapter points to the sealed v2 broker and dispatcher; it does not use the original handoff's portable remote dispatcher.

Verify the extracted package and archive using:

```sh
python3 tools/verify_package.py
python3 tools/verify_package.py --archive /absolute/path/handoff_fixed16_v1.tar.gz
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

The exact sealed dispatcher expects a dedicated two-GPU deployment with devices 0 and 1 and the canonical project name `projects/rq123-v2-adaptive-r1`. Using a different root directory is supported by its relative layout. A one-GPU dispatcher, other project names, or changed scientific sources require a separate tested portability amendment; changing configuration alone does not establish support.

The package's own provider adapter is CPU tested only. Its live CLI flags, model attribution and tool isolation must pass a separate readiness probe on the teammate's machine. No authentication material is supplied, requested in files, or transferred.
