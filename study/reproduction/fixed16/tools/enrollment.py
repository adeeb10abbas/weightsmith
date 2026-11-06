"""Model-neutral, local-only gate before using the separately enrolled adapter."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_SHA = '017e7ab14bdfd330bb11b7addb23f6a04fa4a07374442e41aa253617e4def754'
PROTOCOL_SHA = 'bb55e5202a9cd4508a9678a8c72e9ede026a2166cf7bc8156ee9c4d2d35ee796'
CHECKPOINT_SHA = 'c5e21f3480d1a29e9230ac3963ebd20dcb5aea00c1a2f773abea00c7bdcb5061'
HASH = re.compile(r'^[0-9a-f]{64}$')
SLOTS = list(range(30016, 30048))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or not HASH.fullmatch(str(ref['sha256'])):
        raise ValueError('A local path and exact SHA-256 are required')
    path = Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or sha(path) != ref['sha256']:
        raise ValueError('Missing or changed enrollment evidence')
    return json.loads(path.read_text())


def check_enrollment(config, *, require_runtime=True, fresh_workspace=True):
    record = checked_ref(config['enrollment_ref'])
    for key, expected in {'schema_version': 1, 'record_kind': 'fixed16-teammate-enrollment',
                          'source_manifest_sha256': SOURCE_SHA, 'parent_protocol_sha256': PROTOCOL_SHA,
                          'provider': config['provider'],
                          'model_id': config['model'], 'effort': config['effort'], 'mode': 'adaptive',
                          'other_editor_outcomes_in_context': False, 'heldout_results_seen': False}.items():
        if record.get(key) != expected:
            raise ValueError('Enrollment mismatch: ' + key)
    frozen = datetime.fromisoformat(record['frozen_at_utc'])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError('A real prior UTC enrollment timestamp is required')
    if not config.get('effort') or '<' in config['effort']:
        raise ValueError('An explicit supported provider effort setting is required')
    if config.get('mode') != 'adaptive' or record.get('independent_runs') != 1:
        raise ValueError('This extension enrolls one separate three-proposal adaptive run')
    if record.get('budget_caps') != {'candidate_episodes': 96, 'activation_items': 96, 'wall_seconds': 7200, 'proposals': 3}:
        raise ValueError('Different budgets require a separately versioned enrollment')
    if not isinstance(record.get('enrollment_id'), str) or not record['enrollment_id'].strip() or '<' in record['enrollment_id']:
        raise ValueError('Explicit new enrollment identity required')
    remote = PurePosixPath(config['remote_repo'])
    if not remote.is_absolute() or '..' in remote.parts or remote.parts[-3:] != ('extensions','batch_capture_v2','runtime'):
        raise ValueError('remote_repo must select the sealed fixed16 fork')
    isolation = remote.parents[2]
    if PurePosixPath(config['remote_root']) != isolation / 'projects/rq123-v2-adaptive-r1':
        raise ValueError('The sealed dispatcher requires its canonical fresh project name')
    if not config.get('host') or config['host'].startswith('-'):
        raise ValueError('Explicit teammate SSH host required')
    agent = Path(config['agent_dir'])
    if agent.is_symlink() or not agent.is_dir():
        raise ValueError('A real dedicated agent directory is required')
    if fresh_workspace and {path.name for path in agent.iterdir()} != {'BRIEF.md','MODE.md'}:
        raise ValueError('Fresh agent workspace must contain only BRIEF.md and MODE.md')
    brief = PACKAGE/'bundle/weightsmith/operator/rq123-v2/design/BRIEF.md'
    mode = PACKAGE/'bundle/weightsmith/operator/rq123-v2/design/MODE.md'
    if any((agent/name).is_symlink() or sha(agent/name) != sha(source)
           for name, source in [('BRIEF.md', brief), ('MODE.md', mode)]):
        raise ValueError('Use the unchanged sealed brief and adaptive mode')
    output = {'enrollment_sha256': config['enrollment_ref']['sha256'], 'enrollment_id': record['enrollment_id'],
              'source_manifest_sha256': SOURCE_SHA, 'runtime_evidence_checked': require_runtime}
    if not require_runtime:
        return output
    binding = checked_ref(config['native_binding_ref'])
    validation = checked_ref(config['capture_validation_ref'])
    manifest = checked_ref(config['base_manifest_ref'])
    for key, expected in {'capture_protocol_id': 'wsp-batch-capture-v2-fixed16',
                          'runtime_variant': 'fixed16_new_runtime', 'source_manifest_sha256': SOURCE_SHA,
                          'checkpoint_manifest_sha256': CHECKPOINT_SHA,
                          'protocol_sha256': PROTOCOL_SHA}.items():
        if binding.get(key) != expected: raise ValueError('Native binding mismatch: ' + key)
    if validation.get('status') != 'passed' or validation.get('runtime_binding_sha256') != config['native_binding_ref']['sha256']:
        raise ValueError('Actual passing GPU validation bound to this runtime is required')
    marker=validation.get('capture_validation',{}); verified=validation.get('source_checkpoint_verification',{})
    if (validation.get('real_batch_counts') != [1,2,4,8,16] or validation.get('padded_batch_count') != 16
        or marker.get('protocol_id') != 'wsp-batch-capture-v2-fixed16' or marker.get('status') != 'passed'
        or marker.get('all_real_positions_checked') is not True or marker.get('zero_skipped') is not True
        or marker.get('parity_tolerance') != .02 or marker.get('real_positions_checked') != 31
        or marker.get('unique_packets_checked') != 5 or verified.get('source_manifest_sha256') != SOURCE_SHA
        or verified.get('verified_source_files') != 77 or verified.get('verified_checkpoint_files') != 26):
        raise ValueError('Full native/identity/instrumented validation provenance is required')
    if manifest.get('slots') != SLOTS or manifest.get('threshold') != .5 or manifest.get('protocol') != 'wsp-manipulation-batch-aware-v2':
        raise ValueError('A complete new fixed16 development base is required')
    if manifest.get('files', {}).get('v2_runtime_binding.json') != config['native_binding_ref']['sha256']:
        raise ValueError('Base cohort uses another runtime binding')
    path = Path(config['base_records'])
    if path.is_symlink() or not path.is_file() or sha(path) != manifest.get('files', {}).get('data/episodes/index.jsonl'):
        raise ValueError('Base index must be the exact newly frozen fixed16 cohort')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 32 or {r['slot'] for r in rows} != {f's{i}' for i in SLOTS}:
        raise ValueError('Exactly the fixed16 development slots are required')
    if any(r.get('cand') != 'base' or r.get('fingerprint') != 'base' or type(r.get('success')) is not bool for r in rows):
        raise ValueError('Invalid original-policy base records')
    output.update(native_runtime_binding_sha256=config['native_binding_ref']['sha256'],
                  capture_validation_sha256=config['capture_validation_ref']['sha256'],
                  base_manifest_sha256=config['base_manifest_ref']['sha256'], base_index_sha256=sha(path))
    return output
