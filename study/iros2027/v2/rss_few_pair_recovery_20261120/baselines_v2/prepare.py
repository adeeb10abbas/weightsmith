"""Build frozen inference offsets; never inspect development/test outcomes."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import artifacts as A


def source_metadata(paths):
    return {str(Path(p).resolve()): A.sha(p) for p in paths}


def residual(args):
    original = Path(args.original_capture).resolve()
    clarified = Path(args.teacher_capture).resolve()
    oi, ci = A.read(original/'index.json'), A.read(clarified/'index.json')
    if oi['vlm_layers'] != ci['vlm_layers'] or args.layer not in oi['vlm_layers']:
        raise ValueError('Paired captures must contain the same requested residual layer')
    if len(oi['items']) != len(ci['items']):
        raise ValueError('Paired captures have different source counts')
    indices = args.indices
    if len(indices) != 3 or len(set(indices)) != 3 or any(i < 0 or i >= len(oi['items']) for i in indices):
        raise ValueError('Require exactly3 distinct captured source indices')
    selected = []
    for i in indices:
        a, b = oi['items'][i], ci['items'][i]
        for key in ('episode', 'k', 'seed', 'sim_step'):
            if a[key] != b[key]:
                raise ValueError('Original and clarified capture identities differ: '+key)
        if a['prompt'] == b['prompt']:
            raise ValueError('A correction requires different original and clarified instructions')
        selected.append({'capture_index': i, 'episode': a['episode'], 'decision_index': a['k'],
                         'seed': a['seed'], 'sim_step': a['sim_step'],
                         'original_prompt': a['prompt'], 'clarified_prompt': b['prompt']})
    if len({r['episode'] for r in selected}) != 3:
        raise ValueError('Source triplet must contain3 distinct episodes')
    with np.load(original/'acts.npz', allow_pickle=False) as a, np.load(clarified/'acts.npz', allow_pickle=False) as b:
        direction = A.residual_direction(a['vlm.img_resid_out'], b['vlm.img_resid_out'], indices, oi['vlm_layers'].index(args.layer))
    if not np.any(direction):
        raise ValueError('Paired residual direction is identically zero')
    paths = [original/'acts.npz',original/'index.json',clarified/'acts.npz',clarified/'index.json',Path(__file__)]
    if args.source_manifest:
        paths.append(Path(args.source_manifest))
    construction = {'kind': 'constant_per_cell_residual_offset', 'source_observations': 3,
                    'selected_sources': selected, 'source_capture_indices': indices,
                    'averaging': 'mean over the3 selected source observations, preserving32 image-cell positions',
                    'support': 'first two cameras; each of16 row-major4x4 cells broadcasts to its4x4patch tokens',
                    'target': 'clarified minus original post-block residual, not a residual increment',
                    'source_dependent': True, 'heldout_used': False, 'development_outcomes_used': False,
                    'artifacts': source_metadata(paths)}
    return A.write_artifact(args.output, 'residual_cells', direction, construction, strength=args.strength, layer=args.layer)


def tokenizer_metadata(path):
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=str(path))
    prefix = sp.encode('Task:', add_bos=True)
    # Encoding a delimiter independently adds a leading-space token. Obtain the
    # exact in-context delimiter from a known valid native prompt instead.
    sample = sp.encode('Task: X, State: 0;\nAction: ', add_bos=True)
    comma = sp.piece_to_id(',')
    matches = [i for i, token in enumerate(sample) if token == comma]
    if len(matches) != 1:
        raise ValueError('Cannot identify the native state delimiter')
    delim = sample[matches[0]:matches[0]+3]
    if sp.decode(delim).strip() != ', State:':
        raise ValueError('Unexpected native state-delimiter tokenization')
    descriptor={'model_path': str(Path(path).resolve()), 'model_sha256': A.sha(path),
                'prefix_ids': prefix, 'delimiter_ids': delim, 'max_token_len': 200,
                'scope': 'instruction tokens only; Task prefix, State suffix, padding and image tokens unchanged'}
    return sp,A.upgrade_tokenization(descriptor)


def encoded_instruction(sp, descriptor, prompt):
    prompt = prompt.strip().replace('_', ' ').replace('\n', ' ')
    tokens = sp.encode('Task: '+prompt+', State: 0;\nAction: ', add_bos=True)
    if len(tokens) > descriptor['max_token_len']:
        raise ValueError('Prompt too long for native token buffer')
    valid = np.arange(descriptor['max_token_len']) < len(tokens)
    tokens = np.pad(np.asarray(tokens, dtype=np.int32), (0, descriptor['max_token_len']-len(tokens)))
    mask = A.instruction_mask(tokens, valid, descriptor['prefix_ids'], descriptor['delimiter_ids'],descriptor['separator_token_ids'])
    return tokens, mask


def embedding(args):
    # GPU and memory selection must precede the first JAX import.
    from weightsmith import config as C
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                      XLA_PYTHON_CLIENT_MEM_FRACTION='.85')
    path = Path(args.tokenizer) if args.tokenizer else C.OPENPI_DATA_HOME/'big_vision/paligemma_tokenizer.model'
    sp, descriptor = tokenizer_metadata(path)
    original, omask = encoded_instruction(sp, descriptor, args.original_prompt)
    clarified, cmask = encoded_instruction(sp, descriptor, args.clarified_prompt)
    import jax.numpy as jnp
    from weightsmith.locks import GpuLock
    from weightsmith.batched_server import Server
    with GpuLock(args.gpu):
        server = Server()
        embedded = np.asarray(server.policy._model.PaliGemma.llm(jnp.asarray(np.stack([original,clarified])), method='embed')).astype(np.float64)
    direction = (embedded[1,cmask].mean(axis=0) - embedded[0,omask].mean(axis=0)).astype(np.float32)
    if not np.any(direction):
        raise ValueError('Instruction embedding direction is identically zero')
    paths = [Path(__file__), path, Path(C.OPENPI_DIR)/'src/openpi/models/gemma.py']
    if args.source_manifest:
        paths.append(Path(args.source_manifest))
    construction = {'kind': 'mean_instruction_embedding_offset', 'original_prompt': args.original_prompt,
                    'clarified_prompt': args.clarified_prompt, 'source_observations': 0,
                    'source_dependent': False, 'replication_unit': 'one shared candidate per goal',
                    'averaging': 'unweighted mean over instruction tokens, clarified minus original',
                    'original_instruction_token_ids': original[omask].tolist(),
                    'clarified_instruction_token_ids': clarified[cmask].tolist(),
                    'original_token_count': int(omask.sum()), 'clarified_token_count': int(cmask.sum()),
                    'checkpoint': str(C.CHECKPOINT), 'heldout_used': False, 'development_outcomes_used': False,
                    'artifacts': source_metadata(paths)}
    return A.write_artifact(args.output, 'embedding_mean', direction, construction,
                            strength=args.strength, tokenization=descriptor)


def strength(args):
    spec, direction = A.load_artifact(args.source, verify_sources=True)
    construction = {**spec['construction'], 'parent_artifact': str(Path(args.source).resolve()),
                    'parent_artifact_sha256': A.sha(Path(args.source)/'baseline.json'),
                    'strength_predeclared': True}
    return A.write_artifact(args.output, spec['method'], direction, construction,
              strength=args.strength, layer=spec['layer'], tokenization=spec['tokenization'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('residual');p.add_argument('--original-capture', required=True);p.add_argument('--teacher-capture', required=True)
    p.add_argument('--indices', nargs=3, type=int, required=True);p.add_argument('--layer', type=int, default=16)
    p.add_argument('--source-manifest');p.add_argument('--output',required=True);p.add_argument('--strength',type=float,default=1.)
    p = sub.add_parser('embedding');p.add_argument('--original-prompt',required=True);p.add_argument('--clarified-prompt',required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),required=True);p.add_argument('--tokenizer');p.add_argument('--source-manifest')
    p.add_argument('--output',required=True);p.add_argument('--strength',type=float,default=1.)
    p = sub.add_parser('strength');p.add_argument('--source',required=True);p.add_argument('--strength',type=float,required=True);p.add_argument('--output',required=True)
    args = ap.parse_args();start=time.monotonic();result=globals()[args.command](args)
    print({'method':result['method'],'output':str(Path(args.output).resolve()),'strength':result['strength'],
           'wall_seconds':time.monotonic()-start,'direction_l2':result['direction_l2']},flush=True)


if __name__ == '__main__':
    main()
