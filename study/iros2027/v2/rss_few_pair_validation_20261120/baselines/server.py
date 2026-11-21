"""Fixed16 server for unchanged weights, static weight edits, and simple baselines.

The ordinary workspace load API accepts either a native weight-edit workspace
or a directory containing baseline.json and direction.npy. Base resets restore
the exact original native inference function as well as the model weights.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))


def sibling(name):
    key = '_fewpair_baseline_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


A = sibling('artifacts')
N = sibling('native')
from weightsmith import batched_server as base


class Server(base.Server):
    def __init__(self, config_name='pi05_droid_jointpos'):
        super().__init__(config_name)
        self._native_jitted = self.jitted
        self._baseline_functions = {}
        self.active_baseline = None
        self._instruction_tokenizer = None

    def load(self, workspace):
        self.jitted = self._native_jitted
        self.active_baseline = None
        self._instruction_tokenizer = None
        root = Path(workspace).resolve() if workspace else None
        if root is None or not (root/'baseline.json').is_file():
            return super().load(workspace)
        # A baseline must not inherit any preceding native weight edit.
        # The frozen rollout harness reads the first "loaded {JSON}" record.
        # Suppress only the intermediate reset, then log the actual baseline.
        was_disabled = base.log.disabled
        try:
            base.log.disabled = True
            super().load(None)
        finally:
            base.log.disabled = was_disabled
        spec, direction = A.load_artifact(root)
        import jax.numpy as jnp
        if spec['method'] == 'embedding_mean':
            tokenization = spec['tokenization']
            if A.sha(tokenization['model_path']) != tokenization['model_sha256']:
                raise ValueError('The instruction tokenizer differs from baseline construction')
            import sentencepiece
            self._instruction_tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenization['model_path'])
            prefix = tuple(tokenization['prefix_ids'])
            delimiter = tuple(tokenization['delimiter_ids'])
            key = (spec['method'], prefix, delimiter)
            if key not in self._baseline_functions:
                self._baseline_functions[key] = self.jax.jit(N.make_embedding_function(self.graphdef, prefix, delimiter))
        else:
            key = (spec['method'], spec['layer'])
            if key not in self._baseline_functions:
                self._baseline_functions[key] = self.jax.jit(N.make_residual_function(self.graphdef, spec['layer']))
        fn = self._baseline_functions[key]
        vector, strength = jnp.asarray(direction), jnp.asarray(spec['strength'], dtype=jnp.float32)
        self.jitted = lambda state, rng, observation, noise: fn(state, rng, observation, noise, vector, strength)
        self.active_baseline = spec
        self.report = {'workspace': str(root), 'baseline': spec, 'edited_tensors': {},
                       'fingerprint': A.sha(root/'baseline.json'), 'checkpoint_weights_modified': False,
                       'inference_intervention': True, 'kernel_batch': 16}
        base.log.info('loaded %s', json.dumps({k:v for k,v in self.report.items() if k!='edited_tensors'}, default=str))
        return self.report

    def infer_batch(self, obs_list):
        n = len(obs_list)
        if not 1 <= n <= 16:
            raise ValueError('Fixed16 inference requires1..16 requests')
        if self.active_baseline and self.active_baseline['method'] == 'embedding_mean':
            # Cheap CPU validation catches malformed/truncated instruction spans
            # before the compiled hook. It does not change the actual request.
            tokenization = self.active_baseline['tokenization']
            for obs in obs_list:
                prompt = obs['prompt'].strip().replace('_', ' ').replace('\n', ' ')
                tokens = self._instruction_tokenizer.encode('Task: '+prompt+', State: 0;\nAction: ', add_bos=True)
                if len(tokens) >= 200:
                    raise ValueError('Instruction leaves insufficient space for native state tokens')
                A.instruction_mask(tokens, [True]*len(tokens), tokenization['prefix_ids'], tokenization['delimiter_ids'])
        padded = [copy.deepcopy(obs) for obs in obs_list]
        padded.extend(copy.deepcopy(obs_list[0]) for _ in range(16-n))
        outputs = super().infer_batch(padded)
        kept = []
        for output in outputs[:n]:
            result = dict(output)
            result['policy_timing'] = dict(output.get('policy_timing', {}))
            result['policy_timing'].update(batch=n, requested_n=n, kernel_batch=16,
                    baseline_method=self.active_baseline['method'] if self.active_baseline else None)
            kept.append(result)
        return kept


def main():
    base.Server = Server
    base.main()


if __name__ == '__main__':
    main()
