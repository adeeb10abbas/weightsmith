"""New baseline rows must resolve to their actual recorded batch positions."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
sys.path.insert(0, str(RUNTIME))
from weightsmith.lab.batch_context_v2 import write_packet, sha

spec = importlib.util.spec_from_file_location('v2_development_under_test', RUNTIME/'scripts/study_development.py')
development = importlib.util.module_from_spec(spec);spec.loader.exec_module(development)


class DevelopmentPackets(unittest.TestCase):
    def fixture(self, root):
        (root/'v2_runtime_binding.json').write_text('{"fixture":true}\n')
        runtime_sha = sha(root/'v2_runtime_binding.json')
        config = {'task':'StackYellowOnRedTask','instruction':'Stack the yellow block on the red block'}
        rows=[]
        for block, first in enumerate((30016,30032)):
            packet_id = f'{block:032x}'
            requests=[]
            for slot in range(first,first+16):
                requests.append({'observation/exterior_image_1_left':np.zeros((224,224,3),np.uint8),
                                 'observation/wrist_image_left':np.zeros((224,224,3),np.uint8),
                                 'observation/joint_position':np.arange(7,dtype=np.float32),
                                 'observation/gripper_position':np.zeros(1,np.float32),
                                 'prompt':config['instruction'],
                                 'sampling_seed':((slot//16)*64+slot%16)*100})
            actions=np.zeros((16,15,8),np.float64)+0.1
            packet=write_packet(root/'data/batch_packets'/packet_id,requests,actions,
                        {'checkpoint_manifest_sha256':'a'*64,'runtime_manifest_sha256':runtime_sha,
                         'native_server_sha256':'b'*64,'policy_fingerprint':'base'}, variant='fixed16_new_runtime')
            for position, (slot, request) in enumerate(zip(range(first,first+16),requests)):
                obs=root/f'obs-{slot}.npz'
                np.savez_compressed(obs,base_img=request['observation/exterior_image_1_left'][None],
                    wrist_img=request['observation/wrist_image_left'][None],joint=request['observation/joint_position'][None],
                    gripper=request['observation/gripper_position'][None],seed=np.asarray([request['sampling_seed']],np.int64),
                    step=np.asarray([0]),prompt=np.asarray([request['prompt']]),actions=actions[position:position+1].astype(np.float32),
                    batch_packet_id=np.asarray([packet_id]),batch_packet_sha256=np.asarray([packet['packet_sha256']]),
                    batch_position=np.asarray([position]),batch_real_count=np.asarray([16]),batch_padded_count=np.asarray([16]))
                rows.append({'slot':slot,'ok':True,'fingerprint':'base','gripper_threshold':0.5,
                             'task_name':config['task'],'prompt':config['instruction'],'success':False,'steps':1,'obs':str(obs)})
        return rows, config

    def test_full_linked_baseline_then_pixel_seed_or_position_corruption(self):
        for field in ('base_img','seed','batch_position'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as t:
                root=Path(t);old=development.C.PROJECT_ROOT;development.C.PROJECT_ROOT=root
                try:
                    rows,config=self.fixture(root)
                    server={'fingerprint':'base','workspace':None}
                    development.validate_block(rows,server,.5,config)
                    p=Path(rows[0]['obs'])
                    with np.load(p,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
                    arrays[field].flat[0]+=1
                    np.savez_compressed(p,**arrays)
                    with self.assertRaises(ValueError):development.validate_block(rows,server,.5,config)
                finally:development.C.PROJECT_ROOT=old

    def test_partial_baseline_is_not_complete(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=development.C.PROJECT_ROOT;development.C.PROJECT_ROOT=root
            try:
                rows,config=self.fixture(root)
                with self.assertRaises(development.IncompleteBlockError):
                    development.validate_block(rows[:-1],{'fingerprint':'base','workspace':None},.5,config)
            finally:development.C.PROJECT_ROOT=old


if __name__=='__main__':unittest.main()
