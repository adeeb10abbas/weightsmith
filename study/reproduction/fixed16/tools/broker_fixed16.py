#!/usr/bin/env python3
"""Reuse the sealed fixed16 broker and remote dispatch; no portability bypass."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bundle/weightsmith/extensions/batch_capture_v2/runtime/drivers'))
from study_broker import Broker, serve


class ReadinessBroker(Broker):
    def call(self, tool, args):
        if self.cfg.get('readiness_only') and tool not in {'workspace_list', 'workspace_read'}:
            raise ValueError('Readiness permits only local workspace listing and brief reading')
        return super().call(tool, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); broker = ReadinessBroker(args.config)
    try: serve(broker)
    finally: broker.release()
