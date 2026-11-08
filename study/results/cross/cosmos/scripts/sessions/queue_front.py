"""Insert jobs at the front of a pool queue file (after a given tag if present), atomically; skip tags already queued.
usage (from ~/cross/cosmos): python3 queue_front.py QUEUE.json [--after TAG] CODE:TAG:SLOT_START[:EDIT] ...
"""
import json
import os
import sys

sys.path.insert(0, "scripts")
from tasks import job  # noqa: E402

args = sys.argv[1:]
qf, after = args.pop(0), None
if args and args[0] == "--after":
    args.pop(0)
    after = args.pop(0)
q = json.load(open(qf))
tags = [x["tag"] for x in q]
i = tags.index(after) + 1 if after in tags else 0
for spec in args:
    code, tag, s, *e = spec.split(":")
    if tag in tags:
        continue
    q.insert(i, job(code, tag, int(s), 16, edit=e[0] if e else None))
    tags.insert(i, tag)
    i += 1
json.dump(q, open(qf + ".tmp", "w"), indent=1)
os.replace(qf + ".tmp", qf)
print(qf, [x["tag"] for x in q if not os.path.exists(f"runs/q/{x['tag']}.json")])
