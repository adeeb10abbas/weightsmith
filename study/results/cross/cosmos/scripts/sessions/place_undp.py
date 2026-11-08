"""Keep the t8und-p practice checks behind the first held-out side-effect pairs (coordinator, 06:15 ET): when
session_t8undp.sh puts T5-Et8undp (GPU0) / T8-Et8undp (GPU1) at the front of a queue, move it right after the
T5 / T3 held-out jobs. Polls every 2 s; exits once both are placed."""
import json, os, time
RULES = [("jobs/queue_gpu0.json", "T5-Et8undp-s30000", "R1-Et8und-s96000"),
         ("jobs/queue.json", "T8-Et8undp-s30000", "T3-Et8und-s96016")]
done = set()
while len(done) < len(RULES):
    for f, tag, after in RULES:
        if tag in done:
            continue
        q = json.load(open(f))
        tags = [j["tag"] for j in q]
        if tag not in tags:
            continue
        j = q.pop(tags.index(tag))
        tags = [x["tag"] for x in q]
        q.insert(tags.index(after) + 1, j)
        json.dump(q, open(f + ".tmp", "w"), indent=1)
        os.replace(f + ".tmp", f)
        print(time.strftime("%H:%M:%S"), "placed", tag, "after", after, flush=True)
        done.add(tag)
    time.sleep(2)
