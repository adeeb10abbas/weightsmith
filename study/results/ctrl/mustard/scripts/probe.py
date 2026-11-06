import os, json
ws = json.load(open('/work/workspace.json'))
t = ws['tensors']
w = [k for k,v in t.items() if v.get('writable')]
nw = [k for k,v in t.items() if not v.get('writable')]
print('writable', len(w), 'nonwritable', len(nw))
import collections
c = collections.Counter(['.'.join(k.split('.')[:2]) for k in nw])
print(c.most_common(30))
print(w[:10])
print(len(NAMES))
