import os
print(len(NAMES))
print([n for n in NAMES if not n.startswith('vision') and not n.startswith('language.blocks') and not n.startswith('action.blocks')])
print(os.listdir('/work'))
for d in ['/work/notes', '/work/scratch']:
    try:
        print(d, os.listdir(d))
    except Exception as e:
        print(d, e)
eps = episodes()
print(len(eps))
if eps:
    print(eps[0])
