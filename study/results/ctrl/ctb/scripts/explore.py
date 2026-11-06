print(type(NAMES))
try:
    print(len(list(NAMES)))
except Exception as ex:
    print('err', ex)
g = [k for k in globals().keys() if not k.startswith('__')]
print(g)
import os
print(os.listdir('/work'))
for d in ['/work/scratch', '/work/notes']:
    try:
        print(d, os.listdir(d))
    except Exception as ex:
        print(d, ex)
