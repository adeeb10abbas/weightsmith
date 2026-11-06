import os
print(type(NAMES))
try:
    n = NAMES() if callable(NAMES) else NAMES
    print(len(n)); print(list(n)[:30])
except Exception as e:
    print('err', e)
g = [k for k in globals().keys() if not k.startswith('__')]
print(g)
for f in ['load','base','save','obs','acts','episodes','stats']:
    try:
        print(f, globals()[f].__doc__)
    except Exception as e:
        print(f, e)
print(stats('action.action_out_proj.kernel'))
