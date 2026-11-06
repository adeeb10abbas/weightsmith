print(len(NAMES))
print([n for n in NAMES if 'blocks' not in n])
print([n for n in NAMES if n.startswith('action.blocks.00') or n.startswith('action.blocks_00') or '00' in n][:60])
