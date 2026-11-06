print(len(READABLE_NAMES))
print([n for n in READABLE_NAMES if not n.startswith('vision') and 'blocks' not in n])
print(json.dumps(CONFIG)[:2000] if CONFIG is not None else None)
print(str(REFERENCES)[:1000])
