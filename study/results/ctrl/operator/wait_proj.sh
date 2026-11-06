#!/bin/bash
# Block until no `owsp.py eval|acts` process of project $1 is running (at most ~9 minutes), then report.
R="$1"
for i in $(seq 1 108); do
  busy=""
  for p in $(pgrep -f 'weightsmith/owsp.py (eval|acts)'); do
    if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "WSP_ROOT=$R"; then
      busy=$(tr '\0' ' ' < /proc/$p/cmdline | sed 's/.*owsp.py //')
    fi
  done
  if [ -z "$busy" ]; then echo "idle: no eval or capture is running for this project"; exit 0; fi
  if [ "$i" = 1 ]; then echo "running: $busy (queued jobs wait for the shared GPU)"; fi
  sleep 5
done
echo "still running after 9 minutes; call ./owsp wait again"
