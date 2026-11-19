# Weightsmith experiment monitor

This watcher reads the frozen November 19 correction-scaling experiment. It
never changes its code, sources, candidates, plans, or jobs. Snapshots, status,
deduplication state, and notification-attempt logs are written only to
`projects/rss-repair-scaling-monitor-20261119/`, outside the frozen inputs.

It checks every 60 seconds for exact completed batch receipts and the original
schedule's three complete groups: 320 original-request episodes, 144 unseen-
wording episodes, and 144 competing-goal episodes. Group notifications require
all conditions and their exact slots. Final completion requires the verified
608-episode test report and its 128 development-selection episodes. Small
frozen metadata files are hash-checked; large model and observation arrays are
not repeatedly rehashed during active experiments.

Other events are an explicit stopped/failed experiment, failed receipt audit,
30 minutes without a new completed batch, and a 30-minute progress heartbeat.
A stall is not reported as a confirmed failure. Events are persistently marked
before queue attempts, so an uncertain delivery is not retried every poll.
The watcher stops at its original eight-hour deadline and records one timeout
event if testing is unfinished. Restarting the watcher does not extend that
deadline. It also stops after a final report or an explicit/audited failure.

Notifications are disabled by default. `--once` always writes one non-notifying
snapshot, even if notification arguments are supplied:

```bash
python3 study/iros2027/v2/rss_repair_scaling_monitor_20261119/watch.py --once
python3 -m unittest discover -s study/iros2027/v2/rss_repair_scaling_monitor_20261119 -p 'test_*.py' -v
```

For the user's explicitly authorized updates to this existing conversation:

```bash
python3 study/iros2027/v2/rss_repair_scaling_monitor_20261119/watch.py \
  --notify-thread 01a1dc82-5c12-7623-8075-edadc14debb9 \
  --codex-bin /home/ali/.local/bin/codex
```

The queue command uses an argument list and a 60-second timeout. A zero exit
with a warning is recorded as uncertain delivery. Even an unqualified command
success is not described as proof that the user saw the update. Each queued
event explicitly identifies itself as an automated event for the user's
existing request, respects later cancellation, and asks the assistant to
freshly verify receipts before reporting. It does not authorize restarting
jobs, tuning candidates, editing the paper, or resuming ICWM.

Neither the test suite nor `--once` sends messages or starts a service. The
operator owns the durable service launch and cancellation.

## Durable service

The monitor was launched as the user service
`weightsmith-scaling-monitor-20261119.service` with this command:

```bash
systemd-run --user \
  --unit=weightsmith-scaling-monitor-20261119 \
  --property=WorkingDirectory=/home/ali/weightsmith-iros2027 \
  /usr/bin/python3 \
  /home/ali/weightsmith-iros2027/study/iros2027/v2/rss_repair_scaling_monitor_20261119/watch.py \
  --poll-seconds 60 --max-seconds 28800 \
  --notify-thread 01a1dc82-5c12-7623-8075-edadc14debb9 \
  --codex-bin /home/ali/.local/bin/codex
```

Inspect its service state, recent logs, and verified progress:

```bash
systemctl --user status weightsmith-scaling-monitor-20261119.service --no-pager
journalctl --user -u weightsmith-scaling-monitor-20261119.service -n 40 --no-pager
cat /home/ali/weightsmith-iros2027/projects/rss-repair-scaling-monitor-20261119/status.json
```

Stop monitoring and future monitor queue attempts:

```bash
systemctl --user stop weightsmith-scaling-monitor-20261119.service
```

Stopping this service **does not stop the experiment** or cancel messages
already queued. Experiment workers remain under their separate supervisor.
