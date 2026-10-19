# Keeping the first run and the improvement loop going

`scripts/loop_watchdog.py` checks the portal's Training page every hour and once at sign-in, and does at most one
thing each time:

| It sees | It does |
|---|---|
| `first-run` ended by a crash or reboot | resumes it (at most 3 times in 24 hours) |
| `first-run` finished, held-out sets built, no `loop` run yet | starts `scripts/loop.py` as the Training page run `loop` |
| `loop` ended by a crash or reboot, or failed | resumes it (at most 3 times in 24 hours) |
| `loop` waiting for data, disk or scripts | resumes it when `experiments/loop.yaml` changes, staged scripts are installed, or 6 hours pass |
| anything paused or stopped from the Training page | nothing: a person's Pause or Stop always wins |
| another run holding the GPU | nothing |

It also installs the loop's newer `prepare_dataset.py` and `train_longrun.py` from `staged/` into `scripts/`, each only
after the first run has finished the last stage that runs it (`generate` and `train`). The replaced file is kept in
`staged/replaced/`. Both changes only add options, so the first run's own commands behave the same either way; the
loop refuses to run on the old versions.

Log: `results\watchdog\watchdog.log`. Restart counts: `results\watchdog\state.json`. Loop state:
`python scripts\loop_status.py`.

## Install (no admin rights)

Run as my own Windows user, in `C:\AI\training`:

```bat
schtasks /Create /TN "Anneal loop watchdog" /SC HOURLY /F ^
  /TR "\"C:\AI\training\.venv\Scripts\pythonw.exe\" \"C:\AI\training\scripts\loop_watchdog.py\""
```

and a Startup-folder entry, so it also runs a few minutes after sign-in (the portal starts at sign-in too):

```bat
> "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Anneal loop watchdog.cmd" (
  echo @echo off
  echo start "" "C:\AI\training\.venv\Scripts\pythonw.exe" "C:\AI\training\scripts\loop_watchdog.py" --delay 180
)
```

If Windows refuses to create the task without admin rights, the Startup entry runs the watchdog with
`--every 3600` instead, and it then keeps checking hourly for as long as I am signed in.

The task runs only while I am signed in, like the portal. After a reboot nothing runs until someone signs in,
unless Windows signs in automatically.

Check without changing anything: `.venv\Scripts\python scripts\loop_watchdog.py --dry-run`.

## Turn it off

```bat
schtasks /Change /TN "Anneal loop watchdog" /DISABLE
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Anneal loop watchdog.cmd"
```

Pausing a run from the Training page is enough to stop the watchdog touching that run.
