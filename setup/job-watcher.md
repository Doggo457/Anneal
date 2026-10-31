# Running long jobs without anyone watching

`scripts/job_watcher.py` runs the long jobs (strength scoring, tuning runs, checks) one at a time from a queue, and
starts a job again from its saved work if it stops. It runs on its own, so a job keeps going when a remote session
goes idle or closes.

| It sees | It does |
|---|---|
| a job running | checks the RAM hard line every minute: free RAM under 1.5 GiB or committed memory over 74 GB (decimal) writes `STOP` in the job's folders (the job saves and exits); it never kills anything |
| the job exited, a `STOP` file is there | moves it to `held\`; a person decides (a stopped job can still write its done files, so `STOP` is checked first) |
| the job exited, its done files exist | moves it to `done\` and starts the next one |
| the job exited within 2 minutes without finishing | tries again 10 minutes later, 12 times at most |
| the job stopped later without finishing | starts it again (the jobs skip work already saved), up to `max_attempts`, then moves it to `failed\` |
| no job running | starts the first queued job once the GPU has been free for 3 checks in a row: no `llama-server.exe` and no other training or scoring script running |

Queue: `results\jobs\queue\*.json`, run in file-name order:

```json
{"name": "x0.25", "args": ["scripts\\lora_scale_check.py", "--round", "2", "--scale", "0.25", "--n", "60"],
 "done": ["results\\research\\lora_scale\\r002_x0.25\\DONE"], "stop_dirs": ["results\\research\\lora_scale\\r002_x0.25"],
 "max_attempts": 4}
```

Status: `.venv\Scripts\python scripts\job_watcher.py --status`. Log: `results\jobs\watcher.log`. Job output:
`results\jobs\logs\NAME.log`. One check that changes nothing: `--once --dry-run`.

## Install (no admin rights)

In a normal PowerShell window, as my own Windows user:

```powershell
$a = New-ScheduledTaskAction -Execute 'C:\AI\training\.venv\Scripts\pythonw.exe' -Argument 'scripts\job_watcher.py' -WorkingDirectory 'C:\AI\training'
$t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Anneal job watcher' -Action $a -Trigger $t -Settings $s -Force
Start-ScheduledTask -TaskName 'Anneal job watcher'
```

The task starts the watcher now and tries again every hour; Windows skips the start while the watcher is still
running, and the watcher itself exits if another copy holds `results\jobs\watcher.lock`. It has no time limit, so a
job that runs for days is never cut off. Like the loop watchdog, it runs only while I am signed in.

## Turn it off

```powershell
Disable-ScheduledTask -TaskName 'Anneal job watcher'
Stop-ScheduledTask -TaskName 'Anneal job watcher'
```

To stop a running job cleanly, put a `STOP` file in its folder.
