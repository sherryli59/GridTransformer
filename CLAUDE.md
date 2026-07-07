# GridTransformer — working agreements

## Plots
- **ALWAYS show the full path to any plot/image you generate**, as a clickable link, immediately after creating it. Never describe or display a plot without also printing its path.

## Results durability
- **SAVE ALL RESULTS to the repo, immediately**: raw run logs and one-off harness scripts go to `reports/logs-<date>/` and get committed the moment a run finishes — NEVER leave results only in the session scratchpad (/tmp, wiped) or only in a summary. Benchmark runs must SAVE final configurations/artifacts (`liquid_coupling_flow/artifacts/`), not just print metrics.

## Long-running jobs (training/eval on GPU)
- **NEVER send a long job's stderr to /dev/null** — a silent crash costs hours of assumed progress. Redirect both streams to the log file (`> log.out 2>&1`).
- **Don't check job liveness with pgrep on a pattern that can match your own monitoring/watcher command** (self-satisfying check). The reliable liveness signal is the output file's mtime/content advancing; for identifying GPU jobs use `nvidia-smi --query-compute-apps=pid` + `ps -p <pid>`.
- **CHECKPOINT INCREMENTALLY: a multi-stage run must persist each completed unit (seed/epoch/arm) to disk THE MOMENT it finishes** — never design a run whose only save is at the very end, holding hours of completed work hostage to later stages (PT2 seed-0: 6h of configs unrecoverable in-process because the ladder saved only after both seeds). Before launching any run > ~30 min, check where it saves and add per-unit partial saves if missing.
