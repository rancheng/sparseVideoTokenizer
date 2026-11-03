# videoTokenizer

Sparse spatio-temporal video tokenizer (H-Net dynamic chunking over space and time, learned fovea, event-token memory)
for video-native robot policies. Owner works in Chinese; docs are in English.

- Start here when continuing work: `docs/remote-continue.md` (machine state, setup, datasets, what is done, what is next).
- Results and their caveats: `docs/experiment-log.md`, `docs/v2-spatiotemporal.md`, `docs/lsa-experiments.md`.
- Use `.venv/bin/python`. One GPU job at a time (24 GB). Stop jobs by PID, never `pkill -f` with the pattern in the
  same command.
- Report results honestly with sample sizes; check claims across seeds before stating them (see the retractions in
  `docs/lsa-experiments.md`).
- Commit messages end with a Co-Authored-By line for Claude when Claude wrote the change.
