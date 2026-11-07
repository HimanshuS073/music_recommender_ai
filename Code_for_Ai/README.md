Run `uv run python Code_for_Ai/plot.py` from the repository root.
Add `--show-waveforms` for interactive plots; batch mode is the default.

Results are CSV files under data/processed/:
- audio_metadata.csv: one successful record per repository-relative MP3 path,
  containing file size, nanosecond modification time, and loaded audio statistics.
- audio_failures.csv: latest failure message, size/time and attempt count per path.

Each result is saved immediately using a temporary CSV and atomic replacement.
Restarting skips successes with unchanged size and modification time, and retries
failures. Changed files are processed again. Successful retries clear failures;
failed retries remove stale success rows. The old SQLite file is a retained
backup only; the scanner does not read or update it.

Discovery is recursive. SoundFile reads each complete waveform; validation rejects
empty audio, nonfinite samples, invalid sample rates/channels and invalid duration.
Library diagnostics do not necessarily cause load failure. These statistics are
for validation and analysis, not sufficient recommendation features. No embeddings
or MFCCs are extracted. Raw audio is read only. Removed files are not automatically
purged from saved CSVs. Run only one scanner at a time against the same outputs.
