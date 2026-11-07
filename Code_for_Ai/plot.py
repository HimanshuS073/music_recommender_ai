import argparse
import csv
import os
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "raw" / "fma_small"
OUTPUT = ROOT / "data" / "processed"

MFCC_FIELDS = (
    [f"mfcc_mean_{i + 1}" for i in range(20)]
    + [f"mfcc_std_{i + 1}" for i in range(20)]
)

FIELDS = [
    "relative_path",
    "file_size_bytes",
    "modification_time_ns",
    "sample_rate",
    "channels",
    "samples_per_channel",
    "duration_seconds",
    "nonfinite_sample_count",
    "peak_amplitude",
    "rms_amplitude",
] + MFCC_FIELDS

FAILURE_FIELDS = [
    "relative_path",
    "file_size_bytes",
    "modification_time_ns",
    "error_message",
    "attempts",
]


def load_records(filename):
    if not filename.exists():
        return {}

    with filename.open(newline="", encoding="utf-8") as file:
        records = {}

        for row in csv.DictReader(file):
            if (
                not row.get("relative_path")
                or any(value is None for value in row.values())
            ):
                raise ValueError(f"Incomplete CSV row in {filename}")

            records[row["relative_path"]] = row

        return records


def save_records(filename, fields, records):
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=filename.parent,
            delete=False,
        ) as file:
            temporary = Path(file.name)

            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records.values())

            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, filename)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def discover_mp3(folder, folders_seen=None):
    if folders_seen is None:
        folders_seen = set()

    folder = Path(folder).resolve()

    if folder in folders_seen:
        return

    folders_seen.add(folder)

    for item in sorted(folder.iterdir()):
        if item.is_dir():
            yield from discover_mp3(item, folders_seen)

        elif item.is_file() and item.suffix.lower() == ".mp3":
            yield item


def has_valid_mfcc(record):
    try:
        values = np.array(
            [float(record[field]) for field in MFCC_FIELDS],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return False

    return bool(np.isfinite(values).all())


def check_every_file(path, *, output=OUTPUT, show_waveforms=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    success_file = output / "audio_metadata.csv"
    failure_file = output / "audio_failures.csv"

    successes = load_records(success_file)
    failures = load_records(failure_file)

    counts = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    try:
        for audio_path in discover_mp3(path):
            relative_path = audio_path.relative_to(ROOT).as_posix()
            size = modified = ""

            try:
                stat = audio_path.stat()
                size = str(stat.st_size)
                modified = str(stat.st_mtime_ns)

                existing = successes.get(relative_path, {})

                unchanged = (
                    existing.get("file_size_bytes") == size
                    and existing.get("modification_time_ns") == modified
                )

                if (
                    unchanged
                    and has_valid_mfcc(existing)
                    and relative_path not in failures
                ):
                    counts["skipped"] += 1

                    if sum(counts.values()) % 200 == 0:
                        print(counts, flush=True)

                    continue

                record = get_audio_data(audio_path)

                # Ensure the file did not change during extraction.
                after = audio_path.stat()

                if (
                    str(after.st_size),
                    str(after.st_mtime_ns),
                ) != (size, modified):
                    raise ValueError(
                        "File changed while being read; retry required."
                    )

            except (OSError, RuntimeError, ValueError) as error:
                failures[relative_path] = {
                    "relative_path": relative_path,
                    "file_size_bytes": size,
                    "modification_time_ns": modified,
                    "error_message": f"{type(error).__name__}: {error}",
                    "attempts": int(
                        failures.get(relative_path, {}).get("attempts", 0)
                    ) + 1,
                }

                save_records(failure_file, FAILURE_FIELDS, failures)

                if relative_path in successes:
                    del successes[relative_path]
                    save_records(success_file, FIELDS, successes)

                counts["failed"] += 1

                print(
                    f"Failed to process {relative_path}: {error}",
                    flush=True,
                )

            else:
                successes[relative_path] = {
                    "relative_path": relative_path,
                    "file_size_bytes": size,
                    "modification_time_ns": modified,
                    **record,
                }

                # Save immediately so completed work survives interruption.
                save_records(success_file, FIELDS, successes)

                if relative_path in failures:
                    del failures[relative_path]
                    save_records(failure_file, FAILURE_FIELDS, failures)

                counts["processed"] += 1

                if show_waveforms:
                    display_waveform(audio_path)

            if sum(counts.values()) % 200 == 0:
                print(counts, flush=True)

    finally:
        if not success_file.exists():
            save_records(success_file, FIELDS, successes)

        if not failure_file.exists():
            save_records(failure_file, FAILURE_FIELDS, failures)

        print("Scan counts:", counts, flush=True)

    return counts


def get_audio_data(audio_path):
    # Shape: (samples per channel, number of channels).
    audio, sr = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    samples, channels = audio.shape
    invalid_count = int((~np.isfinite(audio)).sum())

    # Validate before extracting features.
    if samples == 0:
        raise ValueError("Loaded audio is empty.")

    if invalid_count:
        raise ValueError(
            f"Audio contains {invalid_count} nonfinite samples."
        )

    if sr <= 0 or channels <= 0:
        raise ValueError("Invalid sample rate or channel count.")

    duration = samples / sr

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid loaded duration.")

    # Convert (samples, channels) into a mono signal.
    mono_audio = np.mean(audio, axis=1)

    # Use the same sample rate for MFCC extraction on every track.
    feature_sr = 22050

    if sr != feature_sr:
        mono_audio = librosa.resample(
            mono_audio,
            orig_sr=sr,
            target_sr=feature_sr,
        )

    # Shape: (20 coefficients, number of time frames).
    mfcc = librosa.feature.mfcc(
        y=mono_audio,
        sr=feature_sr,
        n_mfcc=20,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    # Summarize each coefficient across time.
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # One vector containing 40 features per track.
    features = np.concatenate([mfcc_mean, mfcc_std])

    if not np.isfinite(features).all():
        raise ValueError("MFCC features contain nonfinite values.")

    record = {
        "sample_rate": int(sr),
        "channels": int(channels),
        "samples_per_channel": int(samples),
        "duration_seconds": float(duration),
        "nonfinite_sample_count": invalid_count,
        "peak_amplitude": float(np.abs(audio).max()),
        "rms_amplitude": float(
            np.sqrt(np.mean(audio.astype("float64") ** 2))
        ),
    }

    record.update({
        name: float(value)
        for name, value in zip(MFCC_FIELDS, features)
    })

    return record


def display_waveform(audio_path):
    import matplotlib.pyplot as plt

    audio, sr = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    samples, channels = audio.shape

    # Convert sample positions into seconds.
    time_axis = np.arange(samples) / sr

    fig, axes = plt.subplots(
        channels,
        1,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    try:
        for channel in range(channels):
            ax = axes[channel, 0]

            ax.plot(
                time_axis,
                audio[:, channel],
                linewidth=0.3,
            )
            ax.set_ylabel(f"Channel {channel + 1}\nAmplitude")
            ax.grid(alpha=0.3)

        axes[-1, 0].set_xlabel("Time (seconds)")
        fig.suptitle(f"Track {audio_path.stem} — Audio waveform")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(100)

    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio metadata and MFCC features into CSV."
    )

    parser.add_argument(
        "--show-waveforms",
        action="store_true",
        help="Display each newly processed waveform",
    )

    args = parser.parse_args()

    check_every_file(
        path,
        show_waveforms=args.show_waveforms,
    )
