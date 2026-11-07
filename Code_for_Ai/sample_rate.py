import soundfile as sf
import numpy as np

path = "/home/himanshus/Projects/music-recommender/data/raw/fma_small/000/000002.mp3"

try:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    samples, channels = audio.shape
    invalid_count = int((~np.isfinite(audio)).sum())

    print("Sample rate:", sr)
    print("Channels:", channels)
    print("Samples per channel:", samples)
    print("Loaded duration:", samples / sr)
    print("Nonfinite samples:", invalid_count)

    if samples == 0:
        print("ERROR: Empty audio")
    elif invalid_count:
        print("ERROR: Audio contains invalid values")
    else:
        print("Minimum amplitude:", audio.min())
        print("Maximum amplitude:", audio.max())
        print("Peak amplitude:", np.abs(audio).max())
        print("RMS amplitude:", np.sqrt(np.mean(audio.astype("float64") ** 2)))

except Exception as error:
    print("Loading failed:", error)
