import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


path = Path(
    "/home/himanshus/Projects/music-recommender/"
    "data/processed/audio_metadata.csv"
)

query_track = "data/raw/fma_small/000/000002.mp3"

MFCC_FIELDS = (
    [f"mfcc_mean_{i + 1}" for i in range(20)]
    + [f"mfcc_std_{i + 1}" for i in range(20)]
)


def main():
    # Load the saved metadata and MFCC features.
    load_file = pd.read_csv(path)

    required_fields = ["relative_path"] + MFCC_FIELDS
    missing_fields = [
        field for field in required_fields
        if field not in load_file.columns
    ]

    if missing_fields:
        raise ValueError(f"Missing CSV columns: {missing_fields}")

    if len(load_file) < 2:
        raise ValueError("At least two tracks are required.")

    if load_file["relative_path"].isna().any():
        raise ValueError("CSV contains missing track paths.")

    if load_file["relative_path"].duplicated().any():
        raise ValueError("CSV contains duplicate track paths.")

    # Extract only the 40 MFCC features.
    mfcc_data = load_file[MFCC_FIELDS].to_numpy(dtype=np.float64)

    if not np.isfinite(mfcc_data).all():
        raise ValueError("MFCC features contain missing or nonfinite values.")

    # Standardize each feature across the catalogue.
    scaler = StandardScaler()
    scaled_mfcc = scaler.fit_transform(mfcc_data)

    # Find the selected track's row.
    matches = np.flatnonzero(
        load_file["relative_path"].to_numpy() == query_track
    )

    if len(matches) != 1:
        raise ValueError(f"Query track not found: {query_track}")

    query_index = int(matches[0])

    # Cosine similarity requires nonzero vectors.
    valid_vectors = np.linalg.norm(scaled_mfcc, axis=1) > 0

    if not valid_vectors[query_index]:
        raise ValueError(
            "Query track has a zero standardized feature vector."
        )

    # Compare one selected track against all tracks.
    similarity_scores = cosine_similarity(
        scaled_mfcc[query_index:query_index + 1],
        scaled_mfcc,
    )[0]

    # Exclude the selected track and zero vectors.
    valid_vectors[query_index] = False

    results = load_file.loc[
        valid_vectors, ["relative_path"]
    ].copy()

    results["cosine_similarity"] = similarity_scores[valid_vectors]

    # Rank from most similar to least similar.
    results = results.sort_values(
        "cosine_similarity",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    results.insert(0, "query_track", query_track)

    # Save results beside the input CSV.
    output_path = path.parent / "cosine_similarity_results.csv"
    results.to_csv(output_path, index=False)

    print(f"\nQuery track: {query_track}")
    print("\nTop 5 similar tracks:")
    print(results.head(5).to_string(index=False))
    print(f"\nSaved {len(results)} results to: {output_path}")


if __name__ == "__main__":
    main()
