Run `python scripts/build_catalogue.py --workers 4` from the project root.
Requires Python 3.12+ and ffmpeg/ffprobe on PATH; no Python packages required.

The script reads the two CSV header rows as (section, field) pairs and the
original first column as track_id, skipping the index-name row. It selects
only set.subset == small. Mapping: title=track.title, artist_id=artist.id,
artist_name=artist.name, album_id=album.id, album_title=album.title,
genre_top=track.genre_top, genres=track.genres, subset=set.subset,
split=set.split. Values are preserved verbatim. Audio properties come from
ffprobe, which also reads embedded tags. Tag comparisons only collapse
whitespace and casefold. No artist identity verification is performed.

SHA-1 manifests are checked for selected audio and supplied metadata files;
SHA-256 groups identify byte-identical audio. FFmpeg decodes each file with
strict error detection. Publication depends only on Stage 1 completeness:
8,000 unique integer IDs, no missing/empty audio files, and no missing required
metadata. Failed completeness preserves an existing catalogue and exits 1.
Checksum mismatches and audio usability failures remain visible in the audit
and report, independently of catalogue publication. Review items
are never corrected automatically. Outputs are sorted by numeric track ID.

Split validation uses training, validation, and test, corresponding to FMA's
train/validation/test split documented at https://github.com/mdeff/fma.
The original loader and path convention are documented in
https://github.com/mdeff/fma/blob/master/utils.py.

The catalogue includes probe_ok, strict_decode_ok, audio_issue, title_tag_status,
and artist_tag_status. Unmeasurable audio properties use the literal
"unavailable"; optional missing metadata remains empty. A probe duration is
not a guarantee that a complete waveform can be loaded. The written catalogue
is read back to verify its row count, unique IDs, ordering, and audio paths.

The intended Stage 2 loader is not yet selected in the project dependencies,
source, or development plan. Its complete-load, empty-output, finite-sample,
and valid-duration checks are explicitly pending, with no claimed pass count.
The JSON report retains separate probing and strict-decoding evidence for the
six investigated tracks. Raw audio is never changed.

Refresh optional Stage 1 metadata without decoding or hashing audio again:
`python scripts/build_catalogue.py --metadata-only`

Verified mappings: language_code = (track, language_code),
album_date_released = (album, date_released), license = (track, license).
There were no equivalent pre-existing catalogue columns. Strings are copied
verbatim using a one-to-one join on the original integer track_id index.
license_missing is True for
empty or whitespace-only source licenses; the license string stays intact.
A present license does not imply permission for every use.

The metadata-only mode checks source/catalogue ID uniqueness, audit ID coverage,
all existing unrelated values, source correspondence, and nonempty audio paths.
It writes a temporary CSV and reads and validates it before atomic replacement.
Existing audio audit and checksum results are reused. The validation JSON gains
metadata_update counts and updated column mappings. A full
build also includes these fields and runs the metadata validation step.

Column provenance and initial feature policy:

| Category | Columns / source |
| --- | --- |
| Dataset metadata | track_id (original index); title, artist_id, artist_name, album_id, album_title, genre_top, genres, subset, split, language_code, album_date_released, license (two-row header mappings above) |
| Technical properties | file_size_bytes from filesystem; duration_seconds, sample_rate_hz, channels from ffprobe audio inspection |
| File locator | audio_path constructed from the dataset track ID; never a source for inferred metadata |
| Validation flags | probe_ok, strict_decode_ok, audio_issue, license_missing |
| Audit-only comparisons | title_tag_status, artist_tag_status; retained in the catalogue for compatibility, never metadata replacements or similarity features |

Language and album release date are optional, verbatim dataset strings. Missing
values remain empty. release_year and its derivation have been removed. No
metadata is inferred using LLMs, filenames, embedded tags, or external searches.
Dataset titles and artists are never overwritten by MP3 tags. Full embedded
values and comparisons remain in the existing audio audit. Optional omissions
do not prevent catalogue completeness or imply metadata has been recovered.

Initial numerical recommendation features will use MFCC statistics, with genre
evaluated separately. Language, dates, titles, artist IDs, and license information
are not initial numerical similarity features. This task does not extract MFCCs.
Metadata-only refresh validates every retained value (including optional fields
and audio health flags) against the existing catalogue, checks dataset metadata
against its source by track ID, and checks required fields before replacement.
