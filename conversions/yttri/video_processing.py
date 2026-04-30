"""
Create NWB files linking timestamped videos via symlinks for DANDI/EMBER upload.

Expected input structure:
    VIDEO_ROOT/
        MMDDYY/                          ← e.g. 010224  (Jan 02 2024)
            YYYY-MM-DD_HH-MM-SS.mp4      ← e.g. 2024-01-02_09-00-00.mp4  (4 or 8 videos)
            ...

Session splitting:
    - Folders with 4 videos → one session  (ses-MMDDYY)
    - Folders with 8 videos → split at the largest timestamp gap into two sessions
      named alphabetically  (ses-MMDDYYa, ses-MMDDYYb)
    - Any other count → warning, folder skipped

Output structure (DANDI-compliant):
    NWB_OUTPUT/
        sub-<SUBJECT_ID>/
            sub-<SUBJECT_ID>_ses-MMDDYY.nwb
            sub-<SUBJECT_ID>_ses-MMDDYYa.nwb
            sub-<SUBJECT_ID>_ses-MMDDYYb.nwb
            YYYY-MM-DD_HH-MM-SS.mp4  → symlink into VIDEO_ROOT
            ...

Requirements:
    pip install pynwb dandi

Usage:
    Edit the CONFIG block below, then:
        python create_nwb_video_sessions.py
"""

from datetime import datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.image import ImageSeries


# ── CONFIG ────────────────────────────────────────────────────────────────────

VIDEO_ROOT   = Path(r"X:\hsu\EY4152_Pl6xAi32\videos")       # parent directory containing MMDDYY folders
NWB_OUTPUT   = Path(r"C:\EmberDandi\Data")       # where NWB files and symlinks will be written

SUBJECT_ID   = "EY4152"
SPECIES      = "Mus musculus"             # NCBI taxonomy name
SEX          = "U"                        # M / F / U / O
AGE          = "P90D/"                     # ISO 8601 duration

EXPERIMENTER = ["Alexander Hsu"]
LAB          = "Yttri Lab"
INSTITUTION  = "Carnegie Mellon University"
VIDEO_RATE   = 60.0                       # frames per second

VIDEOS_PER_SESSION = 4                    # expected number of videos per session

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_folder_date(folder_name: str) -> datetime:
    """Parse a MMDDYY folder name into a timezone-aware datetime (midnight)."""
    try:
        return datetime.strptime(folder_name, "%m%d%y").replace(tzinfo=tzlocal())
    except ValueError:
        raise ValueError(
            f"Folder '{folder_name}' does not match expected MMDDYY format (e.g. 010224)."
        )


def parse_video_timestamp(video_path: Path) -> datetime:
    """Parse a YYYY-MM-DD_HH-MM-SS stem into a datetime."""
    try:
        return datetime.strptime(video_path.stem, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        raise ValueError(
            f"Video '{video_path.name}' does not match expected YYYY-MM-DD_HH-MM-SS format."
        )


def split_by_largest_gap(videos: list[Path]) -> tuple[list[Path], list[Path]]:
    """
    Split a sorted list of videos into two groups by cutting at the largest
    time gap between consecutive timestamps.
    """
    timestamps = [parse_video_timestamp(v) for v in videos]
    gaps = [b - a for a, b in pairwise(timestamps)]
    split_after = gaps.index(max(gaps))   # index of the video just before the gap
    return videos[:split_after + 1], videos[split_after + 1:]


def discover_sessions(video_root: Path) -> list[dict]:
    """
    Return a list of session dicts from all valid MMDDYY subdirectories.

    - Each folder becomes ONE session
    - Only .mp4 files are considered
    - Files that don't match YYYY-MM-DD_HH-MM-SS are ignored
    """
    sessions = []

    for folder in sorted(video_root.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.isdigit() or len(folder.name) != 6:
            print(f"  Skipping non-date folder: {folder.name}")
            continue

        valid_videos = []

        for f in folder.iterdir():
            if not (f.is_file() and f.suffix.lower() == ".mp4"):
                continue
            try:
                parse_video_timestamp(f)  # validate format
                valid_videos.append(f)
            except ValueError:
                print(f"    Skipping malformed filename: {f.name}")

        videos = sorted(valid_videos, key=parse_video_timestamp)

        if len(videos) == 0:
            print(f"  Warning: '{folder.name}' has no valid .mp4 files. Skipping.")
            continue

        sessions.append({
            "session_id": folder.name,
            "start_time": parse_video_timestamp(videos[0]).replace(tzinfo=tzlocal()),
            "video_files": videos,
        })

    return sessions


# ── NWB creation ──────────────────────────────────────────────────────────────

def make_nwb(session: dict, output_dir: Path) -> Path:
    """
    Create one NWB file for a session.
    Symlinks each video into the subject folder so external_file paths are
    simple filenames with no directory traversal.
    """
    subject_dir = output_dir / f"sub-{SUBJECT_ID}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Symlink videos into the subject folder
    symlinked = []
    for video_path in session["video_files"]:
        link = subject_dir / video_path.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(video_path)
        symlinked.append(link)

    # Build NWB file
    nwbfile = NWBFile(
        session_description=f"Recording session {session['session_id']}",
        identifier=str(uuid4()),
        session_start_time=session["start_time"],
        session_id=session["session_id"],
        experimenter=EXPERIMENTER,
        lab=LAB,
        institution=INSTITUTION,
    )

    nwbfile.subject = Subject(
        subject_id=SUBJECT_ID,
        species=SPECIES,
        sex=SEX,
        age=AGE,
    )

    for i, link in enumerate(symlinked):
        series = ImageSeries(
            name=f"video_{i:02d}",
            description=f"Video {i} — {link.name}",
            external_file=[link.name],    # same directory as NWB file
            format="external",
            starting_frame=[0],
            rate=VIDEO_RATE,
            unit="n/a",
        )
        nwbfile.add_acquisition(series)

    nwb_path = subject_dir / f"sub-{SUBJECT_ID}_ses-{session['session_id']}.nwb"
    with NWBHDF5IO(str(nwb_path), mode="w") as io:
        io.write(nwbfile)

    return nwb_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Scanning: {VIDEO_ROOT}\n")
    sessions = discover_sessions(VIDEO_ROOT)

    if not sessions:
        print("No valid sessions found. Check VIDEO_ROOT and folder naming (MMDDYY).")
        return

    print(f"Found {len(sessions)} session(s). Writing to '{NWB_OUTPUT}/'...\n")
    for session in sessions:
        nwb_path = make_nwb(session, NWB_OUTPUT)
        print(f"  [{session['session_id']}]  {nwb_path.name}")
        for v in session["video_files"]:
            print(f"           ↳ {v.name}  (symlinked)")
        print()

    print("Done. Next steps:")
    print(f"  dandi validate {NWB_OUTPUT}/")
    print(f"  dandi organize {NWB_OUTPUT}/ --dandiset-path ./my_dandiset/")
    print(f"  dandi upload ./my_dandiset/")


if __name__ == "__main__":
    main()
