"""
Create NWB files linking timestamped videos via symlinks for DANDI/EMBER upload,
following the DANDI multi-camera layout used by IBL Dandiset 000409:

    sub-<SUB>/
        sub-<SUB>_ses-<SES>_image/
            sub-<SUB>_ses-<SES>_VideoCam00.mp4   (symlinks)
            sub-<SUB>_ses-<SES>_VideoCam01.mp4
            ...
        sub-<SUB>_ses-<SES>_desc-raw_image.nwb

Expected input structure:
    VIDEO_ROOT/
        MMDDYY/                              e.g. 010224  (Jan 02 2024)
            YYYY-MM-DD_HH-MM-SS.mp4          4 or 8 videos
            ...

Session splitting:
    - 4 videos  -> one session     (ses-MMDDYY)
    - 8 videos  -> split at the largest timestamp gap into two sessions
                   named alphabetically (ses-MMDDYYa, ses-MMDDYYb)
    - any other count -> warning, folder skipped

Camera labels:
    Each session is expected to have exactly len(CAMERA_LABELS) videos.
    Videos are sorted by timestamp and assigned to camera labels in order.
    Override CAMERA_LABELS below to match your rig (e.g. ["BodyCamera",
    "LeftCamera", "RightCamera", "TopCamera"]).

Requirements:
    pip install pynwb dandi
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
VIDEO_ROOT = Path(r"X:\hsu\EY4152_Pl6xAi32\videos")
NWB_OUTPUT = Path(r"C:\EmberDandi\Data")

SUBJECT_ID   = "EY4152"
SPECIES      = "Mus musculus"
SEX          = "U"
AGE          = "P90D/"

EXPERIMENTER = ["Alexander Hsu"]
LAB          = "Yttri Lab"
INSTITUTION  = "Carnegie Mellon University"
VIDEO_RATE   = 60.0  # frames per second

# Per-session expectations.  CAMERA_LABELS drives both the expected video
# count and the DANDI-compliant filename suffix (Video<Label>.mp4).
# Edit to reflect your actual camera rig.
CAMERA_LABELS = ["Cam00", "Cam01", "Cam02", "Cam03"]
VIDEOS_PER_SESSION = len(CAMERA_LABELS)          # 4
DOUBLE_SESSION_COUNT = 2 * VIDEOS_PER_SESSION    # 8 -> split into a/b


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_folder_date(folder_name: str) -> datetime:
    """Parse a MMDDYY folder name into a tz-aware datetime (midnight)."""
    return datetime.strptime(folder_name, "%m%d%y").replace(tzinfo=tzlocal())


def parse_video_timestamp(video_path: Path) -> datetime:
    """Parse a YYYY-MM-DD_HH-MM-SS stem into a datetime."""
    return datetime.strptime(video_path.stem, "%Y-%m-%d_%H-%M-%S")


def split_by_largest_gap(videos: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split a sorted list of videos at the largest inter-frame time gap."""
    timestamps = [parse_video_timestamp(v) for v in videos]
    gaps = [b - a for a, b in pairwise(timestamps)]
    split_after = gaps.index(max(gaps))
    return videos[: split_after + 1], videos[split_after + 1 :]


def _collect_valid_videos(folder: Path) -> list[Path]:
    valid = []
    for f in folder.iterdir():
        if not (f.is_file() and f.suffix.lower() == ".mp4"):
            continue
        try:
            parse_video_timestamp(f)
            valid.append(f)
        except ValueError:
            print(f"    Skipping malformed filename: {f.name}")
    return sorted(valid, key=parse_video_timestamp)


def discover_sessions(video_root: Path) -> list[dict]:
    """
    Walk VIDEO_ROOT and return a list of session dicts:
        { "session_id": "010224"  | "010224a" | "010224b",
          "start_time": datetime,
          "video_files": [Path, ...] }
    """
    sessions = []

    for folder in sorted(video_root.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder.name.isdigit() and len(folder.name) == 6):
            print(f"  Skipping non-date folder: {folder.name}")
            continue

        videos = _collect_valid_videos(folder)
        n = len(videos)

        if n == VIDEOS_PER_SESSION:
            groups = [(folder.name, videos)]
        elif n == DOUBLE_SESSION_COUNT:
            first, second = split_by_largest_gap(videos)
            groups = [(folder.name + "a", first), (folder.name + "b", second)]
        else:
            print(
                f"  Warning: '{folder.name}' has {n} videos "
                f"(expected {VIDEOS_PER_SESSION} or {DOUBLE_SESSION_COUNT}). Skipping."
            )
            continue

        for ses_id, vids in groups:
            if len(vids) != VIDEOS_PER_SESSION:
                print(
                    f"  Warning: split session '{ses_id}' has {len(vids)} videos "
                    f"(expected {VIDEOS_PER_SESSION}). Skipping."
                )
                continue
            sessions.append({
                "session_id": ses_id,
                "start_time": parse_video_timestamp(vids[0]).replace(tzinfo=tzlocal()),
                "video_files": vids,
            })

    return sessions


# ── DANDI-style path helpers ──────────────────────────────────────────────────
def session_prefix(session_id: str) -> str:
    """e.g. 'sub-EY4152_ses-010224a'."""
    return f"sub-{SUBJECT_ID}_ses-{session_id}"


def video_dir_name(session_id: str) -> str:
    """
    Per-session video subfolder, mirroring 000409's '_ecephys+image' folders.
    We only have video data, so the modality suffix is just '_image'.
    """
    return f"{session_prefix(session_id)}_image"


def video_filename(session_id: str, camera_label: str) -> str:
    """
    e.g. 'sub-EY4152_ses-010224a_VideoBodyCamera.mp4'.
    Matches the IBL/DANDI 000409 convention.
    """
    return f"{session_prefix(session_id)}_Video{camera_label}.mp4"


def nwb_filename(session_id: str) -> str:
    """e.g. 'sub-EY4152_ses-010224a_desc-raw_image.nwb'."""
    return f"{session_prefix(session_id)}_desc-raw_image.nwb"


# ── NWB creation ──────────────────────────────────────────────────────────────
def make_nwb(session: dict, output_dir: Path) -> Path:
    """
    Create one NWB file for a session, organized DANDI-style:

        sub-<SUB>/
            sub-<SUB>_ses-<SES>_image/
                sub-<SUB>_ses-<SES>_Video<Label>.mp4   (symlinks)
            sub-<SUB>_ses-<SES>_desc-raw_image.nwb
    """
    subject_dir = output_dir / f"sub-{SUBJECT_ID}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    ses_id = session["session_id"]
    video_subdir = subject_dir / video_dir_name(ses_id)
    video_subdir.mkdir(parents=True, exist_ok=True)

    # Symlink each video into the per-session video folder under a
    # DANDI-compliant filename pairing it with a camera label.
    symlinked: list[tuple[str, Path]] = []  # (camera_label, link_path)
    for camera_label, video_path in zip(CAMERA_LABELS, session["video_files"]):
        link = video_subdir / video_filename(ses_id, camera_label)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(video_path)
        symlinked.append((camera_label, link))

    # Build NWB file
    nwbfile = NWBFile(
        session_description=f"Recording session {ses_id}",
        identifier=str(uuid4()),
        session_start_time=session["start_time"],
        session_id=ses_id,
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

    # ImageSeries.external_file paths are RELATIVE to the NWB file.
    # NWB lives in subject_dir; videos live in subject_dir/<video_subdir>/.
    rel_video_dir = video_subdir.name  # just the folder name

    for camera_label, link in symlinked:
        series = ImageSeries(
            name=f"Video{camera_label}",
            description=f"Video from {camera_label} (source: {link.resolve().name})",
            external_file=[f"{rel_video_dir}/{link.name}"],
            format="external",
            starting_frame=[0],
            rate=VIDEO_RATE,
            unit="n/a",
        )
        nwbfile.add_acquisition(series)

    nwb_path = subject_dir / nwb_filename(ses_id)
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
        for label, src in zip(CAMERA_LABELS, session["video_files"]):
            print(f"           ↳ {src.name}  →  Video{label}.mp4  (symlinked)")
        print()

    print("Done. Next steps:")
    print(f"  dandi validate {NWB_OUTPUT}/")
    print(f"  dandi organize {NWB_OUTPUT}/ --dandiset-path ./my_dandiset/")
    print(f"  dandi upload ./my_dandiset/")


if __name__ == "__main__":
    main()
