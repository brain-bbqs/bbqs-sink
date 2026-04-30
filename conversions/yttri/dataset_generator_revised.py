"""
IMPORTANT USAGE NOTE:

The established workflow is this: video + neural activity are saved
directly to a single drive (either F or E). The NWB data is then 
written to the OTHER connected drive due to space limitations. 
No information is written to any other drives, including C or D,
due to transfer speed limitations.

A symlink of the recorded video MUST be created in the target folder
on the OTHER drive at the path 
"F:\BIDSData\sub-{subject_id}\ses-{session_id}\video\video{timestamp}.mkv"
ONLY THEN may this script be executed. The BIDS format requires that
external objects like videos be referenced relative to the created
NWB file. Creating a symlink allows for a relative path on the same
drive as the NWB file.

Expected session folder contents (on recording drive):
    clock.*
    lfp.*
    spike.*
    video{timestamp}.mkv
    digital_inputs.*
    digital_inputs_clock.*
    camera_timestamps.*

Output NWB path:
    F:\BIDSData\sub-{subject_id}\ses-{session_id}\
        sub-{subject_id}_ses-{session_id}_ecephys+behavior.nwb
"""

from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb.ecephys import ElectricalSeries
from pynwb.image import ImageSeries
from pynwb.file import Subject
from pynwb.epoch import TimeIntervals
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import os
import pandas as pd
import time
import ctypes
import sys

# ===========================================================================
# CONFIG — edit these before each session
# ===========================================================================
SUBJECT_ID    = "EY9166"
DATE_OF_BIRTH = datetime(2025, 11, 18, tzinfo=ZoneInfo("America/New_York"))
SEX           = "M"

# Folder on the recording drive containing raw session files
SESSION_FOLDER = r"Z:\Aden\EY9166\3_24"

# Root output folder on the OTHER drive (NWB + symlinked video go here)
BIDS_ROOT = r"G:/"

# Bonsai acquisition parameters
AP_GAIN = 1000
LFP_GAIN = 50
# ===========================================================================

def parse_timestamp(time_str):
    """Parse 'YYYY-MM-DDTHH_MM_SS' into individual ints."""
    ymd, hms = time_str.split("T")
    year, month, day = [int(v) for v in ymd.split("-")]
    hour, minute, second = [int(v) for v in hms.split("_")]
    return year, month, day, hour, minute, second


def parse_folder(folder_path):
    """
    Discover all expected files in a session folder and return a dict of
    labelled paths plus the parsed session start datetime.
    """
    files = os.listdir(folder_path)

    found = {
        "clock": None,
        "lfp": None,
        "spike": None,
        "video": None,
        "digital_inputs_clock": None,   # check before "digital_inputs"
        "digital_inputs": None,
        "camera_timestamps": None,
    }

    for filename in files:
        lower = filename.lower()
        if "digital_inputs_clock" in lower:
            found["digital_inputs_clock"] = filename
        elif "digital_inputs" in lower:
            found["digital_inputs"] = filename
        elif "camera_timestamps" in lower:
            found["camera_timestamps"] = filename
        elif "clock" in lower:
            found["clock"] = filename
        elif "lfp" in lower:
            found["lfp"] = filename
        elif "spike" in lower:
            found["spike"] = filename
        elif "video" in lower:
            found["video"] = filename

    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing expected files in session folder: {missing}")

    # Extract timestamp from video filename: video2026-03-21T15_26_22.mkv
    video_stem = os.path.splitext(found["video"])[0]   # strip extension
    timestamp_str = video_stem[len("video"):]           # strip leading "video"
    year, month, day, hour, minute, second = parse_timestamp(timestamp_str)
    session_start = datetime(year, month, day, hour, minute, second,
                             tzinfo=ZoneInfo("America/New_York"))

    paths = {k: os.path.join(folder_path, v) for k, v in found.items()}

    return paths, session_start


def build_output_paths(bids_root, subject_id, session_start):
    """
    Return the output NWB path and the expected symlinked video path,
    following DANDI/BIDS conventions.

        <bids_root>/
            sub-{subject_id}/
                ses-{YYYYMMDDTHHmmss}/
                    video/
                        video{timestamp}.mkv   <- symlink created manually
                    sub-{subject_id}_ses-{YYYYMMDDTHHmmss}_ecephys+behavior.nwb
    """
    ts = session_start.strftime("%Y%m%dT%H%M%S")
    ses_id    = f"ses-{ts}"
    sub_label = f"sub-{subject_id}"
    ses_dir   = os.path.join(bids_root, sub_label, ses_id)
    video_dir = os.path.join(ses_dir, "video")

    os.makedirs(ses_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    nwb_filename  = f"{sub_label}_{ses_id}_ecephys+behavior.nwb"
    nwb_path      = os.path.join(ses_dir, nwb_filename)

    # Symlink is expected to already exist here (see header note)
    video_symlink = os.path.join(video_dir,
                                 f"video{session_start.strftime('%Y-%m-%dT%H_%M_%S')}.mkv")

    return nwb_path, video_symlink, ses_dir

def create_video_symlink(source_video_path, symlink_path):
    """
    Create a symlink at symlink_path pointing to source_video_path.
    On Windows, requires either Developer Mode or admin privileges.
    Skips silently if a valid symlink already exists.
    """
    # Remove broken symlink if present
    if os.path.islink(symlink_path) and not os.path.exists(symlink_path):
        os.remove(symlink_path)
        print("  Removed broken existing symlink.")
 
    if os.path.exists(symlink_path):
        print(f"  Symlink already exists, skipping:\n    {symlink_path}")
        return
 
    try:
        os.symlink(source_video_path, symlink_path)
        print(f"  Symlink created:\n"
              f"    {symlink_path}\n"
              f"    -> {source_video_path}")
    except OSError as e:
        raise OSError(
            f"Failed to create symlink. On Windows, symlink creation requires either:\n"
            f"  (a) Developer Mode enabled (Settings > For Developers), or\n"
            f"  (b) running this script as Administrator.\n"
            f"Original error: {e}"
        )


def main():
    starting_time = time.time()

    # === Discover session files ===
    paths, session_start = parse_folder(SESSION_FOLDER)

    nwb_path, video_symlink_path, ses_dir = build_output_paths(
        BIDS_ROOT, SUBJECT_ID, session_start
    )

    # Relative path from NWB file to symlinked video (required by DANDI)
    video_rel_path = os.path.relpath(video_symlink_path, start=ses_dir).replace("\\", "/")

    # === Create video symlink on output drive ===
    print("Creating video symlink...")
    create_video_symlink(
        source_video_path=paths["video"],
        symlink_path=video_symlink_path
    )

    # === Recording parameters ===
    ap_sample_rate  = 30000.0
    lfp_sample_rate = 2500.0
    digital_clock_hz = 5e6
    basler_clock_hz = 1e9
    n_channels_ap   = 384
    dtype           = np.uint16

    # === Load TTL CSVs ===
    ttl_port_vals_path  = paths["digital_inputs"]
    ttl_timestamps_path = paths["digital_inputs_clock"]

    ttl_ports      = pd.read_csv(ttl_port_vals_path,  header=None)
    ttl_timestamps = pd.read_csv(ttl_timestamps_path, header=None)

    ttl_port_binary = (ttl_ports[5] == " Pin5").to_numpy()
    ttl_starts = ttl_timestamps[0][ttl_port_binary].astype(float).to_numpy() / digital_clock_hz
    ttl_ends   = ttl_timestamps[0][~ttl_port_binary].astype(float).to_numpy() / digital_clock_hz

    # === Create NWBFile ===
    ses_label = session_start.strftime("%Y-%m-%d")
    ses_id    = f"ses-{session_start.strftime('%Y%m%dT%H%M%S')}"

    nwbfile = NWBFile(
        session_description=ses_label,
        identifier=f"{SUBJECT_ID}_{session_start.strftime('%Y-%m-%dT%H_%M_%S')}",
        session_start_time=session_start,
        subject=Subject(
            subject_id=SUBJECT_ID,
            description=SUBJECT_ID,
            species="Mus musculus",
            date_of_birth=DATE_OF_BIRTH,
            sex=SEX,
        )
    )

    # === Add device and electrode group ===
    device = nwbfile.create_device(
        name="Neuropixels-Probe",
        description="Neuropixels 1.0e",
        manufacturer="imec"
    )
    eg = nwbfile.create_electrode_group(
        name="ProbeGroup",
        description="Neuropixels Bank A (channels 0–383), spanning M1, dorsal striatum, and ventral striatum",
        device=device,
        location="M1, dorsal striatum, ventral striatum"
    )

    for idx in range(n_channels_ap):
        nwbfile.add_electrode(
            id=idx, x=np.nan, y=np.nan, z=np.nan,
            imp=np.nan, location="M1",
            filtering="none", group=eg
        )

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(range(n_channels_ap)),
        description="All electrodes"
    )

    # === Load raw binary data ===
    def load_binary_memmap(file_path, n_channels, dtype=np.int16):
        n_bytes   = os.path.getsize(file_path)
        n_samples = n_bytes // np.dtype(dtype).itemsize // n_channels
        return np.memmap(file_path, dtype=dtype, mode="r", shape=(n_samples, n_channels))

    print("Loading AP...")
    ap_data = load_binary_memmap(paths["spike"], n_channels_ap, dtype)
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    print("Loading LFP...")
    lfp_data = load_binary_memmap(paths["lfp"], n_channels_ap, dtype)
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    ap_h5  = H5DataIO(ap_data,  compression=None, chunks=(10000, ap_data.shape[1]),  link_data=True)
    lfp_h5 = H5DataIO(lfp_data, compression=None, chunks=(10000, lfp_data.shape[1]), link_data=True)

    ap_conversion = (1171.875 / AP_GAIN) * 1e-6
    ap_offset     = -512 * ap_conversion

    lfp_conversion = (1171.875 / LFP_GAIN) * 1e-6
    lfp_offset     = -512 * lfp_conversion

    # === Add ElectricalSeries ===
    print("Adding AP...")
    nwbfile.add_acquisition(ElectricalSeries(
        name="AP_raw",
        data=ap_h5,
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=ap_sample_rate,
        conversion=ap_conversion,
        offset=ap_offset
    ))
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    print("Adding LFP...")
    nwbfile.add_acquisition(ElectricalSeries(
        name="LFP_raw",
        data=lfp_h5,
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=lfp_sample_rate,
        conversion=lfp_conversion,
        offset=lfp_offset
    ))
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    # === Add TTL pulses ===
    print("Adding TTL intervals...")
    ttl_intervals = TimeIntervals(
        name="ttl_pulses",
        description="Start and stop times of TTL pulses from Port 5"
    )
    for start, stop in zip(ttl_starts, ttl_ends):
        ttl_intervals.add_interval(start_time=float(start), stop_time=float(stop), tags=["TTL5"])
    nwbfile.add_time_intervals(ttl_intervals)
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    cam_ts = pd.read_csv(paths['camera_timestamps'], header=None)
    cam_timestamps = cam_ts.squeeze().astype(float).to_numpy()
    cam_timestamps_s = (cam_timestamps - cam_timestamps[0]) / basler_clock_hz

    # === Add video ===
    print("Adding video...")
    video = ImageSeries(
        name="BehaviorVideo",
        external_file=[video_rel_path],
        format="external",
        starting_frame=[0],
        timestamps=cam_timestamps_s,
        description="Behavior video from camera positioned under arena"
    )
    nwbfile.add_acquisition(video)
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    clock_data = np.memmap(paths["clock"], dtype=np.uint64, mode="r")
    # Add as a TimeSeries in acquisition so it's preserved for alignment
    ephys_clock = TimeSeries(
        name="onix_ephys_clock",
        data=H5DataIO(clock_data, compression=None, link_data=True),
        unit="clock_ticks",
        rate=ap_sample_rate,
        description="ONIX acquisition clock timestamps for AP samples"
    )
    nwbfile.add_acquisition(ephys_clock)
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")

    # === Write NWB file ===
    print(f"Writing NWB to:\n  {nwb_path}")
    with NWBHDF5IO(nwb_path, mode="w") as io:
        io.write(nwbfile)
    print(f"  Done | {time.time() - starting_time:.1f}s elapsed")
    print(f"\nNWB file saved to: {nwb_path}")


if __name__ == "__main__":
    main()
