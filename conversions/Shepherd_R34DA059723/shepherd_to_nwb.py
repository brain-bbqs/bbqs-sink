import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from hdmf.backends.hdf5.h5_utils import H5DataIO
from natsort import natsorted
from neuroconv.datainterfaces import DeepLabCutInterface
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb.image import ImageSeries

# TODO: add a thermosistor aligned w/ behavior to the processed nwb


def load_cfg(cfg_path: Path):
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {cfg_path}")


def cfg_get(*path, CFG, default=None):
    node = CFG
    for key in path:
        node = node.get(key, {})
    return node or default


def get_video_info_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,nb_frames,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)

    stream = info["streams"][0]
    duration = float(stream.get("duration", 0))
    frame_count = int(stream.get("nb_frames", 0))
    return duration, frame_count


def get_all_files(DIFolder, AIFolder):

    # get files
    DIFiles = [f for f in os.listdir(DIFolder)]
    AIFiles = [f for f in os.listdir(AIFolder)]
    DIFiles = natsorted(DIFiles)
    AIFiles = natsorted(AIFiles)

    return DIFiles, AIFiles


def convert_to_timeseries(data, rate, unit, start_time, description, type):

    dataAll = np.concatenate(data, axis=1)
    print(f"shape of {type} data: {dataAll.shape}")

    # assumes time is always longer than number of data types, always want time to be 0th axis
    if dataAll.shape[0] < dataAll.shape[1]:
        print(f"Transposing {type} data so that time is the 0th axis, comment this out if you think this is a mistake")
        dataAll = dataAll.T

    ts = TimeSeries(
        name=type, data=dataAll, unit=unit, starting_time=float(start_time), rate=float(rate), description=description
    )

    return ts


def build_nwb(input_file, subject_name, out_nwb, cfg):

    # get metadata for NWB file
    date_string = cfg_get("nwb_raw", "start_time", CFG=cfg)
    date_format = "%Y-%m-%d%H:%M:%S%z"
    session_start = datetime.strptime(date_string, date_format)

    # create the raw NWB file
    nwbfile_raw = NWBFile(
        session_description=cfg_get("nwb_raw", "description", CFG=cfg),
        identifier=cfg_get("nwb_raw", "identifier", CFG=cfg),
        session_start_time=session_start,
        experimenter=cfg_get("session", "experimenter", CFG=cfg),
        institution=cfg_get("session", "instituition", CFG=cfg),
    )

    # subject definition
    subject_id = subject_name
    species = cfg_get("subject", "species", CFG=cfg)
    age = cfg_get("subject", "age", CFG=cfg)
    sex = cfg_get("subject", "sex", CFG=cfg)

    # add subject information
    nwb_subject = Subject(
        subject_id=subject_id,
        species=species,  # required
        age=age,  # required for EMBER upload
        sex=sex,  # required for EMBER upload
    )

    nwbfile_raw.subject = nwb_subject

    # add acquisition data (digital & analog)

    DIFolder = input_file + "/digital"
    AIFolder = input_file + "/analog"

    DIFiles, AIFiles = get_all_files(DIFolder, AIFolder)

    DIAll = [np.load(os.path.join(DIFolder, f)).astype(np.int16) for f in DIFiles]
    AIAll = [np.load(os.path.join(AIFolder, f)).astype(np.int16) for f in AIFiles]

    unit = cfg_get("timeseries", "digital", "unit", CFG=cfg)
    start_time = cfg_get("timeseries", "digital", "start_time", CFG=cfg)
    rate = cfg_get("timeseries", "digital", "rate", CFG=cfg)
    description_DI = cfg_get("timeseries", "digital", "description", CFG=cfg)
    description_AI = cfg_get("timeseries", "analog", "description", CFG=cfg)

    DI_ts = convert_to_timeseries(DIAll, rate, unit, start_time, description_DI, "DigitalInput")
    AI_ts = convert_to_timeseries(AIAll, rate, unit, start_time, description_AI, "AnalogInput")

    # assumes video starts when digital input starts
    first_zero_index = np.where(DIAll[0] == 0)[0][0]  # find where the digital input single row 0 first goes to 0
    print(f"First zero index in DIAll[0]: {first_zero_index}")

    video_start_time = first_zero_index / float(rate)
    print(f"Video start time: {video_start_time} seconds")

    nwbfile_raw.add_acquisition(DI_ts)
    nwbfile_raw.add_acquisition(AI_ts)

    path_to_save_raw = out_nwb + "_desc-raw.nwb"

    # save raw file
    with NWBHDF5IO(path_to_save_raw, "w") as io:
        io.write(nwbfile_raw)

    # get processed metadata
    date_string = cfg_get("nwb_processed", "start_time", CFG=cfg)
    date_format = "%Y-%m-%d%H:%M:%S%z"
    session_start = datetime.strptime(date_string, date_format)

    # create processed NWB
    nwbfile_processed = NWBFile(
        session_description=cfg_get("nwb_processed", "description", CFG=cfg),
        identifier=cfg_get("nwb_processed", "identifier", CFG=cfg),
        session_start_time=session_start,
    )

    nwbfile_processed.subject = nwb_subject

    # package video data
    config_file_path = cfg_get("deep-lab-cut", "config", CFG=cfg)
    pose_estimation_folder = Path(input_file + "/pose_estimation/")
    pose_estimation_file = next(pose_estimation_folder.glob("*.h5"), None)

    video_ext = {".avi"}
    root = Path(input_file + "/videos")
    video_paths = [str(p) for p in root.rglob("*") if p.suffix.lower() in video_ext]

    interface = DeepLabCutInterface(
        file_path=pose_estimation_file, config_file_path=config_file_path, subject_name=subject_id, verbose=False
    )

    # TODO: ask Neha about this assumption
    duration, frame_count = get_video_info_ffprobe(video_paths[0])
    fps = round(frame_count / duration)
    print(f"Video duration: {duration} seconds, Frame count: {frame_count}, FPS: {fps}")

    ts = video_start_time + np.arange(frame_count, dtype=float) / fps

    interface.set_aligned_timestamps(ts)
    metadata = interface.get_metadata()

    interface.add_to_nwbfile(nwbfile=nwbfile_processed, metadata=metadata)

    # make a seperate ImageSeries per angle, embed video in an ImageSeries and apply lossless compression
    video_arrays = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        video_arrays.append(np.array(frames))
        cap = cv2.VideoCapture(video_path)

    for i, video_array in enumerate(video_arrays):
        duration, frame_count = get_video_info_ffprobe(video_paths[i])
        fps = round(frame_count / duration)
        wrapped_data = H5DataIO(
            data=video_array,
            compression="gzip",
        )
        video_series = ImageSeries(
            name=f"VideoAngle{i+1}",
            data=wrapped_data,
            unit="px",
            starting_time=video_start_time,
            rate=fps,
        )
        nwbfile_processed.add_acquisition(video_series)

    path_to_save_processed = out_nwb + "_desc-processed.nwb"
    with NWBHDF5IO(path_to_save_processed, "w") as io:
        io.write(nwbfile_processed)

    print(f"Wrote NWB file to: {out_nwb}")
    return


def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Convert MATLAB .mat to NWB")
    p.add_argument("--input_folder", required=True, type=str, help="File path to input files folder")
    p.add_argument("--subject", required=True, help="Subject identifier (string)")
    p.add_argument("--session", required=True, type=int, help="Session number (int)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML config (default: ./config.yaml)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output NWB file path (default: derive from --mat)",
    )
    return p.parse_args()


def main():

    print("parsing args")

    args = parse_args()

    cfg = load_cfg(args.config)

    # TODO: create subject and session folders automatically? don't let people input out file names?

    out_path = args.out
    if out_path is None:
        out_path = f"sub-{args.subject}_ses-{args.session}.nwb"
    else:
        print(f"out path: {out_path}")

    build_nwb(input_file=args.input_folder, subject_name=args.subject, out_nwb=out_path, cfg=cfg)


if __name__ == "__main__":

    """
    Example CLI usage
    --------
    python shepherd_to_nwb.py --input_folder example_data/2025-06-25 --subject XYZ --session ABC --config ./config.yaml
    """
    print("calling main")

    main()
