## Overview

This is the conversion script for converting data from the Shepherd team (R34DA059723). The experiment involves multicamera angles of rat eating behavior. You can find a subset of the data on EMBER-DANDI, ['Mouse food sniff'](https://dandi.emberarchive.org/dandiset/000463/draft).

### Data Overview
The data includes analog signals at 50 kHz. Which includes thermister and sound data that is recorded simultaneously with video. It also includes digital signals, the TTL camera exposure (where False means it is on) and binary triggers. Lastly, there are video files and pose estimation data which was extracted using DeepLabCut

### Data Organization

The script assumes that input data is organized like the following:

```
input_folder/
├── digital/
│   ├── XYZ.npy ... (any number of files)
├── analog/
│   ├── XYZ.npy ... (any number of files)
├── videos/
│   ├── XYZ.avi ... (any number of files)
└── pose_estimation/
    └── XYZ.h5 (assumes only one pose estimation file)
```

### Folder and File Structure
```
sub-XYZ/
├── sub-XYZ_ses-YYYYMMDD_raw.nwb
│   ├── session_start_time
│   ├── session_description
│   ├── identifier
│   ├── general
│   │   └── subject
│   └── acquisition
│       ├── digital
│       │   └── TTL camera exposure and Triggers (TimeSeries)
│       └── analog
│          └── Thermistor and Sound (TimeSeries)
└── sub-XYZ_ses-YYYYMMDD_processed.nwb
    ├── session_start_time
    ├── session_description
    ├── identifier
    ├── general
    │   └── subject
    └── acquisition
        ├── video_series (ImageSeries)
        └── DLC_pose_estimation (DeepLabCutInterface)
```
