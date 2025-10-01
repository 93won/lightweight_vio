# Downloading Datasets

This guide explains how to download the datasets required for testing the stereo visual-inertial odometry system.

## Supported Datasets

- **EuRoC MAV Dataset**: Indoor drone flights with stereo cameras and IMU
- **TUM VI Dataset**: Indoor/outdoor sequences with fisheye stereo cameras and IMU
- **TUM RGB-D Dataset**: Indoor sequences with RGB-D cameras for visual odometry

---

## EuRoC Dataset

### EuRoC Dataset Overview

The EuRoC MAV (European Robotics Challenge Micro Aerial Vehicle) dataset contains stereo camera images, IMU data, and ground truth trajectory data from a micro aerial vehicle flying in indoor environments. It includes 11 sequences total with different difficulty levels.

## Download Options

### Option 1: Using the Provided Script (Recommended)

The repository includes a convenience script to download all EuRoC MAV datasets automatically.

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/of/dataset
```

This will download all 11 sequences into the specified directory structure:
```
/path/of/dataset/
├── EuRoC/
│   ├── MH_01_easy/
│   ├── MH_02_easy/
│   ├── MH_03_medium/
│   ├── MH_04_difficult/
│   ├── MH_05_difficult/
│   ├── V1_01_easy/
│   ├── V1_02_medium/
│   ├── V1_03_difficult/
│   ├── V2_01_easy/
│   ├── V2_02_medium/
│   └── V2_03_difficult/
```

### Option 2: Manual Download

You can manually download specific sequences from the [EuRoC dataset website](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

1. Visit the EuRoC dataset page
2. Download the desired sequences in ASL Dataset Format
3. Extract the files into the appropriate directory structure

## Dataset Structure

Each sequence contains the following files:
```
MH_01_easy/
├── mav0/
│   ├── cam0/           # Left camera images
│   │   ├── data/
│   │   └── data.csv
│   ├── cam1/           # Right camera images
│   │   ├── data/
│   │   └── data.csv
│   ├── imu0/           # IMU data
│   │   └── data.csv
│   ├── leica0/         # Ground truth poses
│   │   └── data.csv
│   └── state_groundtruth_estimate0/
│       └── data.csv
```

## Sequence Recommendations

## Storage Requirements

- **Single sequence**: ~1-3 GB
- **All sequences**: ~20-25 GB

Make sure you have sufficient disk space before downloading all sequences.

## TUM VI Dataset

### TUM VI Dataset Overview

The TUM VI (Technical University of Munich Visual-Inertial) dataset contains stereo fisheye camera images, IMU data, and ground truth trajectory data recorded in various indoor and outdoor environments. It includes 28 sequences total with different environments and motion patterns.

### Download Options

#### Option 1: Using the Provided Script (Recommended)

The repository includes a convenience script to download TUM VI datasets automatically.

```bash
chmod +x script/download_tum_vi.sh
./script/download_tum_vi.sh                      # Download ALL datasets (default)
./script/download_tum_vi.sh /path/to/datasets    # Download ALL datasets to specified path
./script/download_tum_vi.sh corridor1            # Download specific dataset
./script/download_tum_vi.sh corridor1 /path/to/datasets  # Download specific dataset to path
```

This will download sequences into the specified directory structure:
```
/path/to/datasets/
├── dataset-corridor1_512_16/
├── dataset-corridor2_512_16/
├── dataset-room1_512_16/
├── dataset-magistrale1_512_16/
└── ...
```

#### Option 2: Manual Download

You can manually download specific sequences from the [TUM VI dataset website](https://vision.in.tum.de/data/datasets/visual-inertial-dataset).

1. Visit the TUM VI dataset page
2. Download the desired sequences in EuRoC format (512x512)
3. Extract the files into the appropriate directory structure

### Dataset Structure

Each TUM VI sequence follows the EuRoC format:
```
dataset-corridor1_512_16/
├── mav0/
│   ├── cam0/           # Left fisheye camera images
│   │   ├── data/
│   │   └── data.csv
│   ├── cam1/           # Right fisheye camera images
│   │   ├── data/
│   │   └── data.csv
│   ├── imu0/           # IMU data
│   │   └── data.csv
│   └── mocap0/         # Ground truth poses
│       └── data.csv
```

### Available Sequences

**Environment Types:**
- **Corridor sequences** (5): corridor1-5 - Indoor office corridors
- **Room sequences** (6): room1-6 - Indoor rooms and labs  
- **Magistrale sequences** (6): magistrale1-6 - Indoor large halls
- **Outdoor sequences** (8): outdoors1-8 - Outdoor campus environments
- **Slides sequences** (3): slides1-3 - Presentation rooms

### Storage Requirements

- **Single sequence**: ~1-4 GB
- **All sequences**: ~150+ GB

Make sure you have sufficient disk space before downloading all sequences.



---

## TUM RGB-D Dataset

### TUM RGB-D Dataset Overview

The TUM RGB-D dataset contains RGB-D camera sequences with ground truth trajectory data recorded in various indoor environments. It includes multiple sequences with different objects, lighting conditions, and camera motions, specifically designed for RGB-D SLAM and visual odometry evaluation.

### Download Options

#### Option 1: Official TUM RGB-D Dataset

You can download the complete dataset from the official TUM RGB-D dataset website:

**Official Dataset:** [https://cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

The dataset includes various sequences such as:
- `rgbd_dataset_freiburg1_xyz` - Simple translational motions
- `rgbd_dataset_freiburg1_rpy` - Simple rotational motions  
- `rgbd_dataset_freiburg2_desk` - Desk scenes with objects
- `rgbd_dataset_freiburg3_office` - Office environments

#### Option 2: Quick Start Sample Dataset

For quick testing and evaluation, we provide a sample dataset ready to use:

**Sample Dataset:** [https://drive.google.com/file/d/1HRWnBq9kq-m4gkjVuLdKUHeaZcztacIs/view?usp=sharing](https://drive.google.com/file/d/1HRWnBq9kq-m4gkjVuLdKUHeaZcztacIs/view?usp=sharing)

This sample includes the `rgbd_dataset_freiburg2_desk` sequence, which is ideal for testing RGB-D visual odometry functionality.

### Dataset Structure

Each RGB-D sequence contains the following files:
```
rgbd_dataset_freiburg2_desk/
├── rgb.txt              # RGB image timestamps and filenames
├── depth.txt            # Depth image timestamps and filenames
├── groundtruth.txt      # Ground truth trajectory
├── rgb/                 # RGB images
│   ├── 1311868164.363181.png
│   ├── 1311868164.399026.png
│   └── ...
└── depth/               # Depth images
    ├── 1311868164.374026.png
    ├── 1311868164.407668.png
    └── ...
```

### Storage Requirements

- **Single sequence**: ~500MB - 2GB depending on sequence length
- **Complete dataset**: ~30+ GB

### Installation Instructions

1. Download the dataset from either source above
2. Extract to your preferred dataset directory (e.g., `/home/user/data/RGBD/`)
3. The directory structure should look like:
```
/home/user/data/RGBD/
└── rgbd_dataset_freiburg2_desk/
    ├── rgb.txt
    ├── depth.txt
    ├── groundtruth.txt
    ├── rgb/
    └── depth/
```

---



## Troubleshooting

### Download Issues
- Check your internet connection
- Ensure sufficient disk space
- Verify the target directory has write permissions

### Script Permissions
```bash
chmod +x script/download_euroc.sh
chmod +x script/download_tum_vi.sh
```





## Next Steps

After downloading the dataset:
1. Follow the [Installation Guide](Install.md) to build the project
2. See the [Running Examples](Running_Example.md) to test with the downloaded data
