# Running Examples

This guide provides detailed instructions for running the Lightweight Stereo VIO system with EuRoC, TUM VI, and RGBD datasets.

## Quick Start

### EuRoC Dataset

#### Native Build
```bash
# Visual Odometry (VO) mode
./build/euroc_stereo config/euroc_vo.yaml /path/to/EuRoC/MH_01_easy

# Visual-Inertial Odometry (VIO) mode
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_01_easy
```

#### Docker
```bash
# Visual Odometry (VO) mode
./docker.sh run vo /path/to/EuRoC/MH_01_easy

# Visual-Inertial Odometry (VIO) mode
./docker.sh run vio /path/to/EuRoC/MH_01_easy
```

### TUM VI Dataset

#### Native Build
```bash
# Visual Odometry (VO) mode
./build/tum_stereo config/tum_vo.yaml /path/to/dataset-room1_512_16

# Visual-Inertial Odometry (VIO) mode
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-room1_512_16
```

#### Docker
```bash
# Visual Odometry (VO) mode
./docker.sh run vo /path/to/dataset-room1_512_16

# Visual-Inertial Odometry (VIO) mode
./docker.sh run vio /path/to/dataset-room1_512_16
```

### RGBD Dataset (Intel RealSense D435/D435i)

#### Native Build
```bash
# Visual Odometry (VO) mode with RGBD
./build/rgbd_example config/d435.yaml /path/to/RGBD/dataset
```

**Note:** RGBD + IMU (VIO mode) is not yet implemented. Currently only VO mode is supported.

---

## Detailed Usage

### Command Line Interface

#### EuRoC Dataset
The general syntax for running with EuRoC dataset:

```bash
./build/euroc_stereo <config_file_path> <euroc_dataset_path>
```

**Parameters:**
- `<config_file_path>`: Path to the YAML configuration file (euroc_vo.yaml or euroc_vio.yaml)
- `<euroc_dataset_path>`: Path to a specific EuRoC sequence directory

#### TUM VI Dataset
The general syntax for running with TUM VI dataset:

```bash
./build/tum_stereo <config_file_path> <tum_dataset_path>
```

**Parameters:**
- `<config_file_path>`: Path to the YAML configuration file (tum_vo.yaml or tum_vio.yaml)
- `<tum_dataset_path>`: Path to a specific TUM VI sequence directory

#### RGBD Dataset (Intel RealSense D435/D435i)
The general syntax for running with RGBD dataset:

```bash
./build/rgbd_example <config_file_path> <rgbd_dataset_path>
```

**Parameters:**
- `<config_file_path>`: Path to the YAML configuration file (d435.yaml)
- `<rgbd_dataset_path>`: Path to RGBD dataset directory containing `color/`, `depth/`, and `timestamps.txt`

**Dataset Structure:**
```
dataset_folder/
├── color/           # RGB images
├── depth/           # Depth images
└── timestamps.txt   # Timestamp file
```

### Configuration Files

The system behavior is controlled by YAML configuration files located in the `config/` directory:

#### Visual Odometry (VO) Mode
**EuRoC**: `config/euroc_vo.yaml`
**TUM VI**: `config/tum_vo.yaml`
- Uses only camera data
- Suitable for scenarios with good visual features
- Lower computational requirements

#### Visual-Inertial Odometry (VIO) Mode  
**EuRoC**: `config/euroc_vio.yaml`
**TUM VI**: `config/tum_vio.yaml`
- Uses both camera and IMU data
- More robust to motion blur and challenging conditions
- Higher computational requirements but better accuracy

#### RGBD Visual Odometry Mode
**D435/D435i**: `config/d435.yaml`
- Uses RGB-D camera data (color + depth)
- Suitable for indoor environments
- Currently supports VO mode only (VIO mode not yet implemented)

### Example Commands

#### EuRoC Dataset Examples

```bash
./build/euroc_stereo config/euroc_vo.yaml /path/to/EuRoC/MH_05_difficult

```
#### TUM VI Dataset Examples


**Room sequences:**
```bash
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-room1_512_16
```

#### RGBD Dataset Examples

**OpenLORIS-Scene Dataset (Cafe sequence):**

The example RGBD dataset is from the [OpenLORIS-Scene dataset](https://lifelong-robotic-vision.github.io/dataset/scene.html), specifically the cafe sequence.

**Download:**
- [Cafe sequence (preprocessed)](https://drive.google.com/file/d/1e5-8tpaT5dB2T1-pffpMBEjVnzhrlXMg/view?usp=sharing)

**Original dataset source:**
- OpenLORIS-Scene: A Robotic Vision Dataset and Benchmark for Lifelong SLAM
- Website: https://lifelong-robotic-vision.github.io/dataset/scene.html

**Run the example:**
```bash
./build/rgbd_example config/d435.yaml /path/to/cafe_dataset
```

**Expected output:**
- 3D visualization with Pangolin viewer
- Dense point cloud visualization (toggle with UI button)
- Trajectory estimation in real-time

