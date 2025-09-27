# Running Examples

This guide provides detailed instructions for running the Lightweight Stereo VIO system with both EuRoC and TUM VI datasets.

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
./build/tum_stereo config/tum_vo.yaml /path/to/dataset-corridor1_512_16

# Visual-Inertial Odometry (VIO) mode
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-corridor1_512_16
```

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

### Example Commands

#### EuRoC Dataset Examples

**Easy sequences (recommended for testing):**
```bash
# MH_01_easy with VO
./build/euroc_stereo config/euroc_vo.yaml /path/to/EuRoC/MH_01_easy

# MH_01_easy with VIO  
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_01_easy

# V1_01_easy with VIO
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/V1_01_easy
```

**Medium sequences:**
```bash
# MH_03_medium with VIO (more challenging)
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_03_medium

# V1_02_medium with VIO
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/V1_02_medium
```

#### TUM VI Dataset Examples

**Room sequences (recommended for testing):**
```bash
# Room1 with VO
./build/tum_stereo config/tum_vo.yaml /path/to/dataset-room1_512_16

# Room1 with VIO
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-room1_512_16
```

**Corridor sequences:**
```bash
# Corridor1 with VIO
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-corridor1_512_16

# Corridor3 with VIO
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-corridor3_512_16
```

**Outdoor sequences (challenging):**
```bash
# Outdoor1 with VIO (more robust to lighting changes)
./build/tum_stereo config/tum_vio.yaml /path/to/dataset-outdoors1_512_16
```

---

## Dataset Recommendations

### For Beginners
**EuRoC**: Start with `MH_01_easy` or `V1_01_easy`
**TUM VI**: Start with `room1` or `corridor1`

### Camera Models
- **EuRoC**: Uses pinhole camera model
- **TUM VI**: Uses fisheye camera model (wider field of view)

### Performance Tips
- Use **VO mode** for testing and lighter computational load
- Use **VIO mode** for better accuracy and robustness
- TUM VI sequences generally require more processing power due to fisheye distortion correction

---

## Troubleshooting

### Common Issues
1. **File not found errors**: Ensure dataset path points to the correct sequence directory
2. **Configuration errors**: Verify YAML file paths and syntax
3. **Performance issues**: Consider reducing image resolution or frame rate in config

### Expected Output
The system will display:
- Real-time trajectory visualization (if viewer is enabled)
- Processing statistics and timing information
- Feature tracking and matching results
- Final trajectory accuracy metrics (if ground truth is available)
