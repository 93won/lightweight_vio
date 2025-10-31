# Lightweight Stereo VIO

This is a lightweight stereo visual-inertial odometry (VIO) project designed for real-time performance. It utilizes feature tracking, IMU pre-integration, sliding window optimization with Ceres Solver, and Pangolin for visualization.

This project implements the **statistical uncertainty learning** method proposed in:

**Statistical Uncertainty Learning for Robust Visual-Inertial State Estimation**  
Seungwon Choi, Donggyu Park, Seo-Yeon Hwang, Tae-Wan Kim  
[arXiv:2510.01648](https://arxiv.org/abs/2510.01648)

If you use this work in your research, please cite:

```bibtex
@misc{choi2025statistical,
      title={Statistical Uncertainty Learning for Robust Visual-Inertial State Estimation}, 
      author={Seungwon Choi and Donggyu Park and Seo-Yeon Hwang and Tae-Wan Kim},
      year={2025},
      eprint={2510.01648},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.01648}
}
```

## License

This project is licensed under the 🚀MIT License🚀 - see the [LICENSE](LICENSE) file for details.

## Demo
[![Demo](https://img.youtube.com/vi/Fy-1zU6xXm4/0.jpg)](https://youtu.be/Fy-1zU6xXm4)

## Installation

📋 **[Installation Guide](docs/Install.md)** - Complete installation instructions for both Docker and native builds

## Dataset Download

📁 **[Dataset Download Guide](docs/Download_Dataset.md)** - Dataset download and preparation

### Supported Datasets

- **EuRoC MAV Dataset** - Stereo + IMU (Visual-Inertial Odometry)
- **TUM-VI Dataset** - Stereo + IMU (Visual-Inertial Odometry)
- **Intel RealSense D435/D435i** - RGBD (Visual Odometry only)
  - Supports custom RGBD datasets with color/ and depth/ folders
  - Example configuration: `config/d435.yaml`
  - ⚠️ Note: RGBD + IMU fusion (VIO mode) is under development

## Running the Application

🚀 **[Running Examples](docs/Running_Example.md)** - Usage examples

## Performance Analysis and Evaluation

📊 **[Performance Analysis Guide](docs/Performance_Analysis.md)** - Comprehensive performance evaluation and benchmarking

## Project Structure

The source code is organized into the following directories:

- `app/`: Main application entry points
- `src/`:
  - `database/`: Data structures for `Frame` (including IMU data), `MapPoint`, and `Feature`
  - `processing/`: Core VIO modules, including `Estimator`, `FeatureTracker`, `IMUHandler`, and `Optimizer`
  - `optimization/`: Ceres Solver cost functions for visual reprojection errors and IMU pre-integration constraints
  - `viewer/`: Pangolin-based visualization
  - `util/`: Utility functions for configuration and data loading
- `thirdparty/`: External libraries (Ceres, Pangolin, Sophus, spdlog)
- `config/`: Configuration files for VO and VIO modes
- `docs/`: Detailed documentation guides

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## References

This project's IMU pre-integration implementation is based on the following paper:

```bibtex
@article{forster2016manifold,
  author = {Forster, Christian and Carlone, Luca and Dellaert, Frank and Scaramuzza, Davide},
  year = {2016},
  month = {08},
  title = {On-Manifold Preintegration for Real-Time Visual-Inertial Odometry},
  volume = {33},
  journal = {IEEE Transactions on Robotics},
  doi = {10.1109/TRO.2016.2597321}
}
```



