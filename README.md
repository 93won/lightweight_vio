# Lightweight Stereo VIO

This is a lightweight stereo visual-inertial odometry (VIO) project designed for real-time performance. It utilizes feature tracking, IMU pre-integration, sliding window optimization with Ceres Solver, and Pangolin for visualization.

## License

This project is licensed under the 🚀MIT License🚀 - see the [LICENSE](LICENSE) file for details.

## Demo
[![Stereo VIO Demo](https://img.youtube.com/vi/41o9R-rKQ1s/0.jpg)](https://youtu.be/41o9R-rKQ1s)

## Installation

📋 **[Installation Guide](docs/Install.md)** - Complete installation instructions for both Docker and native builds

## Dataset Download

📁 **[Dataset Download Guide](docs/Download_Dataset.md)** - Dataset download and preparation

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



