# Lightweight VIO

A lightweight **monocular** visual-inertial odometry system.

## ⚠️ Development Status

- 🚧 **Under active development** - system stabilization in progress
- ⚠️ **Unstable** - may exhibit instability in some scenarios
- ✅ **Tested only on EuRoC MH_01_easy** - other datasets not yet validated

> 💡 **For stable performance**, use the **stereo version** instead.

## Installation

```bash
./build.sh
```

For detailed installation instructions: 📋 **[Installation Guide](docs/Install.md)**

## Sample Data

Download EuRoC MAV dataset:
- [EuRoC MH_01_easy](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

## Usage

### Monocular VIO Example
```bash
cd build
./euroc_example ../config/euroc_mono.yaml /path/to/EuRoC/MH_01_easy/
```

## License

MIT License - see [LICENSE](LICENSE)



