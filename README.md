# Lightweight VIO

A lightweight visual odometry and visual-inertial odometry system supporting **stereo**, **monocular**, and **RGB-D** configurations.

⚠️ **Status**: Under active development (monocular development branch)

## Installation

```bash
./build.sh
```

For detailed installation instructions: 📋 **[Installation Guide](docs/Install.md)**

## Sample Data

Download sample TUM RGB-D dataset:
- [TUM Freiburg2 Desk](https://drive.google.com/file/d/1HRWnBq9kq-m4gkjVuLdKUHeaZcztacIs/view?usp=sharing)

## Usage

### EuRoC Example (Stereo/Mono VIO)
```bash
cd build
./euroc_example ../config/euroc_vio.yaml /path/to/euroc/MH_01_easy/
```

## License

MIT License - see [LICENSE](LICENSE)



