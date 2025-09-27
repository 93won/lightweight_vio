# Installation Guide

This guide provides detailed instructions for installing and building the Lightweight Stereo VIO system.

## Installation Methods

We provide two installation methods:
1. **Native Build (Ubuntu 22.04)** - Direct installation on your system
2. **Docker** - Containerized environment (Recommended)

---

## Method 1: Native Build (Ubuntu 22.04)

This method builds the project and all its dependencies directly on your system. It has been tested on Ubuntu 22.04.

### Prerequisites

- **Ubuntu 22.04 LTS**
- **C++17 Compiler** (g++)
- **CMake** (>= 3.10)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/93won/lightweight_vio.git
cd lightweight_vio
```

### Step 2: Run the Build Script

The provided `build.sh` script will automatically install system dependencies (like OpenCV, Eigen, etc.) and then build the third-party libraries (Ceres, Pangolin) and the main application.

```bash
chmod +x build.sh
./build.sh
```

---

## Method 2: Docker Installation (Recommended)

Using Docker is the recommended method as it provides a self-contained, consistent environment across different systems.

### Prerequisites

- **Docker** installed on your system
- **X11 forwarding** support (for visualization)

### Step 1: Install Docker

If Docker is not installed on your system:

#### Ubuntu/Debian:
```bash
# Install Docker
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

#### Other Linux distributions:
Follow the official Docker installation guide for your distribution.

### Step 2: Clone the Repository

```bash
git clone https://github.com/93won/lightweight_vio.git
cd lightweight_vio
```

### Step 3: Build the Docker Image

```bash
chmod +x docker.sh
./docker.sh build
```

This command builds a Docker image named `lightweight-vio:latest` with all necessary dependencies and source code.

### Step 4: Verify Docker Build

Check that the image was created successfully:

```bash
docker images | grep lightweight-vio
```
