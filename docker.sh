#!/bin/bash

# Docker build and run script for lightweight VO/VIO

IMAGE_NAME="lightweight-vio"
TAG="latest"

echo "=============================================="
echo "  Lightweight VIO Docker Helper Script"
echo "=============================================="

# Function to build Docker image
build_image() {
    echo "Building Docker image: $IMAGE_NAME:$TAG"
    docker build -t $IMAGE_NAME:$TAG .
    
    if [ $? -eq 0 ]; then
        echo "Docker image built successfully!"
        echo "Image name: $IMAGE_NAME:$TAG"
    else
        echo "Failed to build Docker image"
        exit 1
    fi
}

# Function to run Docker container
run_app() {
    local app_type=$1
    local dataset_path=$2
    
    if [ -z "$app_type" ] || [ -z "$dataset_path" ]; then
        echo "Usage: $0 run <vo|vio> <dataset_path>"
        echo "Example (EuRoC): $0 run vio /path/to/euroc/MH_01_easy"
        echo "Example (TUM VI): $0 run vio /path/to/dataset-room1_512_16"
        exit 1
    fi
    
    if [ "$app_type" != "vo" ] && [ "$app_type" != "vio" ]; then
        echo "Error: Invalid application type '$app_type'. Must be 'vo' or 'vio'."
        exit 1
    fi

    if [ ! -d "$dataset_path" ]; then
        echo "Error: Dataset path does not exist: $dataset_path"
        exit 1
    fi
    
    echo "Running Docker container with app='$app_type' and dataset='$dataset_path'"

    # Allow Docker to connect to the host's X server for GUI forwarding
    echo "Temporarily allowing local connections to X server..."
    xhost +local:docker
    
    docker run --rm -it \
        -v "$dataset_path:/dataset:ro" \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --privileged \
        $IMAGE_NAME:$TAG $app_type /dataset

    # Revoke access after the container exits
    echo "Revoking local connections to X server."
    xhost -local:docker
}

# Function to run interactive shell in container
run_shell() {
    echo "Starting interactive shell in Docker container"
    docker run --rm -it \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --privileged \
        --entrypoint /bin/bash \
        $IMAGE_NAME:$TAG
}

# Main script logic
case "$1" in
    "build")
        build_image
        ;;
    "run")
        run_app "$2" "$3"
        ;;
    "shell")
        run_shell
        ;;
    "clean")
        echo "Removing Docker image: $IMAGE_NAME:$TAG"
        docker rmi $IMAGE_NAME:$TAG
        ;;
    *)
        echo "Usage: $0 {build|run|shell|clean}"
        echo ""
        echo "Commands:"
        echo "  build              Build the Docker image"
        echo "  run <vo|vio> <path>  Run the specified application with a dataset"
        echo "  shell              Start an interactive shell in the container"
        echo "  clean              Remove the Docker image"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 run vo /path/to/euroc/MH_01_easy"
        echo "  $0 run vio /path/to/euroc/MH_01_easy"
        echo "  $0 run vio /path/to/dataset-room1_512_16"
        echo "  $0 shell"
        echo "  $0 clean"
        exit 1
        ;;
esac
