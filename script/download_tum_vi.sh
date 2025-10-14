#!/bin/bash

# TUM VI Dataset Downloader
# Downloads TUM VI datasets in EuRoC/DSO format (512x512)
# Usage: ./download_tum_vi.sh [dataset_name] [download_directory]
# If no arguments provided, shows available datasets and prompts for selection

# set -e  # Exit on any error - disabled for download_all mode to continue on errors

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base URL for TUM VI dataset
BASE_URL="https://vision.in.tum.de/tumvi/exported/euroc/512_16"

# Available datasets
declare -a DATASETS=(
    "room1"
    "room2"
    "room3"
    "room4"
    "room5"
    "room6"
)

# Function to print usage
print_usage() {
    echo -e "${BLUE}TUM VI Dataset Downloader${NC}"
    echo -e "${BLUE}=========================${NC}"
    echo ""
    echo "Usage: $0 [dataset_name_or_directory] [download_directory]"
    echo ""
    echo "Examples:"
    echo "  $0                              # Interactive mode - shows all datasets"
    echo "  $0 corridor1                    # Download corridor1 to current directory"
    echo "  $0 /path/to/datasets            # Download ALL datasets to specified directory"
    echo "  $0 corridor1 /path/to/datasets  # Download corridor1 to specified directory"
    echo "  $0 all                          # Download ALL datasets (WARNING: ~150GB+)"
    echo ""
}

# Function to show available datasets
show_datasets() {
    echo -e "${YELLOW}Available TUM VI Datasets (512x512 EuRoC format):${NC}"
    echo -e "${YELLOW}================================================${NC}"
    echo ""
    
    echo "Corridor sequences:"
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == corridor* ]]; then
            echo "  - $dataset"
        fi
    done
    
    echo ""
    echo "Magistrale sequences:"
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == magistrale* ]]; then
            echo "  - $dataset"
        fi
    done
    
    echo ""
    echo "Outdoor sequences:"
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == outdoors* ]]; then
            echo "  - $dataset"
        fi
    done
    
    echo ""
    echo "Room sequences:"
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == room* ]]; then
            echo "  - $dataset"
        fi
    done
    
    echo ""
    echo "Slides sequences:"
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == slides* ]]; then
            echo "  - $dataset"
        fi
    done
    echo ""
}

# Function to check if dataset name is valid
is_valid_dataset() {
    local dataset_name=$1
    for dataset in "${DATASETS[@]}"; do
        if [[ "$dataset" == "$dataset_name" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to download a single dataset
download_dataset() {
    local dataset_name=$1
    local download_dir=$2
    local auto_mode=${3:-false}  # Third parameter for auto mode
    local url="${BASE_URL}/dataset-${dataset_name}_512_16.tar"
    local filename="dataset-${dataset_name}_512_16.tar"
    local full_path="${download_dir}/${filename}"
    local extracted_folder="${download_dir}/dataset-${dataset_name}_512_16"
    
    echo -e "${GREEN}Downloading TUM VI dataset: ${dataset_name}${NC}"
    echo -e "${BLUE}URL: ${url}${NC}"
    echo -e "${BLUE}Destination: ${full_path}${NC}"
    echo ""
    
    # Create download directory if it doesn't exist
    mkdir -p "$download_dir"
    
    # Check if extracted folder already exists
    if [[ -d "$extracted_folder" ]]; then
        echo -e "${YELLOW}Dataset already extracted: ${extracted_folder}${NC}"
        echo -e "${BLUE}Skipping download of ${dataset_name}${NC}"
        echo -e "${GREEN}✓ Dataset already available: ${dataset_name}${NC}"
        return 0
    fi
    
    # Check if file already exists
    if [[ -f "$full_path" ]]; then
        echo -e "${YELLOW}Tar file already exists: ${full_path}${NC}"
        if [[ "$auto_mode" == "false" ]]; then
            read -p "Do you want to re-download? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}Skipping download of ${dataset_name}${NC}"
                # Try to extract existing tar file
                echo -e "${GREEN}Extracting existing ${filename}...${NC}"
                cd "$download_dir"
                if tar -xf "$filename"; then
                    echo -e "${GREEN}✓ Successfully extracted: ${filename}${NC}"
                    if [[ "$auto_mode" == "true" ]]; then
                        rm "$filename"
                        echo -e "${GREEN}✓ Removed tar file: ${filename}${NC}"
                    fi
                else
                    echo -e "${RED}✗ Failed to extract: ${filename}${NC}"
                    return 1
                fi
                return 0
            fi
        else
            echo -e "${BLUE}Auto mode: Skipping download of ${dataset_name}${NC}"
            # Try to extract existing tar file
            echo -e "${GREEN}Auto mode: Extracting existing ${filename}...${NC}"
            cd "$download_dir"
            if tar -xf "$filename"; then
                echo -e "${GREEN}✓ Successfully extracted: ${filename}${NC}"
                rm "$filename"
                echo -e "${GREEN}✓ Removed tar file: ${filename}${NC}"
            else
                echo -e "${RED}✗ Failed to extract: ${filename}${NC}"
                return 1
            fi
            return 0
        fi
    fi
    
    # Download with wget
    echo -e "${GREEN}Starting download...${NC}"
    if wget --progress=bar:force:noscroll -O "$full_path" "$url"; then
        echo -e "${GREEN}✓ Successfully downloaded: ${filename}${NC}"
        
        # Auto extract in auto mode, otherwise ask
        if [[ "$auto_mode" == "true" ]]; then
            echo -e "${GREEN}Auto mode: Extracting ${filename}...${NC}"
            cd "$download_dir"
            if tar -xf "$filename"; then
                echo -e "${GREEN}✓ Successfully extracted: ${filename}${NC}"
                echo -e "${GREEN}Auto mode: Removing tar file to save space...${NC}"
                rm "$filename"
                echo -e "${GREEN}✓ Removed tar file: ${filename}${NC}"
            else
                echo -e "${RED}✗ Failed to extract: ${filename}${NC}"
                return 1
            fi
        else
            # Ask if user wants to extract
            read -p "Do you want to extract the dataset? (Y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                echo -e "${BLUE}Dataset downloaded but not extracted.${NC}"
            else
                echo -e "${GREEN}Extracting ${filename}...${NC}"
                cd "$download_dir"
                if tar -xf "$filename"; then
                    echo -e "${GREEN}✓ Successfully extracted: ${filename}${NC}"
                    
                    # Ask if user wants to remove the tar file
                    read -p "Do you want to remove the tar file to save space? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        rm "$filename"
                        echo -e "${GREEN}✓ Removed tar file: ${filename}${NC}"
                    fi
                else
                    echo -e "${RED}✗ Failed to extract: ${filename}${NC}"
                    return 1
                fi
            fi
        fi
    else
        echo -e "${RED}✗ Failed to download: ${filename}${NC}"
        return 1
    fi
    
    echo ""
}

# Function to download all datasets
download_all_datasets() {
    local download_dir=$1
    
    echo -e "${GREEN}Starting download of all TUM VI datasets...${NC}"
    echo -e "${YELLOW}Auto mode: Will download, extract, and remove tar files automatically.${NC}"
    echo ""
    
    local failed_downloads=()
    local successful_downloads=0
    local skipped_downloads=0
    local total_datasets=${#DATASETS[@]}
    
    for i in "${!DATASETS[@]}"; do
        local dataset="${DATASETS[$i]}"
        local current_num=$((i + 1))
        
        echo -e "${YELLOW}[$(date)] Processing dataset ${current_num}/${total_datasets}: ${dataset}${NC}"
        
        # Call download_dataset and capture its return code
        local extracted_folder="${download_dir}/dataset-${dataset}_512_16"
        local was_already_extracted=false
        
        # Check if already extracted before calling download_dataset
        if [[ -d "$extracted_folder" ]]; then
            was_already_extracted=true
        fi
        
        # Call download function and continue regardless of result
        download_dataset "$dataset" "$download_dir" "true" || true
        
        # Update counters based on result
        if [[ -d "$extracted_folder" ]]; then
            if [[ "$was_already_extracted" == "true" ]]; then
                ((skipped_downloads++))
                echo -e "${BLUE}⊘ Skipped (already exists): ${dataset}${NC}"
            else
                ((successful_downloads++))
                echo -e "${GREEN}✓ Successfully processed: ${dataset}${NC}"
            fi
        else
            failed_downloads+=("$dataset")
            echo -e "${RED}✗ Failed: ${dataset}${NC}"
        fi
        
        echo -e "${BLUE}Progress: ${successful_downloads} new, ${skipped_downloads} skipped, ${#failed_downloads[@]} failed${NC}"
        echo "----------------------------------------"
        echo ""
    done
    
    echo ""
    echo -e "${GREEN}=== FINAL SUMMARY ===${NC}"
    echo -e "${GREEN}New downloads: ${successful_downloads}${NC}"
    echo -e "${BLUE}Skipped (already existed): ${skipped_downloads}${NC}"
    echo -e "${RED}Failed downloads: ${#failed_downloads[@]}${NC}"
    echo -e "${YELLOW}Total processed: ${total_datasets}${NC}"
    
    if [[ ${#failed_downloads[@]} -gt 0 ]]; then
        echo -e "${RED}Failed datasets:${NC}"
        for failed in "${failed_downloads[@]}"; do
            echo "  - $failed"
        done
    fi
}

# Main script logic
main() {
    # Default download directory
    DOWNLOAD_DIR="."
    
    # Parse arguments
    if [[ $# -eq 0 ]]; then
        # Default to downloading all datasets
        echo -e "${GREEN}No arguments provided - downloading ALL TUM VI datasets by default${NC}"
        echo -e "${YELLOW}This will download ~150GB+ of data. Press Ctrl+C to cancel within 5 seconds...${NC}"
        sleep 5
        DATASET_NAME="all"
        DOWNLOAD_DIR="."
        
    elif [[ $# -eq 1 ]]; then
        # Check if argument is a path (contains / or starts with . or ~) or looks like a directory
        if [[ "$1" == *"/"* || "$1" == .* || "$1" == ~* || -d "$1" ]]; then
            # Treat as download directory - download all datasets
            DATASET_NAME="all"
            DOWNLOAD_DIR="$1"
        else
            # Treat as dataset name
            DATASET_NAME=$1
        fi
    elif [[ $# -eq 2 ]]; then
        DATASET_NAME=$1
        DOWNLOAD_DIR=$2
    else
        print_usage
        exit 1
    fi
    
    # Convert to absolute path
    DOWNLOAD_DIR=$(realpath "$DOWNLOAD_DIR")
    
    # Check if downloading all datasets
    if [[ "$DATASET_NAME" == "all" ]]; then
        download_all_datasets "$DOWNLOAD_DIR"
        exit 0
    fi
    
    # Validate dataset name
    if ! is_valid_dataset "$DATASET_NAME"; then
        echo -e "${RED}Error: Invalid dataset name '${DATASET_NAME}'${NC}"
        echo ""
        show_datasets
        exit 1
    fi
    
    # Download the dataset
    download_dataset "$DATASET_NAME" "$DOWNLOAD_DIR"
}

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo -e "${RED}Error: wget is not installed. Please install wget first.${NC}"
    echo "Ubuntu/Debian: sudo apt install wget"
    echo "CentOS/RHEL: sudo yum install wget"
    exit 1
fi

# Run main function
main "$@"
