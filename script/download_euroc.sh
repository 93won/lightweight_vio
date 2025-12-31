#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Check for the destination directory argument.
if [ -z "$1" ]; then
  echo "Error: Please provide a destination directory."
  echo "Usage: $0 <path_to_destination_directory>"
  exit 1
fi

DEST_DIR="$1"

# 2. Create the directory and navigate into it.
EUROC_DIR="$DEST_DIR/EuroC"
mkdir -p "$EUROC_DIR"
echo "✅ All datasets will be downloaded and extracted into: ${EUROC_DIR}"
cd "$EUROC_DIR"

# Base URL for the datasets
BASE_URL="https://www.research-collection.ethz.ch/bitstreams"

# Corresponding download URLs for each dataset
DATASET_URL=(
    "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/download"
    "02ecda9a-298f-498b-970c-b7c44334d880/download"
    "ea12bc01-3677-4b4c-853d-87c7870b8c44/download"
)

DATASET_NAME=(
    "machine_hall"
    "vicon_room1"
    "vicon_room2"
)

SEQS_LISTS=(
    "MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult"
    "V1_01_easy V1_02_medium V1_03_difficult"
    "V2_01_easy V2_02_medium V2_03_difficult"
)

declare -A SEQS_BY_ZIP
for idx in "${!DATASET_NAME[@]}"; do
    if [ -n "${SEQS_LISTS[$idx]:-}" ]; then
        SEQS_BY_ZIP["${DATASET_NAME[$idx]}"]="${SEQS_LISTS[$idx]}"
    fi
done

# Loop through each dataset (map dataset name -> dataset path)
for i in "${!DATASET_URL[@]}"; do
    NAME="${DATASET_NAME[$i]}"
    DATASET_PATH="${DATASET_URL[$i]}"

    # Determine zip filename. Prefer the human-readable NAME when path uses '/download'.
    if [[ "${DATASET_PATH}" == */download ]]; then
        ZIP_FILE="${NAME}.zip"
    else
        ZIP_FILE=$(basename "${DATASET_PATH}")
    fi

    echo "=================================================="

    if [ -n "${SEQS_BY_ZIP[$NAME]}" ]; then
        all_present=1
        for seq in ${SEQS_BY_ZIP[$NAME]}; do
            if [ ! -d "${EUROC_DIR%/}/$seq" ]; then
                all_present=0
                break
            fi
        done
        if [ "$all_present" -eq 1 ]; then
            echo "✅ All ${NAME} sequences already exist in '${EUROC_DIR}'. Skipping download of ${NAME}."
            continue
        fi
    fi
    
    echo ">>>>> Processing ${ZIP_FILE}..."
    
    # 4. Download the file using wget.
    echo ">>>>> 1. Downloading..."
    if ! wget -q --show-progress -L --content-disposition -O "${ZIP_FILE}" "${BASE_URL}/${DATASET_PATH}"; then
        echo ">>>>> Error: Failed to download ${ZIP_FILE}. Skipping."
        [ -f "${ZIP_FILE}" ] && rm -f "${ZIP_FILE}"
        continue
    fi
    
    if [ -f "${ZIP_FILE}" ]; then
        # 6. Unzip into the EuroC directory (using the -d option).
        echo ">>>>> 2. Extracting into '${EUROC_DIR}'..."
        unzip -q "${ZIP_FILE}" -d "$EUROC_DIR"

        # move any sequence subdirectories up to the EuroC root so that
        # sequence folders (e.g. V2_01_easy) live directly under EuroC/.
        EXTRACT_DIR="$EUROC_DIR/${NAME}"
        if [ -d "$EXTRACT_DIR" ]; then
            shopt -s nullglob dotglob
            for entry in "$EXTRACT_DIR"/*; do
                if [ -d "$entry" ]; then
                    seqname=$(basename "$entry")
                    dest="$EUROC_DIR/$seqname"
                    if [ -e "$dest" ]; then
                        echo ">>>>> Destination '$dest' already exists; skipping move of '$entry'."
                    else
                        mv "$entry" "$EUROC_DIR/"
                    fi
                fi
            done
            # disable dotglob if available
            if command -v shopt >/dev/null 2>&1; then
                shopt -u dotglob
            fi
            # Remove the now-empty extraction directory if possible
            rm -r "$EXTRACT_DIR" 2>/dev/null || true
        fi

        # 7. Delete the original top-level zip file.
        echo ">>>>> 3. Deleting original zip file: ${ZIP_FILE}"
        rm "${ZIP_FILE}"

        # 8. For each sequence folder that belongs to this archive (if known),
        #    unzip any inner .zip files (typically named like the folder) and
        #    remove those zip files after extraction.
        echo ">>>>> 4. Extracting inner zip files..."
        if [ -n "${SEQS_BY_ZIP[$NAME]}" ]; then
            shopt -s nullglob
            for seq in ${SEQS_BY_ZIP[$NAME]}; do
                seqdir="$EUROC_DIR/$seq"
                if [ -d "$seqdir" ]; then
                    for z in "$seqdir"/*.zip; do
                        if [ -f "$z" ]; then
                            echo ">>>>> '$z'..."
                            unzip -q -o "$z" -d "$seqdir"
                            rm -f "$z"
                        fi
                    done
                else
                    echo ">>>>> Warning: expected sequence dir '$seqdir' not found."
                fi
            done
            # disable nullglob if available
            if command -v shopt >/dev/null 2>&1; then
                shopt -u nullglob
            fi
        else
            # If mapping unknown, attempt to detect any new sequence dirs and
            # extract inner zips inside them.
            shopt -s nullglob
            for seqdir in "$EUROC_DIR"/*/; do
                # normalize path (remove trailing slash)
                seqdir="${seqdir%/}"
                # consider only dirs whose names look like V* or MH_* etc; skip the top-level archive dir
                dirbase=$(basename "$seqdir")
                if [ "$dirbase" != "${NAME}" ]; then
                    for z in "$seqdir"/*.zip; do
                        if [ -f "$z" ]; then
                            echo ">>>>> '$z'..."
                            unzip -q -o "$z" -d "$seqdir"
                            rm -f "$z"
                        fi
                    done
                fi
            done
            # disable nullglob if available
            if command -v shopt >/dev/null 2>&1; then
                shopt -u nullglob
            fi
        fi

        echo ">>>>> Finished processing ${ZIP_FILE}! 👍"
    else
        echo ">>>>> Error: Failed to download ${ZIP_FILE} ❌"
    fi
    
    echo "" # Add a newline for readability
done

echo "🎉 All tasks have been completed in the '${DEST_DIR}' directory."