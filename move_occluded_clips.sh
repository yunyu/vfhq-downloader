#!/bin/bash

# Path to the text file containing file paths
FILE_LIST="data/occluded_video_paths.txt"

# Directory to move the files to. Ensure this directory exists or the script will fail.
MOVED_DIR="data/occluded_videos"

mkdir -p $MOVED_DIR

# Read the text file line by line
while IFS= read -r file_path || [ -n "$file_path" ]; do
    # Check if the file exists
    if [ ! -f "$file_path" ]; then
        echo "File does not exist or is not a regular file: $file_path"
        continue # Skip to the next file
    fi

    # Move the file to the specified directory
    mv "$file_path" "$MOVED_DIR"
    if [ $? -eq 0 ]; then
        echo "Moved: $file_path"
    else
        echo "Failed to move: $file_path"
    fi
done < "$FILE_LIST"
