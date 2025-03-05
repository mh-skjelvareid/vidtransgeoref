#!/bin/bash

# Check if correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.mp4 output.mp4"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Extract creation time
CREATION_TIME=$(ffprobe -v quiet -select_streams v:0 \
    -show_entries stream_tags=creation_time \
    -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")

# Transcode video
ffmpeg -i "$INPUT_FILE" \
    -vf "scale=640:-2" \
    -c:v libx264 \
    -preset slow \
    -crf 28 \
    -c:a aac \
    -b:a 128k \
    -movflags use_metadata_tags \
    -map_metadata 0 \
    -metadata creation_time="$CREATION_TIME" \
    "$OUTPUT_FILE"

# Show file sizes
echo "Original size: $(du -h "$INPUT_FILE" | cut -f1)"
echo "New size: $(du -h "$OUTPUT_FILE" | cut -f1)"
