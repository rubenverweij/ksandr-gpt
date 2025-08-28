#!/bin/bash

# Set source and destination directories
SRC_DIR="/home/ubuntu/ksandr_files/"
DST_DIR="/home/ubuntu/ksandr_files_cleaned/"

# Ensure destination exists
mkdir -p "$DST_DIR"

# Function to clean a file in-place
clean_file() {
    local file="$1"

    # Convert to UTF-8 and Unix line endings, clean whitespace
    iconv -f UTF-8 -t UTF-8 "$file" 2>/dev/null | \
    tr -d '\r' | \
    sed -E 's/[[:space:]]+/ /g' | \
    sed -E 's/^[ \t]+|[ \t]+$//g' | \
    awk 'NF' > "${file}.cleaned"

    # Overwrite original file
    mv "${file}.cleaned" "$file"
}

# Loop through all files in the source directory
find "$SRC_DIR" -type f | while read -r src_file; do
    # Count words in the file
    word_count=$(wc -w < "$src_file")
    
    # Skip if word count is less than 50
    if [ "$word_count" -lt 50 ]; then
        continue
    fi

    # Get relative path and destination path
    rel_path="${src_file#$SRC_DIR/}"
    dst_file="$DST_DIR/$rel_path"
    dst_dir=$(dirname "$dst_file")

    # Create target directory if it doesn't exist
    mkdir -p "$dst_dir"

    # Copy the file
    cp "$src_file" "$dst_file"

    # Clean the copied file
    clean_file "$dst_file"
done

echo "Done. Files copied and cleaned in: $DST_DIR"
