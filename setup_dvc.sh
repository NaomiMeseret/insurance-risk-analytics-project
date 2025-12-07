#!/bin/bash

# Setup script for DVC configuration
# This script initializes DVC and sets up local remote storage

echo "Setting up Data Version Control (DVC)..."

# Initialize DVC
echo "Initializing DVC..."
dvc init

# Create local storage directory
STORAGE_DIR="$HOME/dvc_storage"
echo "Creating local storage directory at $STORAGE_DIR..."
mkdir -p "$STORAGE_DIR"

# Add local storage as remote
echo "Adding local storage as DVC remote..."
dvc remote add -d localstorage "$STORAGE_DIR"

# Show DVC configuration
echo ""
echo "DVC configuration:"
dvc remote list

echo ""
echo "DVC setup complete!"
echo "To add data files, use: dvc add data/your_file.csv"
echo "To commit changes: git add .dvc data/.gitignore && git commit -m 'Add data files'"
echo "To push data: dvc push"

