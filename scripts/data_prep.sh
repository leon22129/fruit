#!/bin/bash

# Check if Kaggle API credentials are set
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
  echo "Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set."
  echo "Please visit https://www.kaggle.com/<your-username>/account and create an API token."
  echo "Then set them: export KAGGLE_USERNAME='your_username' && export KAGGLE_KEY='your_key'"
  exit 1
fi

# Create .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle/

# Create kaggle.json with credentials
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json

# Set permissions for kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "Kaggle API credentials configured."

# Check if the dataset already exists
if [ -d "dataset" ]; then
  echo "Dataset 'dataset' already exists. Skipping download."
else
  echo "Downloading Kaggle Fruits dataset..."
  kaggle datasets download -d moltean/fruits
  echo "Unzipping dataset..."
  unzip fruits.zip -d dataset
  rm fruits.zip
  echo "Dataset downloaded and unzipped successfully into 'dataset'."
fi
