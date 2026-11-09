#!/bin/sh
# Download the UCI Superconductivity Data (Hamidieh, 2018; CC BY 4.0) into data/raw.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/download data/raw
curl -sSL -o data/download/superconductivty_data.zip \
  "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
echo "87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e  data/download/superconductivty_data.zip" | shasum -a 256 -c -
unzip -o -q data/download/superconductivty_data.zip -d data/raw
