#!/bin/sh
set -e

sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/partial/*
sudo apt-get update
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0
sudo apt-get clean
python -m pip install --upgrade pip
