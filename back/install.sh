#!/bin/sh
apt-get update
python3 -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir -r requirements.txt