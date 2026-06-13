#!/usr/bin/env bash
set -euo pipefail

# Go to repository root, no matter where script is called from
cd "$(dirname "$0")/.."

mkdir -p checkpoints/faster-whisper-medium

hf download Systran/faster-whisper-medium \
  --repo-type model \
  --local-dir checkpoints/faster-whisper-medium

echo "Downloaded Faster-Whisper medium checkpoint to checkpoints/faster-whisper-medium"
