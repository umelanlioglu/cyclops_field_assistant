#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root
cd "$(dirname "$0")/.."

# -----------------------------
# Default paths
# -----------------------------
GELEN_DIR="${GELEN_DIR:-data/gelen_json}"
GIDEN_DIR="${GIDEN_DIR:-data/giden_json}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-checkpoints/yolo26s_cr10smart_seg_final.pt}"
RAG_CHUNKS="${RAG_CHUNKS:-rag/data/cr10smart_manual_chunks_multilingual.json}"
WHISPER_MODEL="${WHISPER_MODEL:-checkpoints/faster-whisper-medium}"

WHISPER_DEVICE="${WHISPER_DEVICE:-cpu}"
WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-}"

GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite}"

# -----------------------------
# Basic checks
# -----------------------------
mkdir -p "$GELEN_DIR" "$GIDEN_DIR"

if [ ! -f "$YOLO_WEIGHTS" ]; then
  echo "ERROR: YOLO checkpoint not found:"
  echo "  $YOLO_WEIGHTS"
  exit 1
fi

if [ ! -f "$RAG_CHUNKS" ]; then
  echo "ERROR: RAG chunks file not found:"
  echo "  $RAG_CHUNKS"
  exit 1
fi

if [ ! -d "$WHISPER_MODEL" ]; then
  echo "Faster-Whisper checkpoint not found:"
  echo "  $WHISPER_MODEL"
  echo "Downloading it now..."
  ./scripts/download_faster_whisper.sh
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  echo "Set it with:"
  echo '  export GEMINI_API_KEY="your_gemini_api_key_here"'
  exit 1
fi

export GEMINI_MODEL="$GEMINI_MODEL"

echo "Starting Cyclops live AI pipeline..."
echo "Incoming directory: $GELEN_DIR"
echo "Outgoing directory: $GIDEN_DIR"
echo "YOLO checkpoint:    $YOLO_WEIGHTS"
echo "RAG chunks:         $RAG_CHUNKS"
echo "Whisper model:      $WHISPER_MODEL"
echo "Gemini model:       $GEMINI_MODEL"
echo

python live_worker/live_ai_worker.py \
  --rag-package-dir rag \
  --gelen-dir "$GELEN_DIR" \
  --giden-dir "$GIDEN_DIR" \
  --yolo-weights "$YOLO_WEIGHTS" \
  --manual-chunks "$RAG_CHUNKS" \
  --whisper-model "$WHISPER_MODEL" \
  --whisper-device "$WHISPER_DEVICE" \
  --whisper-compute-type "$WHISPER_COMPUTE_TYPE" \
  --whisper-language "$WHISPER_LANGUAGE" \
  --semantic-index \
  --latest-frame-only \
  --load-stt-at-startup \
  --use-gemini \
  --use-gemini-rerank \
  --use-gemini-answer \
  "$@"
