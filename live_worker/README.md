# Live Worker

This folder contains the file-bridge live worker used to connect incoming headset data with the AI pipeline.

The worker watches incoming camera/audio files, runs segmentation, speech transcription, RAG reasoning, and visual annotation, then writes response files back to the outgoing directory.

## Main Files

- `live_ai_worker.py` — final live worker used for the integrated Gemini + RAG + YOLO pipeline
- `demo_threaded_gemini_captions.py` — threaded demo interface with Gemini-generated visual captions
- `live_yolo_preview.py` — lightweight YOLO preview utility
- `measure_live_latency.py` — latency measurement utility
- `simulate_gelen_stream.py` — utility for simulating incoming stream files
- `archive_conversations.py` — utility for archiving old live worker conversations

## Required Local Files

The following files/folders are expected locally:

```text
checkpoints/yolo26s_cr10smart_seg_final.pt
checkpoints/faster-whisper-medium/
rag/data/cr10smart_manual_chunks_multilingual.json
```

The Faster-Whisper checkpoint can be downloaded with:

```bash
./scripts/download_faster_whisper.sh
```

## Gemini

Set your own Gemini API key locally before running:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL="gemini-3.1-flash-lite"
```

Do not commit API keys or `.env` files.

## Example Run

From the repository root:

```bash
python live_worker/live_ai_worker.py \
  --rag-package-dir rag \
  --gelen-dir data/gelen_json \
  --giden-dir data/giden_json \
  --yolo-weights checkpoints/yolo26s_cr10smart_seg_final.pt \
  --manual-chunks rag/data/cr10smart_manual_chunks_multilingual.json \
  --whisper-model checkpoints/faster-whisper-medium \
  --whisper-device cpu \
  --whisper-compute-type int8 \
  --whisper-language "" \
  --semantic-index \
  --latest-frame-only \
  --load-stt-at-startup \
  --use-gemini \
  --use-gemini-rerank \
  --use-gemini-answer
```

Passing `--whisper-language ""` enables Faster-Whisper automatic language detection.

## Outputs

The worker reads from:

```text
data/gelen_json/<conversation_id>/
```

and writes responses to:

```text
data/giden_json/<conversation_id>/
```

Generated runtime outputs are local artifacts and should not be committed.
