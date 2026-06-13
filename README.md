# Cyclops Field Assistant

This repository contains the RAG, computer vision, speech, text-to-speech, and live worker modules of the **Cyclops Field Assistant** senior design project.

Cyclops Field Assistant is an AI-powered wearable service assistant designed for the Augmency Cyclops HMD Pro-G headset. The system provides scene-aware installation, service, and troubleshooting support for a Creality CR-10 Smart 3D printer by combining component segmentation, speech transcription, retrieval-augmented generation, visual guidance, and spoken feedback.

## Environment

The project was developed inside the following Conda environment:

```bash
conda activate ar-rag-yolo
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
cyclops_field_assistant/
├── README.md
├── requirements.txt
├── checkpoints/
│   └── yolo26s_cr10smart_seg_final.pt
├── rag/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── expectations.py
│   ├── gemini_client.py
│   ├── labels.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── references.py
│   ├── request_guard.py
│   ├── retrieval.py
│   ├── routing_fallback.py
│   ├── session_memory.py
│   ├── source_scope.py
│   ├── turn_state.py
│   ├── vision.py
│   ├── visual_guidance.py
│   └── data/
│       └── cr10smart_manual_chunks_multilingual.json
├── segmentation/
│   ├── __init__.py
│   ├── detection_stabilizer.py
│   ├── labels.py
│   └── yolo_live.py
├── live_worker/
│   ├── README.md
│   ├── live_ai_worker.py
│   ├── demo_threaded_gemini_captions.py
│   ├── live_yolo_preview.py
│   ├── measure_live_latency.py
│   ├── simulate_gelen_stream.py
│   └── archive_conversations.py
├── speech/
│   ├── __init__.py
│   └── transcribe_audio.py
└── scripts/
    └── download_faster_whisper.sh
```

## Computer Vision Checkpoint

The final YOLO26 Small instance-segmentation checkpoint is stored at:

```text
checkpoints/yolo26s_cr10smart_seg_final.pt
```

It is used for Creality CR-10 Smart printer-component segmentation. The checkpoint was selected from the final Demo Day training run and is used as the default visual perception model.

## Faster-Whisper Checkpoint

The Faster-Whisper checkpoint is not committed directly because it is large. Download it locally with:

```bash
./scripts/download_faster_whisper.sh
```

This downloads the multilingual medium checkpoint to:

```text
checkpoints/faster-whisper-medium
```

The speech module can load it with:

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "checkpoints/faster-whisper-medium",
    device="cpu",
    compute_type="int8"
)
```

## Gemini Configuration

Gemini API keys are not stored in this repository. Each user should provide their own key locally.

Example:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL="gemini-3.1-flash-lite"
```

Alternatively, users may create a local `.env` file. Local `.env` files should not be committed.

## RAG Module

The RAG module uses structured CR-10 Smart task chunks together with Gemini-based answer generation. The final multilingual chunk database is stored at:

```text
rag/data/cr10smart_manual_chunks_multilingual.json
```

The module performs:

- request-scope filtering for greetings, miscellaneous, and out-of-scope questions
- hybrid retrieval over structured printer task chunks
- visual expectation checking using detected printer components
- grounded answer generation with Gemini
- visual guidance target selection for segmentation-based annotations

## Segmentation Module

The `segmentation/` folder contains the YOLO-based computer vision module.

Main files:

- `yolo_live.py` — loads the final YOLO26 segmentation checkpoint and runs inference on camera frames
- `detection_stabilizer.py` — stabilizes detections across live frames
- `labels.py` — contains printer-component class labels and label normalization utilities

The module uses the final checkpoint:

```text
checkpoints/yolo26s_cr10smart_seg_final.pt

## Speech Module

The `speech/` folder contains the Faster-Whisper speech-to-text wrapper used to transcribe technician voice queries.

The Faster-Whisper checkpoint is downloaded locally with:

```bash
./scripts/download_faster_whisper.sh

## Live Worker

The `live_worker/` folder contains the file-bridge worker and demo utilities used to connect incoming headset data with the AI pipeline.

The live worker reads camera/audio inputs, runs segmentation, speech transcription, RAG reasoning, and visual annotation, then writes response files back to the outgoing directory.

Expected local files:

```text
checkpoints/yolo26s_cr10smart_seg_final.pt
checkpoints/faster-whisper-medium/
rag/data/cr10smart_manual_chunks_multilingual.json
```

Example run:

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

## Notes

Large generated outputs, datasets, downloaded speech checkpoints, temporary files, and local secrets are intentionally excluded from the repository.
