cat > README.md <<'EOF'
# Cyclops Field Assistant

This repository contains the RAG, computer vision, and speech-related AI modules of the **Cyclops Field Assistant** senior design project.

Cyclops Field Assistant is an AI-powered wearable service assistant designed for the Augmency Cyclops HMD Pro-G headset. The system provides scene-aware installation, service, and troubleshooting support for a Creality CR-10 Smart 3D printer by combining component segmentation, speech transcription, retrieval-augmented generation, and visual guidance.

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
├── text2speech/
│   ├── __init__.py
│   └── text_to_speech.py
└── scripts/
    └── download_faster_whisper.sh
```

## Computer Vision Checkpoint

The final YOLO26 Small instance-segmentation checkpoint is stored at:

```text
checkpoints/yolo26s_cr10smart_seg_final.pt
```

It is used for Creality CR-10 Smart printer-component segmentation.

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

Alternatively, you may create a local `.env` file.

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

EOF