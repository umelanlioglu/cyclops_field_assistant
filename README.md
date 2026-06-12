# Cyclops Field Assistant — RAG and Computer Vision Backend

This repository contains the backend-side RAG and computer vision components of the **AI-Based Service and Installation Support** senior design project.

The system is designed for the Augmency Cyclops HMD Pro-G headset and provides AI-powered field assistance for service, installation, and troubleshooting tasks. It combines real-time printer-component segmentation, speech transcription, retrieval-augmented generation, and scene-aware visual guidance.

## Environment Setup

The backend was developed and tested inside the following Conda environment:

```bash
conda activate ar-rag-yolo
```

After activating the environment, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Repository Contents

This repository currently includes:

* `requirements.txt` — Python package dependencies for the backend environment
* `README.md` — setup and project description

Additional RAG, computer vision, and backend source files will be added as part of the final project submission.

## Notes

Model weights, API keys, environment files, datasets, and large generated files are not included in this repository. They should be provided separately or configured locally for security and storage reasons.
