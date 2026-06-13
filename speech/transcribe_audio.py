"""
Speech-to-text module for Cyclops Field Assistant.

This module wraps Faster-Whisper for transcribing technician voice queries.
The final system uses this transcription as the text input to the RAG module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from faster_whisper import WhisperModel


def load_whisper_model(
    model_path: str = "checkpoints/faster-whisper-medium",
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    """
    Load a Faster-Whisper model from a local checkpoint directory.
    """
    return WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
    )


def transcribe_audio(
    audio_path: str | Path,
    model: Optional[WhisperModel] = None,
    model_path: str = "checkpoints/faster-whisper-medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    beam_size: int = 5,
) -> Dict[str, Any]:
    """
    Transcribe an audio file using Faster-Whisper.

    Parameters
    ----------
    audio_path:
        Path to the input audio file, e.g. .m4a, .wav, .mp3.
    model:
        Optional pre-loaded WhisperModel. Reuse this in live systems.
    model_path:
        Local Faster-Whisper checkpoint directory.
    device:
        "cpu" or "cuda".
    compute_type:
        Faster-Whisper compute type, e.g. "int8", "float16", "float32".
    language:
        Optional language code. If None, Faster-Whisper performs automatic detection.
    beam_size:
        Decoding beam size.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model is None:
        model = load_whisper_model(
            model_path=model_path,
            device=device,
            compute_type=compute_type,
        )

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
    )

    segment_list = []
    text_parts = []

    for segment in segments:
        item = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        segment_list.append(item)
        text_parts.append(item["text"])

    transcript = " ".join(text_parts).strip()

    return {
        "audio_path": str(audio_path),
        "text": transcript,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "segments": segment_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with Faster-Whisper.")
    parser.add_argument("audio_path", type=str, help="Path to audio file.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/faster-whisper-medium",
        help="Local Faster-Whisper checkpoint path.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--compute-type", type=str, default="int8", help="int8, float16, float32, etc.")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code. Omit for automatic language detection.",
    )
    parser.add_argument("--beam-size", type=int, default=5)

    args = parser.parse_args()

    result = transcribe_audio(
        audio_path=args.audio_path,
        model_path=args.model_path,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
