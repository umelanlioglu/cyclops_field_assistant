"""Tiny Gemini wrapper for structured JSON answers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def get_gemini_client(env_path: str | None = None) -> genai.Client:
    if env_path:
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Put it in your .env file.")

    return genai.Client(api_key=api_key)


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)
    return json.loads(text)


def generate_json_answer(prompt: str, model: str | None = None) -> Dict[str, Any]:
    client = get_gemini_client()
    model = model or get_gemini_model()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    return _safe_json_loads(response.text)

def generate_text_answer(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.4,
) -> str:
    client = get_gemini_client()
    model = model or get_gemini_model()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    return (response.text or "").strip()

def test_gemini() -> str:
    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_gemini_model(),
        contents="Say hello in one short sentence.",
    )
    return response.text
