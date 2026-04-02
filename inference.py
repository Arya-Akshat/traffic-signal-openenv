from __future__ import annotations

import os
from typing import Any

import requests


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def _llm_action(step_index: int, state: dict[str, Any] | None = None) -> str:
    if not OPENAI_API_KEY:
        if state is None:
            return "KEEP"
        return "SWITCH" if step_index % 3 == 2 else "KEEP"

    try:
        from openai import OpenAI
    except Exception:
        return "KEEP"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    prompt = {
        "step": step_index,
        "state": state,
        "actions": ["KEEP", "SWITCH"],
    }
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "Return only KEEP or SWITCH for traffic signal control.",
            },
            {"role": "user", "content": str(prompt)},
        ],
    )
    text = response.output_text.strip().upper()
    return "SWITCH" if "SWITCH" in text else "KEEP"


def run() -> None:
    headers = _build_headers()
    state = requests.get(f"{BASE_URL}/reset", headers=headers, timeout=30).json()

    for step_index in range(10):
        action = _llm_action(step_index, state)
        response = requests.post(
            f"{BASE_URL}/step",
            json={"action": action},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        state = response.json()
        print(state)


if __name__ == "__main__":
    run()
