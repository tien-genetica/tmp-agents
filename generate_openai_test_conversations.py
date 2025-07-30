#!/usr/bin/env python3
"""Generate 100 conversation test cases using the OpenAI API.

The script asks the model to write short conversations between a user and a
medical-AI assistant that *embed* the demographic and contact details needed
by :pyfunc:`patient_profile_extractor.extract_patient_profile` – name, gender,
birth date (ISO 8601 string), marital status, language, city/country,
and at least one phone number.

The 100 conversations are saved to ``tests/conversations_openai.jsonl``
(one JSON object per line) with the schema: ``{"id": str, "conversation": str}``.

Prerequisites:
    • ``OPENAI_API_KEY`` environment variable must be set.
    • ``pip install openai>=1.14`` (already listed in *requirements.txt*).

Usage::

    python generate_openai_test_conversations.py  # takes a few minutes / $$
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Global settings -----------------------------------------------------------
# ---------------------------------------------------------------------------
COUNT = 100  # total number of conversations
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CONCURRENCY = int(os.getenv("OPENAI_CONCURRENCY", "5"))  # parallel requests
OUTPUT_PATH = Path("tests/conversations_openai.jsonl")

SYSTEM_PROMPT = (
    "You are a data generator that writes realistic but *synthetic* chat "
    "transcripts. The transcript should look like a natural conversation "
    "between a patient (label lines with 'User:') and a medical AI assistant "
    "(label lines with 'Medical AI:'). Provide 5–10 alternating exchanges.\n"  # noqa: E501
    "\nEach transcript MUST embed ALL of the following patient details in "
    "natural language (not as separate JSON):\n"
    "  • Full name (first & last)\n"
    "  • Gender (male, female, other or unknown)\n"
    "  • Birth date in ISO 8601 format YYYY-MM-DD\n"
    "  • Marital status (single, married, divorced, widowed, separated)\n"
    "  • Preferred language\n"
    "  • City and country of residence\n"
    "  • At least one phone number (any format)\n"
    "\nRules:\n"
    "  • Do NOT wrap the conversation in markdown or code fences.\n"
    "  • Keep it short – no extra explanations before/after.\n"
    "  • The first line should *always* start with 'User:' and mention name, "
    "gender and birth date in parenthesis like: "
    "   User: Hi, I'm John Doe (male, born 1990-05-15). ...\n"
)

USER_PROMPT = "Please generate the conversation as specified."

# Regex to strip unintended code fences (robustness)
_RE_FENCE_OPEN = re.compile(r"^```(?:\w+)?\s*", flags=re.IGNORECASE)
_RE_FENCE_CLOSE = re.compile(r"\s*```$")


# ---------------------------------------------------------------------------
# Async generation helpers --------------------------------------------------
# ---------------------------------------------------------------------------

async def _generate_one(idx: int, client: AsyncOpenAI, sem: asyncio.Semaphore) -> Dict[str, str]:
    """Generate a single conversation (async)."""
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                    ],
                    seed=idx,  # deterministic-ish per idx
                    temperature=1.0,
                )
                content = resp.choices[0].message.content.strip()
                # Strip code fences if the model ignored the rule.
                content = _RE_FENCE_OPEN.sub("", content)
                content = _RE_FENCE_CLOSE.sub("", content)
                return {"id": f"openai-{idx:03d}", "conversation": content}
            except OpenAIError as exc:
                # Simple exponential backoff on failures (rate limits, etc.)
                wait = 2 ** attempt
                print(f"[warn] ({idx}) OpenAIError {exc}; retrying in {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Failed to generate conversation {idx} after retries")


async def _main_async() -> None:
    tic_total = time.perf_counter()

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [_generate_one(i, client, sem) for i in range(COUNT)]
    records: List[Dict[str, str]] = await asyncio.gather(*tasks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    dur = time.perf_counter() - tic_total
    print(f"Wrote {len(records)} conversations to {OUTPUT_PATH} in {dur:.1f}s")


# ---------------------------------------------------------------------------
# CLI entry-point -----------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:  # pragma: no cover – user abort
        print("Interrupted – exiting", file=sys.stderr) 