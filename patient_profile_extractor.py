from __future__ import annotations

import time
import asyncio
import json

from openai import AsyncOpenAI

from patient_profile import PatientProfile


_SYSTEM_PROMPT = """
**You are the Patient-Profile Extractor.** Your task is to analyse patient-provided text and pull out demographic facts only.

# Task
Read the input text (usually one or more chat messages from the patient). Identify any demographic information (name, gender, birth date, phone, email, address, spoken languages, emergency contacts).

# JSON schema of the profile
```
{
  "_id": "<string identifier – generate or reuse if present>",
  "name": {
    "first_name": "<given name>",
    "last_name":  "<family name>",
    "full_name":  "<optional full string>"
  },
  "other_names": [ { …same shape as name… } ],
  "gender":        "male | female | other | unknown",
  "birth_date":    "YYYY-MM-DD",
  "phones":  [ { "value": "<digits>",  "use_for": "home|work|mobile" } ],
  "emails":  [ { "value": "<email>", "use_for": "home|work" } ],
  "faxes":   [ { "value": "<digits>",  "use_for": "home|work" } ],
  "addresses": [ { "line": ["street…"], "city": "", "state": "", "country": "" } ],
  "languages": [ { "value": "<ISO-639-1>", "preferred": true|false } ],
  "contacts": [
     {
       "relationship": ["mother" | "father" | …],
       "name": { … },
       "phones":  [...],
       "emails":  [...],
       "addresses": [...]
     }
  ]
}
```

# Output
Return **raw JSON (no markdown fences)** that matches the schema above (keys may be omitted if unknown). If the text contains **no** demographic facts, output `{}`.
"""


class PatientProfileExtractor:
    """Callable extractor object wrapping OpenAI responses.parse."""

    def __init__(
        self, client: AsyncOpenAI | None = None, system_prompt: str = _SYSTEM_PROMPT
    ):
        self.client = client or AsyncOpenAI()
        self.system_prompt = system_prompt

    async def extract_async(self, text: str, model: str = "gpt-4o-mini") -> PatientProfile:
        tic = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        )

        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            raise ValueError("Model did not return valid JSON:\n" + raw) from e

        print(f"[PatientProfileExtractor] elapsed: {time.perf_counter()-tic:.2f}s")
        return PatientProfile.model_validate(data)

    def extract(self, text: str, model: str = "gpt-4o-mini") -> PatientProfile:
        """Sync wrapper around `extract_async`."""

        return asyncio.run(self.extract_async(text, model=model))


# Default singleton for convenience
_DEFAULT_EXTRACTOR = PatientProfileExtractor()


def extract_patient_profile(text: str, model: str = "gpt-4o-mini") -> PatientProfile:
    """Convenience wrapper around a module-level `PatientProfileExtractor`."""

    return _DEFAULT_EXTRACTOR.extract(text, model=model)


if __name__ == "__main__":
    SAMPLE = (
        "Hi doctor, my name is Maria Elena García López but friends call me Mariel. "
        "I was born on 1980-10-08 and I’m female. Current address: 24 Rue de Rivoli, Paris, France. "
        "Mobile +33 6 12 34 56 78; work email maria.garcia@louvre.fr. "
        "I speak Spanish (preferred), French and English."
    )

    print(extract_patient_profile(SAMPLE))
