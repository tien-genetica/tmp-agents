from __future__ import annotations


import asyncio
import time

from openai import AsyncOpenAI, OpenAI

from medical_guideline import MedicalGuideline


_SYSTEM_PROMPT = """
**You are the Medical-Guideline Extractor.** Your job is to read text (e.g. news, papers, policy documents) and output ONLY structured guideline metadata.

# Task
Identify guideline information: id/title/category (international | vietnamese | other), source organisation, url, version, effective date, tags, language, lab test reference ranges.

# JSON schema
Return raw JSON matching the `MedicalGuideline` Pydantic model. Main keys:
```
{
  "id": "<string id>",
  "title": "<title>",
  "description": "<optional description>",
  "category": "international | vietnamese | other",
  "source": "<organisation>",
  "url": "<link>",
  "effectiveDate": "YYYY-MM-DD",
  "version": "<version string>",
  "tags": ["<tag>", ...],
  "language": "<ISO-639-1>",
  "labTests": [
     {
       "code": "<loinc-or-code>",
       "name": "<test name>",
       "internationalRanges": [ {"lower": n?, "upper": n?, "unit": "", "ageMin": n?, "ageMax": n?, "sex": "male|female"?} ],
       "vietnameseRanges":    [ ... ]
     }
  ]
}
```

# Output
• Provide raw JSON only (no markdown).  
• Omit fields you cannot infer.  
• If no guideline info is present, output `{}`.
"""


class MedicalGuidelineExtractor:
    """Async extractor using `AsyncOpenAI.responses.parse`."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self.client = client or AsyncOpenAI()
        self.system_prompt = system_prompt

    async def extract_async(
        self, text: str, model: str = "gpt-4o-mini"
    ) -> MedicalGuideline:
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

        print(f"[MedicalGuidelineExtractor] elapsed: {time.perf_counter()-tic:.2f}s")
        return MedicalGuideline.model_validate(data)

    def extract(self, text: str, model: str = "gpt-4o-mini") -> MedicalGuideline:
        """Sync helper around `extract_async`."""

        return asyncio.run(self.extract_async(text, model=model))


_DEFAULT_EXTRACTOR = MedicalGuidelineExtractor()


def extract_guideline(text: str, model: str = "gpt-4o-mini") -> MedicalGuideline:
    return _DEFAULT_EXTRACTOR.extract(text, model=model)


if __name__ == "__main__":
    SAMPLE = (
        "The World Health Organization published its 2024 Hypertension Management guideline on 2024-03-15. "
        "Document URL: https://www.who.int/hypertension2024 . Tags: hypertension, adult. Language: en.\n"
        "Lab reference: systolic blood pressure 90-120 mmHg (international)."
    )

    import json

    guideline = extract_guideline(SAMPLE)
    print(json.dumps(guideline.to_json(), indent=2))
