from __future__ import annotations


from openai import OpenAI

from medical_guideline import MedicalGuideline

import time


_SYSTEM_PROMPT = "You are a medical guideline extractor. Output ONLY JSON that fits the MedicalGuideline Pydantic model."


class MedicalGuidelineExtractor:
    """Wrapper around OpenAI responses.parse for guidelines."""

    def __init__(
        self, client: OpenAI | None = None, system_prompt: str = _SYSTEM_PROMPT
    ):
        self.client = client or OpenAI()
        self.system_prompt = system_prompt

    def extract(self, text: str, model: str = "gpt-4o-mini") -> MedicalGuideline:
        start = time.perf_counter()
        resp = self.client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            text_format=MedicalGuideline,
        )

        print(f"[MedicalGuidelineExtractor] elapsed: {time.perf_counter()-start:.2f}s")
        return resp.output_parsed


_DEFAULT_EXTRACTOR = MedicalGuidelineExtractor()


def extract_guideline(text: str, model: str = "gpt-4o-mini") -> MedicalGuideline:
    """Convenience wrapper around a module-level `MedicalGuidelineExtractor`."""

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
