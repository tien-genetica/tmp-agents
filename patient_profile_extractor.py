from __future__ import annotations

import time
import asyncio

from openai import AsyncOpenAI, OpenAI

from patient_profile import PatientProfile


_SYSTEM_PROMPT = (
    "You are a patient-profile extractor.\n\n"
    "OUTPUT: raw JSON (no markdown) parsable by the `PatientProfile` Pydantic model.\n"
    "• Only include fields you can confidently identify.\n"
    "• If the text contains NO demographic facts, return an empty JSON object `{}`."
)


class PatientProfileExtractor:
    """Callable extractor object wrapping OpenAI responses.parse."""

    def __init__(
        self, client: AsyncOpenAI | None = None, system_prompt: str = _SYSTEM_PROMPT
    ):
        self.client = client or AsyncOpenAI()
        self.system_prompt = system_prompt

    async def extract_async(
        self, text: str, model: str = "gpt-4o-mini"
    ) -> PatientProfile:
        start = time.perf_counter()
        resp = await self.client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            text_format=PatientProfile,
        )

        print(f"[PatientProfileExtractor] elapsed: {time.perf_counter() - start:.2f}s")
        return resp.output_parsed

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
