from __future__ import annotations


from openai import OpenAI

from patient_profile import PatientProfile


client = OpenAI()


_SYSTEM_PROMPT = """You are an information-extraction assistant. Extract only patient DEMOGRAPHICS from the user text and output JSON that can be parsed into the provided Pydantic model."""


def extract_patient_profile(text: str, model: str = "gpt-4o-mini") -> PatientProfile:
    """Return `PatientProfile` parsed directly via OpenAI responses.parse."""

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=PatientProfile,
    )

    return response.output_parsed


if __name__ == "__main__":
    SAMPLE = (
        "Hi doctor, my name is Maria Elena García López but friends call me Mariel. "
        "I was born on 1980-10-08 and I’m female. Current address: 24 Rue de Rivoli, Paris, France. "
        "Mobile +33 6 12 34 56 78; work email maria.garcia@louvre.fr. "
        "I speak Spanish (preferred), French and English."
    )

    profile = extract_patient_profile(SAMPLE)
    print(profile.to_fhir())
