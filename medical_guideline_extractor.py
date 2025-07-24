from __future__ import annotations


from openai import OpenAI

from medical_guideline import MedicalGuideline

client = OpenAI()

_SYSTEM_PROMPT = (
    "You are a clinical guideline extraction assistant.\n\n"
    "TASK: Read the user's text and produce a JSON object that can be parsed into the `MedicalGuideline` Pydantic model.\n\n"
    "INCLUDE only these keys: id, title, description?, category (international|vietnamese|other), source, url?, effectiveDate?, version?, tags?, language?, labTests?\n"
    "• For labTests include: code, name, internationalRanges, vietnameseRanges.\n"
    "• Each reference range object needs: lower?, upper?, unit, ageMin?, ageMax?, sex?.\n"
    "• Omit any field you cannot infer. Use ISO-8601 for dates (YYYY-MM-DD)."
)


def extract_guideline(text: str, model: str = "gpt-4o-mini") -> MedicalGuideline:
    """Return a `MedicalGuideline` parsed from *text*."""

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=MedicalGuideline,
    )

    return resp.output_parsed


if __name__ == "__main__":
    SAMPLE = (
        "The World Health Organization published its 2024 Hypertension Management guideline on 2024-03-15. "
        "Document URL: https://www.who.int/hypertension2024 . Tags: hypertension, adult. Language: en.\n"
        "Lab reference: systolic blood pressure 90-120 mmHg (international)."
    )

    guideline = extract_guideline(SAMPLE)
    import json, pprint

    print(json.dumps(guideline.to_json(), indent=2)) 