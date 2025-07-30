from __future__ import annotations

import time
import asyncio
import json
import time
import re
from typing import Any, Dict, List, MutableMapping, Optional

from openai import AsyncOpenAI

from patient_profile import PatientProfile

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASIC_INFO_SYSTEM_PROMPT = """
You are an AI specialized in extracting demographic information specifically about the patient from input text, such as conversations, profiles, or other text data. The text may mention multiple individuals, but you must focus only on the patient's information. Your task is to analyze the text and populate the following JSON schema with the patient's extracted information:

```json
{
  "name": {
    "first_name": "",
    "last_name": "",
    "full_name": ""
  },
  "other_names": [
    {
      "first_name": "",
      "last_name": "",
      "full_name": ""
    }
  ],
  "gender": "",
  "birth_date": "",
  "marital_status": "",
  "languages": [
    {
      "value": "",
      "preferred": false
    }
  ],
  "deceased": {
    "status": false,
    "date": ""
  }
}
```

Instructions:

### Field Extraction Rules
1. **Name**: Extract the exactly patient's primary name. Populate `first_name`, `last_name`, and `full_name`.
2. **Other Names**: Extract any aliases or additional names for the patient, populating `other_names` with the same structure as `name`.
3. **Gender**: Extract the patient's gender, using "male", "female", "other", or "unknown". Default to "unknown" if not specified.
4. **Birth Date**: Extract the patient's birth date, aiming for ISO 8601 format (YYYY-MM-DD). If only partial information is provided (e.g., year or year and month), populate with as much detail as available (e.g., "YYYY" or "YYYY-MM"). If only age is provided, estimate birth year (current year is 2025) and use "YYYY-01-01". Leave empty if no birth date or age is determinable.
5. **Marital Status**: Extract the patient's marital status, using "single", "married", "divorced", "widowed", "separated", or "unknown". Default to "unknown" if not specified.
6. **Languages**: Extract the patient's spoken languages, populating `value` (e.g., "English", "Spanish") and `preferred` (true only if explicitly stated as preferred, else false).
7. **Deceased**: Determine if the patient is deceased. Set `status` to true and extract `date` (YYYY-MM-DD) if provided; otherwise, set `status` to false and leave `date` empty.

### Key Considerations
1. **Patient Focus**: Extract information only about the patient, ignoring details about other individuals (e.g., family members, doctors) unless clearly tied to the patient.
2. **Missing Information**: For missing or unclear information, use empty strings, null, or empty arrays/objects as per the schema.
3. **Output Format**: Return the result as a JSON object adhering to the provided schema.
"""

CONTACT_INFO_SYSTEM_PROMPT = """
You are an AI specialized in extracting contact information specifically about the patient from input text, such as conversations, profiles, or other text data. The text may mention multiple individuals, but you must focus only on the patient's contact information. Your task is to analyze the text and populate the following JSON schema with the patient's extracted information:

```json
{
  "phones": [
    {
      "value": "",
      "use_for": ""
    }
  ],
  "emails": [
    {
      "value": "",
      "use_for": ""
    }
  ],
  "faxes": [
    {
      "value": "",
      "use_for": ""
    }
  ],
  "addresses": [
    {
      "line": [],
      "city": "",
      "state": "",
      "postal_code": "",
      "country": ""
    }
  ]
}
```

Instructions:

### Field Extraction Rules
1. **Phones**: Extract the patient's phone numbers. Populate `value` with the phone number and `use_for` as one of: "home", "work", "mobile", or "other". Default to "mobile" if the type is not specified.
2. **Emails**: Extract the patient's email addresses. Populate `value` with the email address and `use_for` as one of: "home", "work", or "other". Default to "other" if the type is not specified.
3. **Faxes**: Extract the patient's fax numbers. Populate `value` with the fax number and `use_for` as one of: "home", "work", or "other". Default to "other" if the type is not specified.
4. **Addresses**: Extract the patient's addresses. Populate `line` (street address as a list of strings, combining multiple lines if necessary), `city`, `state`, `postal_code`, and `country`. Leave fields empty if specific components are not provided.

### Key Considerations
1. **Patient Focus**: Extract contact information only about the patient, ignoring details about other individuals (e.g., family members, doctors) unless clearly tied to the patient.
2. **Missing Information**: For missing or unclear information, use empty strings, null, or empty arrays/objects as per the schema.
3. **Output Format**: Return the result as a JSON object adhering to the provided schema.
"""


RELATIONSHIPS_SYSTEM_PROMPT = """
You are an AI specialized in extracting information about a patient’s contacts (e.g., family members, emergency contacts) from input text, such as conversations, profiles, or other text data. The text may mention multiple individuals, but you must focus only on the patient’s contacts, excluding the patient or unrelated individuals. Your task is to analyze the text and populate the following JSON schema with the extracted information for the patient’s contacts:

```json
{
  "contacts": [
    {
      "relationship": [],
      "name": {
        "first_name": "",
        "last_name": "",
        "full_name": ""
      },
      "phones": [
        {
          "value": "",
          "use_for": ""
        }
      ],
      "emails": [
        {
          "value": "",
          "use_for": ""
        }
      ],
      "faxes": [
        {
          "value": "",
          "use_for": ""
        }
      ],
      "addresses": [
        {
          "line": [],
          "city": "",
          "state": "",
          "postal_code": "",
          "country": ""
        }
      ],
      "gender": "",
      "organizations": [
        {
          "reference": "",
          "display": ""
        }
      ]
    }
  ]
}
```

Instructions:

### Field Extraction Rules
1. **Relationship**: Extract the relationship(s) of each contact to the patient (e.g., "wife", "husband", "child") and populate `relationship` as a list of strings.
2. **Name**: Extract exactly each contact’s name. Populate `first_name`, `last_name`, and `full_name`.
3. **Phones**: Extract each contact’s phone numbers. Populate `value` with the phone number and `use_for` as one of: "home", "work", "mobile", or "other". Default to "mobile" if the type is not specified.
4. **Emails**: Extract each contact’s email addresses. Populate `value` with the email address and `use_for` as one of: "home", "work", or "other". Default to "other" if the type is not specified.
5. **Faxes**: Extract each contact’s fax numbers. Populate `value` with the fax number and `use_for` as one of: "home", "work", or "other". Default to "other" if the type is not specified.
6. **Addresses**: Extract each contact’s addresses. Populate `line` (street address as a list of strings, combining multiple lines if necessary), `city`, `state`, `postal_code`, and `country`. Leave fields empty if specific components are not provided.
7. **Gender**: Extract each contact’s gender, using "male", "female", "other", or "unknown". Default to "unknown" if not specified.
8. **Organizations**: Extract any organizations associated with each contact (e.g., workplace, hospital). Populate `reference` (e.g., an identifier or URL) and `display` (e.g., organization name) for each organization.

### Key Considerations
1. **Contact Focus**: Extract information only about the patient’s contacts (e.g., family members, emergency contacts)
2. **Missing Information**: For missing or unclear information, use empty strings, null, or empty arrays/objects as per the schema.
3. **Output Format**: Return the result as a JSON object adhering to the provided schema.
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _JsonExtractor:
    """Low-level wrapper that calls the OpenAI ChatCompletion API and parses JSON."""

    def __init__(self, system_prompt: str, client: Optional[AsyncOpenAI] = None):
        self._prompt = system_prompt
        self._client: AsyncOpenAI = client or AsyncOpenAI()

    async def extract_async(
        self, text: str, model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """Return a *partial* patient profile section (dict)."""
        tic = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": text},
            ],
        )

        raw_content: str = resp.choices[0].message.content.strip()
        # Handle common LLM habit of wrapping JSON inside markdown fences.
        cleaned = raw_content
        if cleaned.startswith("```"):
            # Remove opening ``` or ```json line
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            # Remove trailing fence
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data: Dict[str, Any] = json.loads(cleaned) if cleaned else {}
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(
                "Model did not return valid JSON:\n" + raw_content
            ) from exc

        dur = time.perf_counter() - tic
        return data

    def extract(self, text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Synchronous wrapper around :meth:`extract_async`."""
        return asyncio.run(self.extract_async(text, model=model))


# Concrete extractor singletons ------------------------------------------------
_BASIC_INFO_EXTRACTOR = _JsonExtractor(BASIC_INFO_SYSTEM_PROMPT)
_CONTACT_INFO_EXTRACTOR = _JsonExtractor(CONTACT_INFO_SYSTEM_PROMPT)
_RELATIONSHIPS_EXTRACTOR = _JsonExtractor(RELATIONSHIPS_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Public helper functions – each returns *partial* JSON dictionaries.
# ---------------------------------------------------------------------------


def extract_basic_info(text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Extract basic demographic information only (name, gender, birth date, …)."""
    return _BASIC_INFO_EXTRACTOR.extract(text, model=model)


def extract_contact_info(text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Extract phones, emails, faxes and addresses only."""
    return _CONTACT_INFO_EXTRACTOR.extract(text, model=model)


def extract_relationships(text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Extract family / emergency contact information only."""
    return _RELATIONSHIPS_EXTRACTOR.extract(text, model=model)


# ---------------------------------------------------------------------------
# Merging utilities
# ---------------------------------------------------------------------------


def _deep_merge(dest: MutableMapping[str, Any], src: MutableMapping[str, Any]) -> None:
    """Recursively merge *src* into *dest* (in-place).

    Lists are concatenated, dictionaries are merged recursively, and scalar
    values in *dest* are overwritten *only if* they are falsy ("empty").
    """
    for key, src_val in src.items():
        if src_val in (None, "", [], {}):
            # Skip empty values coming from the model.
            continue

        if key not in dest or dest[key] in (None, "", [], {}):
            dest[key] = src_val
            continue

        dest_val = dest[key]

        # Merge lists → concatenate unique elements preserving order.
        if isinstance(dest_val, list) and isinstance(src_val, list):
            dest_val.extend(x for x in src_val if x not in dest_val)
        # Merge dicts → recurse.
        elif isinstance(dest_val, dict) and isinstance(src_val, dict):
            _deep_merge(dest_val, src_val)
        else:
            # Conflict – prefer existing *dest* value and ignore *src*.
            continue


# ---------------------------------------------------------------------------
# Utility: sanitize merged dict ------------------------------------------------
# ---------------------------------------------------------------------------


def _sanitize(obj: Any) -> Any:  # type: ignore[return-any]
    """Recursively replace empty strings with ``None`` and prune empty containers.

    This ensures the payload conforms to the `PatientProfile` schema where many
    fields are optional (accepting ``None``) but *not* empty strings.
    """
    if isinstance(obj, dict):
        new_dict: Dict[str, Any] = {}
        for k, v in obj.items():
            cleaned = _sanitize(v)
            if cleaned in ("", [], {}, None):
                # Keep booleans and numeric zeroes; drop other empties.
                if isinstance(cleaned, bool) or cleaned == 0:
                    new_dict[k] = cleaned
                elif cleaned is None:
                    # Only keep key if schema requires it explicitly (e.g. status)
                    if k == "status":
                        new_dict[k] = cleaned
                    # else skip
                else:
                    # skip empty string/list/dict
                    continue
            else:
                new_dict[k] = cleaned
        return new_dict
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj if i not in ("", [], {}, None)]
    if obj == "":
        return None
    return obj


# ---------------------------------------------------------------------------
# High-level convenience – build a full PatientProfile by combining all prompts.
# ---------------------------------------------------------------------------


async def _extract_patient_profile_async(
    text: str, model: str = "gpt-4o-mini"
) -> Optional[PatientProfile]:
    # Gather all three extractions concurrently.
    basic_task = _BASIC_INFO_EXTRACTOR.extract_async(text, model=model)
    contact_task = _CONTACT_INFO_EXTRACTOR.extract_async(text, model=model)
    rel_task = _RELATIONSHIPS_EXTRACTOR.extract_async(text, model=model)

    basic, contact, relationships = await asyncio.gather(
        basic_task, contact_task, rel_task
    )

    # Merge sections into a single dict.
    merged: Dict[str, Any] = {}
    for section in (basic, contact, relationships):
        _deep_merge(merged, section)

    # Ensure mandatory fields exist so validation passes.
    merged.setdefault("_id", "")
    if "name" not in merged:
        merged["name"] = {"first_name": None, "last_name": None, "full_name": None}

    # Sanitize empty strings/collections before validation
    merged = _sanitize(merged)

    # Drop invalid language entries lacking a 'value'.
    if "languages" in merged:
        merged["languages"] = [
            lang for lang in merged["languages"] if lang.get("value")
        ]
        if not merged["languages"]:
            del merged["languages"]

    # Prune telecom lists with missing 'value'.
    for tel_key in ("phones", "emails", "faxes"):
        if tel_key in merged:
            merged[tel_key] = [t for t in merged[tel_key] if t.get("value")]
            if not merged[tel_key]:
                del merged[tel_key]

    # Prune empty addresses.
    if "addresses" in merged:
        merged["addresses"] = [
            a
            for a in merged["addresses"]
            if any(
                a.get(k) for k in ("line", "city", "state", "postal_code", "country")
            )
        ]
        if not merged["addresses"]:
            del merged["addresses"]

    # Clean contacts recursively.
    if "contacts" in merged:
        valid_contacts = []
        for c in merged["contacts"]:
            # Telecom inside contact
            for tel_key in ("phones", "emails", "faxes"):
                if tel_key in c:
                    c[tel_key] = [t for t in c[tel_key] if t.get("value")]
                    if not c[tel_key]:
                        c.pop(tel_key, None)
            # Addresses inside contact
            if "addresses" in c:
                c["addresses"] = [
                    a
                    for a in c["addresses"]
                    if any(
                        a.get(k)
                        for k in ("line", "city", "state", "postal_code", "country")
                    )
                ]
                if not c["addresses"]:
                    c.pop("addresses", None)
            # Ensure name field
            if "name" not in c or not isinstance(c["name"], dict):
                c["name"] = {"first_name": None, "last_name": None, "full_name": None}
            valid_contacts.append(c)
        if valid_contacts:
            merged["contacts"] = valid_contacts
        else:
            del merged["contacts"]

    # Guarantee the required 'name' object remains after sanitization.
    if "name" not in merged or not isinstance(merged["name"], dict):
        merged["name"] = {"first_name": None, "last_name": None, "full_name": None}
    elif not merged["name"]:
        merged["name"] = {"first_name": None, "last_name": None, "full_name": None}

    if not merged:
        return None

    return PatientProfile.model_validate(merged)


def extract_patient_profile(
    text: str, model: str = "gpt-4o-mini"
) -> Optional[PatientProfile]:
    """Extract *all* known patient information by combining the 3 partial prompts."""
    return asyncio.run(_extract_patient_profile_async(text, model=model))


# ---------------------------------------------------------------------------
# CLI / manual test ----------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = (
        "Hi, I'm John Doe (male, born 1990-05-15). Call me on +1-234-567-8901. "
        "My wife Jane Doe (female) can be reached at 987-654-3210 if there's an emergency."
    )

    test_report_2 = """
    Back See All Profiles
    Picture of LEMTRADA patient Kim.
    Kim
    44
    Magnolia, TX
    married, mother of 3, pre-school teacher
    “I live an active life, and my disease progression made caring for my 3 children increasingly difficult.”
    DISEASE HISTORY
    20 Years
    since RMS diagnosis

    PATIENT BACKGROUND
    RMS Risk Factors
    African American, incomplete recovery from relapse

    RMS TREATMENT HISTORY

    2 prior DMTs:
    1 injectable and
    1 oral therapy

    WHAT MOTIVATED YOU TO EXPLORE OTHER TREATMENT OPTIONS?
    I was having multiple relapses and I was determined to find another treatment option. My HCP team agreed to discuss options and see what to do next.

    WHY LEMTRADA?
    After reviewing both the safety and efficacy data with my doctor, my family and I felt the potential benefits outweighed the risks for me. The required monthly monitoring and self-checks would allow me and my HCP to keep an eye on my health. And if getting to my HCP or lab was too difficult, I knew that I could have a visiting lab technician come to my home.

    DID YOU HAVE ANY CONCERNS ABOUT STARTING LEMTRADA?
    Potential autoimmune side effects were my greatest concern. As a result of one of my regular self-checks, I discovered a blister in my mouth and I contacted my doctor immediately. He told me it was ITP, or immune thrombocytopenic purpura, which meant that I had low blood platelet levels.

    WHAT WOULD YOU TELL SOMEONE ABOUT YOUR EXPERIENCE?
    I’m at the end of my scheduled monitoring since taking LEMTRADA 5 years ago. The ITP was treated, I've had no relapses, and my doctor hasn’t seen any signs of disability progression. I'm glad I learned more about LEMTRADA and was able to make an informed decision. In the future, if my doctor thinks I need an additional course of treatment and monitoring, I will do it again.

    Are your patients interested in learning
    more about LEMTRADA?
    INDICATION
    LEMTRADA is indicated for the treatment of relapsing forms of multiple sclerosis (MS), to include relapsing-remitting disease and active secondary progressive disease, in adults. Because of its safety profile, the use of LEMTRADA should generally be reserved for patients who have had an inadequate response to two or more drugs indicated for the treatment of MS.

    Limitations of Use: LEMTRADA is not recommended for use in patients with clinically isolated syndrome (CIS) because of its safety profile.

    IMPORTANT SAFETY INFORMATION
    WARNING: AUTOIMMUNITY, INFUSION REACTIONS, STROKE AND MALIGNANCIES
    LEMTRADA causes serious, sometimes fatal, autoimmune conditions such as immune thrombocytopenia and anti-glomerular basement membrane (anti-GBM) disease. Monitor complete blood counts with differential, serum creatinine levels, and urinalysis with urine cell counts before starting treatment and then at monthly intervals until 48 months after the last dose of LEMTRADA.
    LEMTRADA causes serious and life-threatening infusion reactions. LEMTRADA must be administered in a setting with appropriate equipment and personnel to manage anaphylaxis or serious infusion reactions. Monitor patients for two hours after each infusion. Make patients aware that serious infusion reactions can also occur after the 2-hour monitoring period.
    Serious and life-threatening stroke (including ischemic and hemorrhagic stroke) has been reported within 3 days of LEMTRADA administration. Instruct patients to seek immediate medical attention if symptoms of stroke occur.
    LEMTRADA may cause an increased risk of malignancies, including thyroid cancer, melanoma, and lymphoproliferative disorders. Perform baseline and yearly skin exams.
    Because of the risk of autoimmunity, infusion reactions, and malignancies, LEMTRADA is available only through restricted distribution under a Risk Evaluation and Mitigation Strategy (REMS) Program. Call 1-855-676-6326 to enroll in the LEMTRADA REMS Program.
    CONTRAINDICATIONS
    LEMTRADA is contraindicated in patients:

    with known hypersensitivity or anaphylactic reactions to alemtuzumab or any of the excipients in LEMTRADA
    who are infected with Human Immunodeficiency Virus (HIV) because LEMTRADA causes prolonged reductions of CD4+ lymphocyte counts
    with an active infection
    WARNINGS AND PRECAUTIONS
    Autoimmunity: Treatment with LEMTRADA can result in the formation of autoantibodies and increase the risk of serious autoimmune-mediated conditions, which may be life threatening. Measure the urine protein to creatinine ratio prior to initiation of treatment. Obtain complete blood counts with differential, serum creatinine levels, and urinalysis with cell counts before starting treatment and then monitor at monthly intervals until 48 months after the last dose of LEMTRADA, or longer, if clinically indicated.
    Infusion Reactions: LEMTRADA causes cytokine release syndrome resulting in infusion reactions. In clinical studies, 92% of LEMTRADA-treated patients experienced infusion reactions. Serious reactions occurred in 3% of these patients and included anaphylaxis in 2 patients (including anaphylactic shock), angioedema, bronchospasm, hypotension, chest pain, bradycardia, tachycardia (including atrial fibrillation), transient neurologic symptoms, hypertension, headache, pyrexia, and rash. In some patients, infusion reactions were reported more than 24 hours after LEMTRADA infusion. Postmarketing cases of pulmonary alveolar hemorrhage, myocardial ischemia, and myocardial infarction have been reported with time to onset of 1-3 days from LEMTRADA infusion in the majority of cases. Patients should be informed about the signs and symptoms and advised to seek immediate medical attention if any of these symptoms occur. Cases of severe, including fatal, neutropenia have been reported within 2 months of LEMTRADA infusion. Mild to moderate decreases in platelet counts, starting at the time of alemtuzumab infusion have been reported. Consider additional monitoring in patients with medical conditions which predispose them to cardiovascular or pulmonary compromise.

    Premedicate patients with corticosteroids immediately prior to LEMTRADA infusion for the first 3 days of each treatment course. Consider pretreatment with antihistamines and/or antipyretics. Infusion reactions may occur despite pretreatment.

    LEMTRADA can only be administered in certified healthcare settings that have on-site access to equipment and personnel trained to manage infusion reactions (including anaphylaxis and cardiac and respiratory emergencies).

    Stroke and Cervicocephalic Arterial Dissection (CAD): In the postmarketing setting, serious and life-threatening stroke and cases of CAD involving multiple arteries have been reported within 1-3 days of LEMTRADA administration.

    Educate patients on the symptoms and instruct patients to seek immediate medical attention if symptoms of stroke or CAD occur.

    Malignancies: LEMTRADA may cause an increased risk of malignancies, including thyroid cancer, melanoma, and lymphoproliferative disorders. Monitor for symptoms of thyroid cancer. Perform baseline and yearly skin exams. Because LEMTRADA is an immunomodulatory therapy, caution should be exercised in initiating LEMTRADA in patients with pre-existing or ongoing malignancies.
    LEMTRADA REMS Program: Only prescribers, patients, pharmacies and healthcare facilities certified and enrolled in the REMS program can prescribe, receive, dispense or administer LEMTRADA.
    Immune thrombocytopenia (ITP) occurred in 2% of LEMTRADA-treated patients in clinical studies in MS. One LEMTRADA-treated patient developed ITP that went unrecognized prior to the implementation of monthly monitoring requirements, and died from an intracerebral hemorrhage. ITP has been diagnosed more than 3 years after the last LEMTRADA dose. If ITP is confirmed, promptly initiate medical intervention.
    Glomerular nephropathies, including anti-GBM disease, occurred in 0.3% of LEMTRADA-treated patients in MS clinical trials and have been diagnosed up to 40 months after the last dose of LEMTRADA. In postmarketing cases, some LEMTRADA-treated patients with anti-GBM disease developed end-stage renal disease requiring dialysis or renal transplantation. Urgent evaluation and treatment is required, because early detection and treatment of nephropathies can improve the preservation of renal function and may decrease the risk of poor outcomes. Anti-GBM disease can be life-threatening if left untreated. Alveolar hemorrhage, manifested as hemoptysis, is a common component of anti-GBM disease and has been reported in postmarketing cases. Increased serum creatinine with hematuria or signs of pulmonary involvement of anti-GBM disease warrant immediate evaluation. Patients and caregivers should be instructed to seek medical advice if they have concerns.
    Thyroid endocrine disorders, including autoimmune thyroid disorders, occurred in 36.8% of LEMTRADA-treated patients in MS clinical studies. Newly diagnosed thyroid disorders occurred throughout the uncontrolled clinical study follow-up period, more than 7 years after the first LEMTRADA dose. Serious thyroid events occurred in 5.2% of patients and included cardiac and psychiatric events. In LEMTRADA-treated patients, 3.8% underwent thyroidectomy. Thyroid disease poses special risks in women who are pregnant. In patients with an ongoing thyroid disorder, LEMTRADA should be administered only if the potential benefit justifies the potential risks. Obtain thyroid function tests prior to initiation of treatment and every 3 months until 48 months after the last infusion, or longer, if clinically indicated or in case of pregnancy.
    Other autoimmune cytopenias occurred in LEMTRADA-treated patients in MS clinical trials, including neutropenia, hemolytic anemia, and pancytopenia. One LEMTRADA-treated patient with autoimmune pancytopenia died from sepsis. Prompt medical intervention is indicated if a cytopenia is confirmed.
    Autoimmune hepatitis causing liver injury, including acute liver failure requiring transplant, has been reported in patients in the postmarketing setting. Obtain serum transaminases and total bilirubin levels prior to starting LEMTRADA, at periodic intervals until 48 months after the last dose, and promptly upon patient developing signs or symptoms suggestive of hepatic dysfunction. Interrupt or discontinue treatment, as appropriate.
    Hemophagocytic lymphohistiocytosis (HLH) has occurred in patients treated with LEMTRADA, with symptoms reported to occur within approximately thirteen months and thirty-three months following treatment initiation. HLH is associated with high mortality rates if not recognized and treated early. In cases of HLH reported with LEMTRADA, most patients presented with fever, elevated ferritin, transaminitis, hypertriglyceridemia, and all patients required hospitalization. Additional common findings include hepatosplenomegaly, rash, lymphadenopathy, neurologic symptoms, cytopenias, and coagulation abnormalities. Patients who develop early manifestations of pathologic immune activation should be evaluated immediately, and a diagnosis of HLH should be considered. LEMTRADA should be discontinued if an alternate etiology for the signs or symptoms cannot be established.
    Adult Onset Still’s Disease (AOSD) has been reported during postmarketing use in patients treated with LEMTRADA. Patients with AOSD may have a combination of the following signs and symptoms: fever, arthritis, rash and leukocytosis in the absence of infections, malignancies, and other rheumatic conditions. Patients with manifestations of AOSD should be evaluated immediately and LEMTRADA should be discontinued if an alternate etiology cannot be established.
    Thrombotic Thrombocytopenic Purpura (TTP) has been reported in patients treated with LEMTRADA and is associated with high morbidity and mortality rates if not recognized and treated early. If TTP is suspected, evaluate immediately and discontinue LEMTRADA if TTP is confirmed or an alternate etiology is not confirmed.
    Autoimmune Encephalitis (AIE) has been reported during postmarketing use in patients treated with LEMTRADA. Clinical manifestations of AIE may include subacute onset of memory impairment, altered mental status, psychiatric symptoms, neurological findings, and seizures. LEMTRADA should be discontinued if AIE is confirmed by the presence of neural autoantibodies or an alternate etiology cannot be established.
    Acquired Hemophilia A has been reported in clinical trial and postmarketing settings. Inform patients about the signs and symptoms of acquired hemophilia A and to seek immediate medical attention. Obtain a coagulopathy panel including aPTT in patients who present with spontaneous subcutaneous hematomas, extensive bruising, hematuria, epistaxis, or gastrointestinal or other types of bleeding.
    Immune-Mediated Colitis has been reported in the postmarketing setting. Monitor patients for new or persistent diarrhea or other gastrointestinal symptoms, and evaluate promptly if colitis is suspected.
    Infections occurred in 71% of LEMTRADA-treated patients compared to 53% of patients treated with interferon beta-1a in clinical studies. Serious infections occurred in 3% of patients treated with LEMTRADA as compared to 1% of patients treated with interferon beta-1a. Serious infections in the LEMTRADA group included: appendicitis, gastroenteritis, pneumonia, herpes zoster, and tooth infection.
    Do not administer live viral vaccines following a course of LEMTRADA, as patients may be at increased risk of infection.
    LEMTRADA administration is contraindicated in patients with active infection.
    Concomitant use of antineoplastic or immunosuppressive therapies could increase the risk of immunosuppression.
    In the postmarketing setting, serious, sometimes fatal, opportunistic infections have been reported, including aspergillosis, coccidioidomycosis, histoplasmosis, Pneumocystis jirovecii pneumonia, nocardiosis, Epstein-Barr virus, and cytomegalovirus infections.
    Listeria monocytogenes infections, including fatal cases of Listeria meningoencephalitis, have occurred in LEMTRADA-treated patients. Listeria infections have occurred between 3 days to 8 months after taking LEMTRADA. Advise patients to avoid or adequately heat foods that are potential sources for Listeria monocytogenes. Initiate these precautions prior to receiving LEMTRADA. Advise patients to watch for symptoms of Listeria infection and seek prompt medical help if symptoms occur.
    Herpes viral infection developed in 16% of LEMTRADA-treated patients compared to 3% of interferon beta-1a patients. Administer antiviral prophylaxis for herpetic viral infections starting on the first day of each treatment course and continue for a minimum of two months following treatment with LEMTRADA or until CD4+ lymphocyte count is ≥200 cells per microliter, whichever occurs later.
    Cervical human papilloma virus (HPV) infection occurred in 2% of LEMTRADA-treated patients. Annual screening is recommended for female patients.
    Active and latent tuberculosis cases occurred in 0.3% of LEMTRADA-treated patients, most often in endemic regions.
    Fungal infections, especially oral and vaginal candidiasis, occurred in 12% of LEMTRADA-treated patients compared to 3% of interferon beta-1a patients.
    Before initiating LEMTRADA, consider screening patients at high risk of Hepatitis B Virus (HBV) and Hepatitis C Virus (HCV) infection. Carriers of HBV and/or HCV who receive LEMTRADA may be at risk of irreversible liver damage relative to a potential virus reactivation.
    Progressive Multifocal Leukoencephalopathy (PML) has occurred in a patient with MS treated with LEMTRADA, diagnosed two months after the second course of treatment. The patient had previously received multiple MS therapies, but had not received other drugs for treatment of MS for more than one year. PML is an opportunistic viral infection of the brain caused by the JC virus (JCV) that typically only occurs in patients who are immunocompromised, and that usually leads to death or severe disability. At the first sign or symptom suggestive of PML, withhold LEMTRADA and perform an appropriate diagnostic evaluation. Typical symptoms associated with PML are diverse, progress over days to weeks, and include progressive weakness on one side of the body or clumsiness of limbs, disturbance of vision, and changes in thinking, memory, and orientation leading to confusion and personality changes. MRI findings may be apparent before clinical signs or symptoms. Instruct the patient to contact their doctor if they develop any symptoms suggestive of PML.
    Acute Acalculous Cholecystitis (AAC): LEMTRADA may increase the risk of AAC, which occurred in 0.2% of LEMTRADA-treated MS patients compared to 0% of patients treated with interferon beta-1a. Postmarketing cases of AAC have also been reported. Time to onset of symptoms ranged from less than 24 hours to 2 months after LEMTRADA infusion. Typical risk or predisposing factors such as concurrent critical illness were often not reported. AAC is associated with high morbidity and mortality if not diagnosed early and treated. If AAC is suspected, evaluate and treat promptly.
    Pneumonitis, including hypersensitivity pneumonitis and pneumonitis with fibrosis, occurred in 0.5% of LEMTRADA-treated patients in clinical studies. Cases of hypersensitivity pneumonitis and pneumonitis with fibrosis occurred in clinical studies. Advise patients to report symptoms of pneumonitis (e.g., shortness of breath, cough, wheezing, chest pain or tightness, and hemoptysis).
    Drug Products with Same Active Ingredient: LEMTRADA contains the same active ingredient (alemtuzumab) found in CAMPATH®. If LEMTRADA is considered for use in a patient who has previously received CAMPATH, exercise increased vigilance for additive and long-lasting effects on the immune system.
    Most Common Adverse Reactions
    In controlled clinical trials, the most common adverse reactions (incidence ≥10% and >interferon beta-1a) with LEMTRADA vs interferon beta-1a were: rash (53% vs 6%), headache (52% vs 23%), pyrexia (29% vs 9%), nasopharyngitis (25% vs 19%), nausea (21% vs 9%), urinary tract infection (19% vs 8%), fatigue (18% vs 13%), insomnia (16% vs 15%), upper respiratory tract infection (16% vs 13%), herpes viral infection (16% vs 3%), urticaria (16% vs 2%), pruritus (14% vs 2%), thyroid gland disorders (13% vs 3%), fungal infection (13% vs 4%), arthralgia (12% vs 9%), pain in extremity (12% vs 9%), back pain (12% vs 8%), diarrhea (12% vs 6%), sinusitis (11% vs 8%), oropharyngeal pain (11% vs 5%), paresthesia (10% vs 8%), dizziness (10% vs 5%), abdominal pain (10% vs 5%), flushing (10% vs 4%), and vomiting (10% vs 3%).

    Use in Specific Populations
    Physicians are encouraged to report pregnancies by calling 1-800-745-4447, option 2. Antibodies, including anti-CD52 and autoantibodies, may be transferred from the mother to the fetus during pregnancy. Placental transfer of anti-thyroid antibodies resulting in neonatal Graves’ disease has been reported.

    Safety and effectiveness in pediatric patients less than 17 years of age have not been established. Use of LEMTRADA is not recommended in pediatric patients due to the risks of autoimmunity and infusion reactions, stroke, and because it may increase the risk of malignancies.

    Please see full Prescribing Information, including Boxed WARNING.
    """

    test_text_3 = """
    User: hi, i want to visit the clinic. when can i go?

    Medical AI: Hello! I can help you schedule your visit to the clinic. What day and time are you looking to come in?

    User: today is good. is the clinic open til late? i get off work at 5pm.

    Medical AI: Yes, the clinic is open until 7 PM on weekdays. We have an opening at 5:30 PM today. Would that work for you?

    User: 5.30 today is fine. what is the clinic phone number?

    Medical AI: Great. Your appointment is confirmed for today at 5:30 PM. The clinic's phone number is (555) 123-4567.

    User: thanks, i will call if i get late. oh, and can you also remind my son, Alex, he is 10, about his appointment next week?

    Medical AI: You're welcome! I can certainly do that. To remind Alex about his appointment, could you please provide his phone number?

    User: sure, his mom's phone is fine, its 555-987-6543.

    Medical AI: Thank you. I have noted to send a reminder to that number for Alex's appointment next week. Is there anything else I can assist you with today?
    """

    profile = extract_patient_profile(test_report_2)
    print("Merged profile:", profile.model_dump(exclude_none=True) if profile else None)
