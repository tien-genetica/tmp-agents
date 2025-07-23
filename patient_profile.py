from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Identifier(BaseModel):
    system: str
    value: str


class HumanName(BaseModel):
    family: str
    given: List[str] = Field(default_factory=list)


class ContactPoint(BaseModel):
    system: str  # phone | email | fax | etc.
    value: str
    use: str  # home | work | mobile | temp | old


class Address(BaseModel):
    line: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = Field(None, alias="postalCode")
    country: Optional[str] = None

    class Config:
        validate_by_name = True


class Coding(BaseModel):
    system: Optional[str] = None
    code: Optional[str] = None
    display: Optional[str] = None


class CodeableConcept(BaseModel):
    coding: List[Coding] = Field(default_factory=list)
    text: Optional[str] = None


class Reference(BaseModel):
    reference: Optional[str] = None
    display: Optional[str] = None


class Contact(BaseModel):
    relationship: List[CodeableConcept] = Field(default_factory=list)
    name: Optional[HumanName] = None
    telecom: List[ContactPoint] = Field(default_factory=list)
    address: Optional[Address] = None
    gender: Optional[str] = None
    organization: Optional[Reference] = None


class PatientCommunication(BaseModel):
    language: CodeableConcept
    preferred: Optional[bool] = None


# Patient resource (FHIR)


class PatientProfile(BaseModel):
    # Basic
    id: str
    identifier: List[Identifier] = Field(default_factory=list)
    active: bool = True

    # Demographics
    name: List[HumanName] = Field(default_factory=list)
    telecom: List[ContactPoint] = Field(default_factory=list)
    gender: Optional[str] = None  # male | female | other | unknown
    birth_date: Optional[date] = Field(None, alias="birthDate")
    deceased_boolean: Optional[bool] = Field(None, alias="deceasedBoolean")
    deceased_date: Optional[date] = Field(None, alias="deceasedDate")

    # Socio-economic
    address: List[Address] = Field(default_factory=list)
    marital_status: Optional[str] = Field(None, alias="maritalStatus")
    language: Optional[str] = None

    # Relationships
    contact: List[Contact] = Field(default_factory=list)
    communication: List[PatientCommunication] = Field(default_factory=list)
    managing_organization: Optional[Reference] = Field(
        None, alias="managingOrganization"
    )
    general_practitioner: List[Reference] = Field(
        default_factory=list, alias="generalPractitioner"
    )

    # Embedded medical record
    encounters: List["Encounter"] = Field(default_factory=list)
    conditions: List["Condition"] = Field(default_factory=list)
    observations: List["Observation"] = Field(default_factory=list)
    medication_requests: List["MedicationRequest"] = Field(
        default_factory=list, alias="medicationRequests"
    )

    class Config:
        validate_by_name = True
        str_strip_whitespace = True
        validate_assignment = True
        frozen = True  # make instances hashable & immutable (optional)

    # ------------------------------------------------------------------
    # Helpers for FHIR compliant import/export
    # ------------------------------------------------------------------

    def to_fhir(self) -> Dict[str, Any]:
        """Serialize the model to a FHIR-compliant JSON dict."""
        payload = self.model_dump(
            by_alias=True,
            exclude_none=True,
            mode="json",
            exclude={
                "encounters",
                "conditions",
                "observations",
                "medication_requests",
            },
        )
        payload["resourceType"] = "Patient"
        return payload

    @classmethod
    def from_fhir(cls, payload: Dict[str, Any]) -> "PatientProfile":
        """Build a PatientProfile from a FHIR JSON dict."""
        if payload.get("resourceType") != "Patient":
            raise ValueError("payload is not a FHIR Patient resource")
        # Remove the resourceType key so pydantic ignores it
        payload = {k: v for k, v in payload.items() if k != "resourceType"}
        return cls.model_validate(payload)

    # Bundle helper: patient + clinical resources

    def to_profile(self) -> Dict[str, Any]:
        """Return a full FHIR Bundle ("profile") containing patient & clinical resources."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": self.to_fhir()}],
        }

        for seq in (
            self.encounters,
            self.conditions,
            self.observations,
            self.medication_requests,
        ):
            bundle["entry"].extend({"resource": r.to_fhir()} for r in seq)

        return bundle


# Clinical resource models


class Condition(BaseModel):
    id: str
    subject: Reference  # Reference to Patient
    code: CodeableConcept
    clinical_status: Optional[str] = Field(None, alias="clinicalStatus")
    onset_date: Optional[date] = Field(None, alias="onsetDate")
    recorded_date: Optional[date] = Field(None, alias="recordedDate")
    note: Optional[str] = None

    class Config:
        validate_by_name = True
        str_strip_whitespace = True

    def to_fhir(self) -> Dict[str, Any]:
        payload = self.model_dump(by_alias=True, exclude_none=True, mode="json")
        payload["resourceType"] = "Condition"
        return payload


class Observation(BaseModel):
    id: str
    subject: Reference
    code: CodeableConcept
    status: str
    effective_datetime: Optional[datetime] = Field(None, alias="effectiveDateTime")
    value: Optional[str] = Field(None, alias="valueString")

    class Config:
        validate_by_name = True
        str_strip_whitespace = True

    def to_fhir(self) -> Dict[str, Any]:
        payload = self.model_dump(by_alias=True, exclude_none=True, mode="json")
        payload["resourceType"] = "Observation"
        return payload


class Encounter(BaseModel):
    id: str
    subject: Reference
    status: str
    class_code: Optional[str] = Field(None, alias="class")
    period_start: Optional[datetime] = Field(None, alias="periodStart")
    period_end: Optional[datetime] = Field(None, alias="periodEnd")

    class Config:
        validate_by_name = True
        str_strip_whitespace = True

    def to_fhir(self) -> Dict[str, Any]:
        payload = self.model_dump(by_alias=True, exclude_none=True, mode="json")
        payload["resourceType"] = "Encounter"
        return payload


class MedicationRequest(BaseModel):
    id: str
    subject: Reference
    status: str
    intent: str
    medication: CodeableConcept
    authored_on: Optional[date] = Field(None, alias="authoredOn")
    dosage_instruction: Optional[str] = Field(None, alias="dosageInstruction")

    class Config:
        validate_by_name = True
        str_strip_whitespace = True

    def to_fhir(self) -> Dict[str, Any]:
        payload = self.model_dump(by_alias=True, exclude_none=True, mode="json")
        payload["resourceType"] = "MedicationRequest"
        return payload


if __name__ == "__main__":
    patient = PatientProfile(
        id="example-001",
        identifier=[
            Identifier(system="http://hospital.smarthealth.org", value="MRN123456")
        ],
        name=[HumanName(family="Doe", given=["John"])],
        telecom=[ContactPoint(system="phone", value="+1-555-555-0000", use="mobile")],
        gender="male",
        birthDate=date(1985, 5, 23),  # we can use alias name
        address=[
            Address(
                line=["1 Main St"],
                city="Metropolis",
                state="NY",
                postalCode="12345",
                country="USA",
            )
        ],
        communication=[
            PatientCommunication(
                language=CodeableConcept(
                    text="English", coding=[Coding(system="urn:ietf:bcp:47", code="en")]
                ),
                preferred=True,
            )
        ],
    )

    # Create a sample diagnosis (Condition)
    condition = Condition(
        id="cond-001",
        subject=Reference(reference=f"Patient/{patient.id}"),
        code=CodeableConcept(text="Hypertension"),
        clinicalStatus="active",
        onsetDate=date(2020, 1, 1),
    )

    # Export as FHIR Bundle containing patient and condition
    import json, sys

    json.dump(patient.to_profile(), sys.stdout, indent=2)
