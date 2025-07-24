from __future__ import annotations

from datetime import date
from typing import List, Optional, Dict, Any, Literal, TypeAlias


from pydantic import BaseModel, Field

# Enumerations
Gender: TypeAlias = Literal["male", "female", "other", "unknown"]
MaritalStatus: TypeAlias = Literal[
    "single",
    "married",
    "divorced",
    "widowed",
    "separated",
    "unknown",
]


class DeceasedStatus(BaseModel):
    """Represents patient death information."""

    is_deceased: bool = False
    date: Optional[date] = None


class HumanName(BaseModel):
    first_name: str
    last_name: str
    full_name: Optional[str] = None

    def to_fhir_dict(self) -> Dict[str, Any]:
        """Convert this HumanName to a FHIR-compliant dict."""
        text_val = self.full_name or f"{self.first_name} {self.last_name}"
        return {
            "family": self.last_name,
            "given": [self.first_name],
            "text": text_val,
        }


class Telecom(BaseModel):
    system: Literal["phone", "fax", "email", "url", "sms", "other"]
    value: str
    use_for: Optional[Literal["home", "work", "temp", "old", "mobile"]] = None


class Address(BaseModel):
    line: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class Organization(BaseModel):
    reference: Optional[str] = None
    display: Optional[str] = None


class Contact(BaseModel):
    relationship: List[str] = Field(default_factory=list)
    name: Optional[HumanName] = None
    telecoms: List[Telecom] = Field(default_factory=list)
    addresses: List[Address] = Field(default_factory=list)
    gender: Optional[Gender] = None
    organizations: List[Organization] = Field(default_factory=list)


class Language(BaseModel):
    language: str
    preferred: Optional[bool] = None


class PatientProfile(BaseModel):
    id: str
    name: HumanName
    other_names: List[HumanName] = Field(default_factory=list)
    telecoms: List[Telecom] = Field(default_factory=list)
    gender: Optional[Gender] = None
    birth_date: Optional[date] = None
    addresses: List[Address] = Field(default_factory=list)
    deceased: DeceasedStatus = Field(default_factory=DeceasedStatus)
    marital_status: Optional[MaritalStatus] = None
    contacts: List[Contact] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    managing_organization: Optional[Organization] = None
    medical_records: List["MedicalRecord"] = Field(default_factory=list)

    def to_fhir(self) -> Dict[str, Any]:
        """Serialize the model to a FHIR-compliant JSON dict."""
        # Base demographic payload excluding complex lists
        payload: Dict[str, Any] = {}

        # Identifiers
        payload["id"] = self.id

        # Names
        names_list = [self.name.to_fhir_dict()] + [
            n.to_fhir_dict() for n in self.other_names
        ]
        payload["name"] = names_list

        # Telecoms
        if self.telecoms:
            payload["telecom"] = [t.__dict__ for t in self.telecoms]

        # Gender / birth / marital
        if self.gender:
            payload["gender"] = self.gender
        if self.birth_date:
            payload["birthDate"] = self.birth_date.isoformat()
        if self.marital_status:
            payload["maritalStatus"] = {
                "coding": [{"code": self.marital_status}],
                "text": self.marital_status,
            }

        # Addresses
        if self.addresses:
            payload["address"] = [
                a.model_dump(exclude_none=True, mode="json") for a in self.addresses
            ]

        # Languages
        if self.languages:
            payload["communication"] = [
                {"language": {"text": l.language}, "preferred": l.preferred}
                for l in self.languages
            ]

        # Managing organization
        if self.managing_organization:
            payload["managingOrganization"] = self.managing_organization.model_dump(
                exclude_none=True, mode="json"
            )

        # Contacts
        if self.contacts:
            payload["contact"] = [
                {
                    "relationship": c.relationship,
                    "name": c.name.to_fhir_dict() if c.name else None,
                    "telecom": [t.__dict__ for t in c.telecoms] if c.telecoms else None,
                    "address": (
                        [
                            a.model_dump(exclude_none=True, mode="json")
                            for a in c.addresses
                        ]
                        if c.addresses
                        else None
                    ),
                    "gender": c.gender,
                    "organization": (
                        [
                            o.model_dump(exclude_none=True, mode="json")
                            for o in c.organizations
                        ]
                        if c.organizations
                        else None
                    ),
                }
                for c in self.contacts
            ]

        # Deceased mapping
        if self.deceased.is_deceased:
            payload["deceasedBoolean"] = True
        if self.deceased.date is not None:
            payload["deceasedDate"] = self.deceased.date.isoformat()

        # Resource type
        payload["resourceType"] = "Patient"

        return payload

    @classmethod
    def from_fhir(cls, payload: Dict[str, Any]) -> "PatientProfile":
        """Build a PatientProfile from a FHIR JSON dict."""
        if payload.get("resourceType") != "Patient":
            raise ValueError("payload is not a FHIR Patient resource")
        data: Dict[str, Any] = {}
        data["id"] = payload.get("id")

        # Names
        name_entries = payload.get("name", [])
        if name_entries:
            first = name_entries[0]
            data["name"] = HumanName(
                first_name=first.get("given", [""])[0],
                last_name=first.get("family", ""),
                full_name=first.get("text"),
            )
            data["other_names"] = [
                HumanName(
                    first_name=n.get("given", [""])[0],
                    last_name=n.get("family", ""),
                    full_name=n.get("text"),
                )
                for n in name_entries[1:]
            ]

        # Telecoms
        data["telecoms"] = [Telecom(**t) for t in payload.get("telecom", [])]

        # Gender, birth, marital
        data["gender"] = payload.get("gender")
        if "birthDate" in payload:
            data["birth_date"] = date.fromisoformat(payload["birthDate"])
        if ms := payload.get("maritalStatus"):
            data["marital_status"] = ms.get("coding", [{}])[0].get("code") or ms.get(
                "text"
            )

        # Addresses
        data["addresses"] = [
            Address.model_validate(a) for a in payload.get("address", [])
        ]

        # Languages (communication)
        data["languages"] = [
            Language(
                language=comm.get("language", {}).get("text", ""),
                preferred=comm.get("preferred"),
            )
            for comm in payload.get("communication", [])
        ]

        # Deceased
        dec_bool = payload.get("deceasedBoolean", False)
        dec_date = payload.get("deceasedDate")
        deceased_obj = DeceasedStatus(
            is_deceased=bool(dec_bool),
            date=date.fromisoformat(dec_date) if dec_date else None,
        )
        data["deceased"] = deceased_obj

        # Contacts
        contacts_data = []
        for c in payload.get("contact", []):
            contacts_data.append(
                Contact(
                    relationship=c.get("relationship", []),
                    name=(
                        HumanName(
                            first_name=c.get("name", {}).get("given", [""])[0],
                            last_name=c.get("name", {}).get("family", ""),
                        )
                        if c.get("name")
                        else None
                    ),
                    telecoms=[Telecom(**t) for t in c.get("telecom", [])],
                    addresses=(
                        [Address.model_validate(a) for a in c.get("address", [])]
                        if c.get("address")
                        else []
                    ),
                    gender=c.get("gender"),
                )
            )
        data["contacts"] = contacts_data

        return cls.model_validate(data)


class MedicalRecord(BaseModel):
    """Placeholder for medical record grouping; currently no fields."""

    pass


def main() -> None:
    """Quick manual test for PatientProfile serialization."""

    # Build example patient
    patient = PatientProfile(
        id="patient-001",
        name=HumanName(first_name="John", last_name="Doe"),
        other_names=[HumanName(first_name="Johnny", last_name="Doe")],
        telecoms=[Telecom(system="phone", value="+1-555-555-0000", use_for="mobile")],
        gender="male",
        birth_date=date(1985, 4, 20),
        addresses=[
            Address(line=["1 Main St"], city="Metropolis", state="NY", country="USA")
        ],
        marital_status="single",
        languages=[Language(language="en", preferred=True)],
        contacts=[
            Contact(
                relationship=["mother"],
                name=HumanName(first_name="Jane", last_name="Doe"),
                telecoms=[Telecom(system="phone", value="+1-555-555-0001")],
                gender="female",
            )
        ],
    )

    # Convert to FHIR JSON
    as_fhir = patient.to_fhir()

    # Round-trip back to object
    patient_rt = PatientProfile.from_fhir(as_fhir)

    import json, pprint

    print("FHIR Patient JSON:")
    print(json.dumps(as_fhir, indent=2))

    print("\nRound-trip object:")
    pprint.pp(patient_rt)


if __name__ == "__main__":
    main()
