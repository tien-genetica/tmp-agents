"""MedicalGuideline – generalized clinical guideline model.

This module defines a Pydantic v2 model `MedicalGuideline` that can store
both international (e.g. WHO, CDC) and local Vietnamese medical
standards.  The model is intentionally simple but can be expanded to map
to full FHIR resources such as *PlanDefinition* or *Guideline* if
needed.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field


class ReferenceRange(BaseModel):
    """Numeric reference range for a lab test.

    Attributes
    ----------
    lower : float | None
        Lower bound (inclusive) for normal range; ``None`` if not defined.
    upper : float | None
        Upper bound (inclusive) for normal range; ``None`` if not defined.
    unit : str
        Measurement unit (e.g. ``"mmol/L"``).
    age_min : int | None
        Minimum age in years the range applies to (``None`` for any age).
    age_max : int | None
        Maximum age in years the range applies to (``None`` for any age).
    sex : str | None
        ``"male"``, ``"female"``, or ``None`` for all sexes.
    """

    lower: Optional[float] = None
    upper: Optional[float] = None
    unit: str
    age_min: Optional[int] = Field(None, alias="ageMin")
    age_max: Optional[int] = Field(None, alias="ageMax")
    sex: Optional[str] = None


class LabTestStandard(BaseModel):
    """Represents reference ranges for a particular laboratory test."""

    code: str  # LOINC code or local code
    name: str
    international_ranges: List[ReferenceRange] = Field(
        default_factory=list, alias="internationalRanges"
    )
    vietnamese_ranges: List[ReferenceRange] = Field(
        default_factory=list, alias="vietnameseRanges"
    )

    class Config:
        validate_by_name = True
        str_strip_whitespace = True


class MedicalGuideline(BaseModel):
    """Clinical guideline metadata.

    Attributes
    ----------
    id : str
        Unique logical identifier of the guideline.
    title : str
        Human-readable title.
    description : str | None
        Detailed description or abstract.
    category : str
        Either ``"international"`` or ``"vietnamese"`` (or another custom
        taxonomy defined by your system).
    source : str
        Originating organization (e.g. *WHO*, *CDC*, *Bộ Y tế Việt Nam*).
    url : str | None
        Canonical URL or reference document link.
    effective_date : date | None
        Date the guideline came into effect.
    version : str | None
        Version label (e.g. *2024-05*).
    tags : list[str]
        Additional free-form tags for quick filtering.
    language : str | None
        Language code ( ``"en"`` , ``"vi"`` … ).
    lab_tests: List[LabTestStandard] = Field(default_factory=list, alias="labTests")
    """

    id: str
    title: str
    description: Optional[str] = None
    category: Literal["international", "vietnamese", "other"]
    source: str
    url: Optional[str] = None
    effective_date: Optional[date] = Field(None, alias="effectiveDate")
    version: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = None  # ISO-639-1
    lab_tests: List[LabTestStandard] = Field(default_factory=list, alias="labTests")

    class Config:
        validate_by_name = True
        str_strip_whitespace = True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (no extras)."""
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "MedicalGuideline":
        """Instantiate from JSON payload."""
        return cls.model_validate(payload)


# Quick demo when run as script
if __name__ == "__main__":
    who_bp = MedicalGuideline(
        id="who-bp-2024",
        title="WHO Guideline on Hypertension Management",
        category="international",
        source="WHO",
        url="https://www.who.int/publications/i/item/9789240079544",
        effectiveDate=date(2024, 3, 15),
        version="2024",
        tags=["hypertension", "adult"],
        language="en",
    )

    vn_diabetes = MedicalGuideline(
        id="vn-t2dm-2023",
        title="Vietnamese National Guideline for Type 2 Diabetes",
        category="vietnamese",
        source="Ministry of Health Vietnam",
        version="2023",
        effectiveDate=date(2023, 7, 1),
        tags=["diabetes"],
        language="vi",
        labTests=[
            LabTestStandard(
                code="14749-6",
                name="Glucose [Moles/volume] in Blood",
                internationalRanges=[
                    ReferenceRange(lower=3.9, upper=5.6, unit="mmol/L")
                ],
                vietnameseRanges=[ReferenceRange(lower=4.0, upper=6.0, unit="mmol/L")],
            )
        ],
    )

    import json, sys

    json.dump([who_bp.to_json(), vn_diabetes.to_json()], sys.stdout, indent=2)
