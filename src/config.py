# Column mapping: logical key → Silver table column name.
COLUMNS = {
    "id":               "unique_id",
    "name":             "name",
    "city":             "city",
    "state":            "state",
    "postcode":         "postcode",
    "latitude":         "latitude",
    "longitude":        "longitude",
    "geo_source":       "geo_source",   # ORIGINAL/POSTCODE/CITY_AVG/STATE_AVG/UNKNOWN
    "geo_valid":        "geo_valid",    # True if any coordinate was resolved
    "specialties":      "specialties",
    "description":      "description",
    "capability":       "capability",
    "procedure":        "procedure_text",
    "equipment":        "equipment",
    "num_doctors":      "num_doctors",
    "capacity":         "capacity",
    "year_established": "year_established",
    "source_urls":      "source_urls",
    "phone":            "phone",
    "website":          "website",
    "org_type":         "organization_type",
    "completeness":     "completeness_score",
}

# Silver pre-flattens all JSON arrays — no runtime parsing needed.
ARRAY_FIELDS = []

# Fields scanned for "matching evidence"
EVIDENCE_TEXT_FIELDS = ["specialties", "description", "capability", "procedure", "equipment"]

# Fields whose absence is flagged as "missing evidence"
COMPLETENESS_FIELDS = ["num_doctors", "capacity", "year_established", "source_urls"]

# Care-need synonyms.
# Silver expands camelCase specialty codes to space-separated words:
#   emergencyMedicine  -> "emergency Medicine"  (searches as "emergency medicine")
#   orthopedicSurgery  -> "orthopedic Surgery"  (searches as "orthopedic surgery")
# Synonyms use the plain-English form so they match both expanded Silver text
# and verbose description/procedure/capability fields.
CARE_NEED_SYNONYMS = {
    "dialysis": [
        "dialysis", "hemodialysis", "haemodialysis",
        "renal care", "renal medicine",
        "nephrology", "kidney", "peritoneal",
    ],
    "emergency": [
        "emergency medicine", "emergency surgery", "emergency care",
        "emergency department", "emergency preparedness",
        "trauma", "casualty", "accident",
    ],
    "cardiology": [
        "cardiology", "cardiac", "heart",
        "interventional cardiology", "cardiothoracic",   # "cardiothoracicSurgery" expanded
        "heart failure", "cath lab", "angioplasty", "bypass",
    ],
    "maternity": [
        "gynecology and obstetrics", "gynecology", "obstetrics",
        "neonatology", "perinatal",                      # "neonatologyPerinatalMedicine" expanded
        "maternal", "maternity",
        "antenatal", "prenatal", "postnatal",
        "labour ward", "labor ward", "delivery room", "delivery suite",
        "birthing", "caesarean", "cesarean",
    ],
    "oncology": [
        "oncology", "cancer",
        "medical oncology", "surgical oncology",
        "radiation oncology", "gynecological oncology",
        "chemotherapy", "radiotherapy", "tumor",
    ],
    "orthopedics": [
        "orthopedic", "orthopaedic",                     # "orthopedicSurgery" expanded -> need substring
        "bone", "fracture",
        "joint replacement", "joint reconstruction",
        "spine", "spinal",
    ],
    "icu": [
        "critical care medicine", "critical care",
        "intensive care", "ventilator", "icu",
    ],
    "neurology": [
        "neurology", "neurosurgery", "neuro",
        "spine neurosurgery", "peripheral nerve",
        "stroke", "brain", "epilepsy", "parkinson",
    ],
    "ophthalmology": [
        "ophthalmology", "eye", "cataract", "retina",
        "glaucoma", "cornea",
    ],
    "pediatrics": [
        "pediatric", "paediatric",                       # "pediatricSurgery" expanded
        "neonatology", "neonatal", "infant",
        "child",
    ],
    "general surgery": [
        "general surgery", "surgery",
        "laparoscopy", "hernia", "appendectomy", "gastrointestinal",
    ],
    "radiology": [
        "radiology", "imaging", "mri", "ct scan", "x-ray",
        "ultrasound", "diagnostic imaging", "pet scan",
    ],
    "ent": [
        "otolaryngology", "ent", "ear nose throat",
        "hearing", "sinusitis", "tonsil",
    ],
    "dermatology": [
        "dermatology", "skin", "dermatitis", "eczema", "psoriasis",
    ],
    "psychiatry": [
        "psychiatry", "neuropsychiatry", "psychology", "mental health",
        "behavioral", "addiction", "rehabilitation",
    ],
}
