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
        "trauma", "casualty", "accident", "injury", "injured",
        "urgent care", "critical", "unconscious", "bleeding",
        "chest pain", "heart attack", "stroke",
    ],
    "cardiology": [
        "cardiology", "cardiac", "heart",
        "interventional cardiology", "cardiothoracic",   # "cardiothoracicSurgery" expanded
        "heart failure", "cath lab", "angioplasty", "bypass",
    ],
    "maternity": [
        # American spellings (camelCase expanded by Silver)
        "gynecology and obstetrics", "gynecology", "obstetrics", "obstetric",
        # British spellings — widely used in Indian hospitals
        "gynaecology and obstetrics", "gynaecology", "gynaecological", "obstetrical",
        # Common Indian abbreviations
        "obg", "ob/g", "ob-g", "ob&g",
        # Newborn / perinatal
        "neonatology", "perinatal", "nicu", "sncu", "newborn care",
        # Antenatal / postnatal (single and two-word forms)
        "maternal", "maternity",
        "antenatal", "ante natal", "prenatal", "pre natal",
        "postnatal", "post natal", "postpartum", "post partum",
        # Delivery & women's health
        "labour ward", "labor ward", "delivery room", "delivery suite",
        "birthing", "caesarean", "cesarean", "midwifery",
        "women's hospital", "women hospital", "mother and child",
        "reproductive health", "family planning",
        "fetal", "foetal",
    ],
    "oncology": [
        "oncology", "cancer",
        "medical oncology", "surgical oncology",
        "radiation oncology", "gynecological oncology",
        "chemotherapy", "radiotherapy", "tumor",
    ],
    "orthopedics": [
        "orthopedic", "orthopaedic",                     # "orthopedicSurgery" expanded -> need substring
        "bone", "fracture", "broken", "broken bone",
        "joint replacement", "joint reconstruction",
        "spine", "spinal",
        "knee", "hip replacement", "back pain", "dislocation", "sprain",
        "leg injury", "leg pain", "arm injury", "shoulder injury",
        "musculoskeletal", "ligament", "tendon",
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
        "glaucoma", "cornea", "vision", "optical",
        "eye pain", "eye infection", "eye injury", "blurred vision",
        "foreign body eye", "optometry", "lasik",
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
    "general medicine": [
        "general medicine", "internal medicine", "general practice",
        "family medicine", "primary care", "outpatient",
        "fever", "cold", "cough", "flu", "influenza",
        "body ache", "body pain", "headache", "fatigue",
        "vomiting", "nausea", "diarrhea", "loose motion",
        "weakness", "rash", "allergy", "infection",
        "diabetes", "hypertension", "blood pressure",
        "thyroid", "anaemia", "anemia",
    ],
    "pulmonology": [
        "pulmonology", "respiratory", "lung", "pulmonary",
        "asthma", "bronchitis", "pneumonia", "copd",
        "breathing difficulty", "shortness of breath", "cough blood",
    ],
    "gastroenterology": [
        "gastroenterology", "gastro", "liver", "hepatology",
        "stomach", "abdomen", "gut", "bowel", "digestive",
        "ulcer", "jaundice", "hepatitis", "pancreatitis",
        "endoscopy", "colonoscopy",
    ],
    "urology": [
        "urology", "urinary", "kidney stone", "prostate",
        "bladder", "ureter", "urethra", "uti",
        "burning urination", "blood in urine",
    ],
}
