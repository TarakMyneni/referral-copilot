import pandas as pd

from .config import COLUMNS, EVIDENCE_TEXT_FIELDS, COMPLETENESS_FIELDS


def _get(row, key, default=""):
    """Look up a logical field (via COLUMNS mapping) on a row, NaN-safe."""
    col = COLUMNS.get(key, key)
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return val


def evaluate_evidence(row, keywords):
    """
    Given a facility row and a list of care-need keywords, return:
      - matching: fields where a keyword was found (supports the referral)
      - missing: fields with no data at all (low confidence, not necessarily wrong)
      - suspicious: internal inconsistencies worth a human double-check
    """
    keywords_lower = [k.lower() for k in keywords]

    # --- Matching evidence ---
    matching = []
    for field in EVIDENCE_TEXT_FIELDS:
        text = str(_get(row, field, "")).lower()
        if not text:
            continue
        for kw in keywords_lower:
            if kw in text:
                matching.append({"field": field, "keyword": kw})
                break  # one hit per field is enough

    # --- Missing evidence ---
    missing = []
    for field in COMPLETENESS_FIELDS:
        val = _get(row, field, "")
        if val == "" or (isinstance(val, str) and val.strip() == ""):
            missing.append(field)

    # --- Suspicious evidence ---
    suspicious = []

    # 1. Care need is claimed in specialties but not corroborated anywhere else.
    # Checked at the need level (any synonym counts as corroboration), not
    # per-exact-keyword, since "renal care" and "nephrology" etc. are
    # interchangeable claims about the same capability.
    specialties_text = str(_get(row, "specialties", "")).lower()
    other_text = " ".join(
        str(_get(row, f, "")) for f in ["description", "capability", "procedure", "equipment"]
    ).lower()
    specialty_claims_need = any(kw in specialties_text for kw in keywords_lower)
    corroborated_elsewhere = any(kw in other_text for kw in keywords_lower)
    if specialty_claims_need and not corroborated_elsewhere:
        suspicious.append(
            "This care need is listed under specialties but isn't mentioned in "
            "description/capability/procedure/equipment - worth confirming directly."
        )

    # 2. Matches the need, but reports zero doctors
    if matching:
        num_doctors = _get(row, "num_doctors", "")
        try:
            if num_doctors != "" and float(num_doctors) == 0:
                suspicious.append(
                    "Facility appears to match this care need but reports 0 doctors."
                )
        except (ValueError, TypeError):
            pass

    # 3. No source URL to verify any of the above
    source = _get(row, "source_urls", "")
    if source == "" or (isinstance(source, str) and source.strip() == ""):
        suspicious.append("No source URL provided - claims could not be traced to an origin.")

    return {"matching": matching, "missing": missing, "suspicious": suspicious}


def trust_label(evidence):
    """
    Collapse the evidence breakdown into a single human-scannable badge.
    Heuristic, not a statistical score - intended to draw the eye to
    facilities that need a closer look before referring a patient.
    """
    if evidence["suspicious"]:
        return "⚠️ Needs verification"
    if len(evidence["missing"]) >= 2:
        return "◐ Partial evidence"
    if evidence["matching"]:
        return "✓ Strong evidence"
    return "— No supporting evidence found"
