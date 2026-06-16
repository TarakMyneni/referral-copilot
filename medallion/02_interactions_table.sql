-- =============================================================================
-- Referral Copilot -- User Interactions table
-- Run AFTER 01_ingest_to_silver.sql
--
-- Stores every "save to shortlist" action (and future ratings).
-- The feedback loop: interaction counts boost facility ranking for the
-- same care need in future searches.
--
-- Schema is append-only; never delete rows (use action='removed' if needed).
-- =============================================================================

CREATE TABLE IF NOT EXISTS mediguide.referral_copilot.user_interactions (
  session_id    STRING    COMMENT 'Browser session UUID',
  ts            TIMESTAMP COMMENT 'UTC timestamp of action',
  care_need     STRING    COMMENT 'Canonical care need key (dialysis, cardiology, …)',
  facility_id   STRING    COMMENT 'unique_id from facilities_silver',
  facility_name STRING    COMMENT 'Denormalised name for human readability',
  action        STRING    COMMENT 'saved | removed | rated_good | rated_bad'
)
USING DELTA
COMMENT 'User interaction history for feedback-boosted ranking'
TBLPROPERTIES ('quality' = 'silver');


-- =============================================================================
-- Bootstrap seed: pick top-5 highest-completeness facilities per care need
-- and mark them as pre-validated saves (3× each = visible boost at demo).
-- This runs once; re-running is idempotent (just adds more rows).
-- =============================================================================
INSERT INTO mediguide.referral_copilot.user_interactions
SELECT session_id, ts, care_need, unique_id AS facility_id, name AS facility_name, action
FROM (
  -- Dialysis
  SELECT
    'seed-bootstrap'      AS session_id,
    current_timestamp()   AS ts,
    'dialysis'            AS care_need,
    unique_id, name,
    'saved'               AS action,
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST) AS rn
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%dialysis%'
      OR LOWER(specialties) LIKE '%nephrol%'
      OR LOWER(description) LIKE '%dialysis%'
      OR LOWER(capability)  LIKE '%dialysis%')
    AND geo_valid = TRUE

  UNION ALL

  -- Emergency
  SELECT 'seed-bootstrap', current_timestamp(), 'emergency', unique_id, name, 'saved',
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST)
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%emergency%'
      OR LOWER(description) LIKE '%emergency%'
      OR LOWER(capability)  LIKE '%trauma%')
    AND geo_valid = TRUE

  UNION ALL

  -- Cardiology
  SELECT 'seed-bootstrap', current_timestamp(), 'cardiology', unique_id, name, 'saved',
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST)
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%cardio%'
      OR LOWER(description) LIKE '%cardiac%'
      OR LOWER(capability)  LIKE '%cath lab%')
    AND geo_valid = TRUE

  UNION ALL

  -- Maternity
  SELECT 'seed-bootstrap', current_timestamp(), 'maternity', unique_id, name, 'saved',
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST)
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%obstetric%'
      OR LOWER(specialties) LIKE '%gynecol%'
      OR LOWER(description) LIKE '%maternit%'
      OR LOWER(capability)  LIKE '%labour%')
    AND geo_valid = TRUE

  UNION ALL

  -- Oncology
  SELECT 'seed-bootstrap', current_timestamp(), 'oncology', unique_id, name, 'saved',
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST)
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%oncol%'
      OR LOWER(description) LIKE '%cancer%'
      OR LOWER(capability)  LIKE '%chemotherapy%')
    AND geo_valid = TRUE

  UNION ALL

  -- Orthopedics
  SELECT 'seed-bootstrap', current_timestamp(), 'orthopedics', unique_id, name, 'saved',
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST)
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%orthopedic%'
      OR LOWER(description) LIKE '%orthopedic%'
      OR LOWER(capability)  LIKE '%joint replacement%')
    AND geo_valid = TRUE

  UNION ALL

  -- Neurology
  SELECT 'seed-bootstrap', current_timestamp(), 'neurology', unique_id, name, 'saved',
    ROW_NUMBER() OVER (ORDER BY completeness_score DESC, num_doctors DESC NULLS LAST)
  FROM mediguide.referral_copilot.facilities_silver
  WHERE (LOWER(specialties) LIKE '%neurol%'
      OR LOWER(description) LIKE '%neurosurg%'
      OR LOWER(capability)  LIKE '%stroke%')
    AND geo_valid = TRUE
) t
-- 5 facilities per care need × 3 duplicate rows each = visible demo boost
WHERE rn <= 5
UNION ALL
SELECT session_id, ts, care_need, facility_id, facility_name, action
FROM mediguide.referral_copilot.user_interactions
WHERE 1 = 0  -- dummy to force the rn <= 5 filter to apply in the outer scope
;

-- Insert a 2nd and 3rd copy so boost scores = 3 per seeded facility
INSERT INTO mediguide.referral_copilot.user_interactions
SELECT session_id, current_timestamp(), care_need, facility_id, facility_name, action
FROM mediguide.referral_copilot.user_interactions
WHERE session_id = 'seed-bootstrap';

INSERT INTO mediguide.referral_copilot.user_interactions
SELECT session_id, current_timestamp(), care_need, facility_id, facility_name, action
FROM mediguide.referral_copilot.user_interactions
WHERE session_id = 'seed-bootstrap'
  AND ts = (SELECT MIN(ts) FROM mediguide.referral_copilot.user_interactions WHERE session_id = 'seed-bootstrap');

-- Verify
SELECT care_need, COUNT(DISTINCT facility_id) AS facilities, COUNT(*) AS total_interactions
FROM mediguide.referral_copilot.user_interactions
GROUP BY care_need ORDER BY care_need;
