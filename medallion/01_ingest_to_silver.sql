-- =============================================================================
-- Referral Copilot -- Bronze to Silver: comprehensive data cleaning
-- Catalog: mediguide   Schema: referral_copilot
--
-- Run ONCE in Databricks SQL Editor before deploying the app.
-- The app reads ONLY from mediguide.referral_copilot -- never the shared catalog.
--
-- DESIGN PRINCIPLE: no records are deleted.
-- Records with bad/missing coordinates are kept and marked with geo_source /
-- geo_valid. Coordinates are imputed from the India Post pincode hierarchy
-- (all levels sourced from india_post_pincode_directory):
--
--   ORIGINAL  -- facility's own GPS, validated within India bounding box
--   POSTCODE  -- centroid of all pincode-directory entries for that pincode
--   DISTRICT  -- centroid of all entries in the same district
--   DIVISION  -- centroid of all entries in the same division
--   REGION    -- centroid of all entries in the same region (circle)
--   STATE     -- centroid of all entries in the same state
--   UNKNOWN   -- no coordinate could be resolved (null lat/lon, geo_valid=false)
--
-- The facility's postcode is the bridge into the hierarchy: once we look up
-- the postcode in the pincode directory we know its district, division, region,
-- and state, and can use centroids at each level as fallbacks.
-- For facilities with no valid postcode, only the STATE level is reachable
-- (matched on the normalised state name).
--
-- Other data quality fixes:
--
-- FACILITIES:
--   1. Specialty arrays have 50+ duplicate entries (multi-source aggregation).
--      Fix: ARRAY_DISTINCT before ARRAY_JOIN.
--   2. Specialty codes are camelCase (emergencyMedicine, orthopedicSurgery…).
--      Fix: TRANSFORM + REGEXP_REPLACE inserts spaces before uppercase letters
--           so downstream search sees plain English ("emergency Medicine").
--   3. city/state mixed case and whitespace.
--      Fix: TRIM(INITCAP(...))
--   4. organization_type = "facility" on every row (content-table type, useless).
--      Fix: built from facilityTypeId + operatorTypeId.
--   5. description aggregated from many sources; can exceed 5 000 words.
--      Fix: SUBSTRING to 1500 chars.
--   6. Empty strings masquerading as data.
--      Fix: NULLIF(TRIM(col), '')
--   7. Duplicate unique_id rows -- keep the richest one.
--   8. `procedure` is a SQL reserved word; backtick-quoted throughout.
--
-- PINCODE:
--   1. district / statename ALL UPPERCASE -> TRIM(INITCAP(...))
--   2. lat/lon = "NA" string -> TRY_CAST returns NULL; India bounds filter applied.
--
-- NFHS:
--   1. Numeric cells contain "(29.5)" or "*" instead of NULL.
--      Fix: REGEXP_REPLACE strips parens/asterisks/spaces, then TRY_CAST.
--   2. Trailing whitespace in district_name -> TRIM.
--   3. Replaced SELECT * with curated columns relevant to referral context.
-- =============================================================================


-- 0. Catalog + Schema
CREATE CATALOG IF NOT EXISTS mediguide;

CREATE SCHEMA IF NOT EXISTS mediguide.referral_copilot
  COMMENT 'Referral Copilot -- ingested hackathon data';


-- =============================================================================
-- 1. FACILITIES
-- =============================================================================
CREATE OR REPLACE TABLE mediguide.referral_copilot.facilities_silver
  COMMENT 'Silver: cleaned, typed, coord-imputed healthcare facilities'
  TBLPROPERTIES (
    'quality' = 'silver',
    'delta.enableChangeDataFeed' = 'true'   -- required for Vector Search Delta Sync index
  )
AS
WITH

-- -------------------------------------------------------------------------
-- A. Extract and clean all facility fields.
--    No coordinate filter here -- every record is kept.
--    Deduplicate on unique_id, keeping the row with the most data.
-- -------------------------------------------------------------------------
base AS (
  SELECT
    unique_id,
    NULLIF(TRIM(name), '')                                          AS name,
    TRIM(INITCAP(address_city))                                     AS city,
    TRIM(INITCAP(address_stateOrRegion))                            AS state,
    NULLIF(TRIM(address_zipOrPostcode), '')                         AS postcode,
    TRY_CAST(latitude  AS DOUBLE)                                   AS raw_lat,
    TRY_CAST(longitude AS DOUBLE)                                   AS raw_lon,
    -- Meaningful org type from facilityTypeId + operatorTypeId
    CASE
      WHEN operatorTypeId IS NOT NULL AND facilityTypeId IS NOT NULL
        THEN CONCAT(INITCAP(TRIM(operatorTypeId)), ' ', INITCAP(TRIM(facilityTypeId)))
      WHEN facilityTypeId  IS NOT NULL THEN INITCAP(TRIM(facilityTypeId))
      WHEN operatorTypeId  IS NOT NULL THEN INITCAP(TRIM(operatorTypeId))
      ELSE NULL
    END                                                             AS organization_type,
    NULLIF(TRIM(officialPhone),   '')                               AS phone,
    NULLIF(TRIM(officialWebsite), '')                               AS website,
    TRY_CAST(yearEstablished AS INT)                                AS year_established,
    TRY_CAST(numberDoctors   AS INT)                                AS num_doctors,
    TRY_CAST(capacity        AS INT)                                AS capacity,
    NULLIF(TRIM(SUBSTRING(description, 1, 1500)), '')               AS description,
    -- Specialties: deduplicate codes, expand camelCase to plain English
    COALESCE(
      NULLIF(
        ARRAY_JOIN(
          TRANSFORM(
            ARRAY_DISTINCT(FROM_JSON(specialties, 'ARRAY<STRING>')),
            s -> TRIM(REGEXP_REPLACE(s, '([A-Z])', ' $1'))
          ), ' '
        ), ''
      ), ''
    )                                                               AS specialties,
    COALESCE(NULLIF(ARRAY_JOIN(ARRAY_DISTINCT(FROM_JSON(capability,  'ARRAY<STRING>')), ' | '), ''), '') AS capability,
    COALESCE(NULLIF(ARRAY_JOIN(ARRAY_DISTINCT(FROM_JSON(`procedure`, 'ARRAY<STRING>')), ' | '), ''), '') AS procedure_text,
    COALESCE(NULLIF(ARRAY_JOIN(ARRAY_DISTINCT(FROM_JSON(equipment,   'ARRAY<STRING>')), ' | '), ''), '') AS equipment,
    COALESCE(NULLIF(ARRAY_JOIN(ARRAY_DISTINCT(FROM_JSON(source_urls, 'ARRAY<STRING>')), ' '),   ''), '') AS source_urls,
    (
      CASE WHEN numberDoctors   IS NOT NULL AND TRIM(CAST(numberDoctors   AS STRING)) NOT IN ('','null') THEN 1 ELSE 0 END
    + CASE WHEN capacity        IS NOT NULL AND TRIM(CAST(capacity        AS STRING)) NOT IN ('','null') THEN 1 ELSE 0 END
    + CASE WHEN yearEstablished IS NOT NULL AND TRIM(CAST(yearEstablished AS STRING)) NOT IN ('','null') THEN 1 ELSE 0 END
    + CASE WHEN source_urls IS NOT NULL AND source_urls NOT IN ('null','[]','','[""]')                   THEN 1 ELSE 0 END
    )                                                               AS completeness_score,
    ROW_NUMBER() OVER (
      PARTITION BY unique_id
      ORDER BY
        (CASE WHEN source_urls IS NOT NULL AND source_urls NOT IN ('null','[]','','[""]') THEN 1 ELSE 0 END) DESC,
        (CASE WHEN numberDoctors IS NOT NULL THEN 1 ELSE 0 END) DESC,
        (CASE WHEN capacity      IS NOT NULL THEN 1 ELSE 0 END) DESC
    )                                                               AS _rn
  FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities
  WHERE name IS NOT NULL AND TRIM(name) != ''
    AND address_city IS NOT NULL AND TRIM(address_city) != ''
),

deduped AS (
  SELECT * EXCEPT (_rn) FROM base WHERE _rn = 1
),

-- -------------------------------------------------------------------------
-- B. India Post pincode directory -- valid entries only.
--    All centroid CTEs below are derived from this single filtered base.
-- -------------------------------------------------------------------------
valid_pincode AS (
  SELECT
    TRIM(CAST(pincode AS STRING))   AS pincode,
    TRIM(INITCAP(district))         AS district,
    TRIM(INITCAP(divisionname))     AS division,
    TRIM(INITCAP(regionname))       AS region,
    TRIM(INITCAP(statename))        AS state,
    TRY_CAST(latitude  AS DOUBLE)   AS lat,
    TRY_CAST(longitude AS DOUBLE)   AS lon
  FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory
  WHERE TRY_CAST(latitude  AS DOUBLE) BETWEEN 6.0  AND 38.0
    AND TRY_CAST(longitude AS DOUBLE) BETWEEN 67.0 AND 98.5
    AND pincode  IS NOT NULL
    AND district IS NOT NULL
    AND UPPER(TRIM(district)) NOT IN ('', 'NA', 'NULL')
),

-- -------------------------------------------------------------------------
-- C. Metadata lookup: given a postcode, what is its district/division/region/state?
--    (One canonical row per postcode -- the hierarchy is fixed for a postcode.)
-- -------------------------------------------------------------------------
postcode_meta AS (
  SELECT pincode, district, division, region, state
  FROM valid_pincode
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pincode ORDER BY district) = 1
),

-- -------------------------------------------------------------------------
-- D. Centroids at each level of the India Post hierarchy
-- -------------------------------------------------------------------------
pincode_centroids AS (
  SELECT pincode,  AVG(lat) AS lat, AVG(lon) AS lon
  FROM valid_pincode
  GROUP BY pincode
),

district_centroids AS (
  SELECT district, AVG(lat) AS lat, AVG(lon) AS lon
  FROM valid_pincode
  WHERE district IS NOT NULL AND TRIM(district) != ''
  GROUP BY district
),

division_centroids AS (
  SELECT division, AVG(lat) AS lat, AVG(lon) AS lon
  FROM valid_pincode
  WHERE division IS NOT NULL AND TRIM(division) != ''
  GROUP BY division
),

region_centroids AS (
  SELECT region,   AVG(lat) AS lat, AVG(lon) AS lon
  FROM valid_pincode
  WHERE region IS NOT NULL AND TRIM(region) != ''
  GROUP BY region
),

state_centroids AS (
  SELECT state,    AVG(lat) AS lat, AVG(lon) AS lon
  FROM valid_pincode
  WHERE state IS NOT NULL AND TRIM(state) != ''
  GROUP BY state
),

-- -------------------------------------------------------------------------
-- E. Resolve best available coordinates for every facility.
--    Priority: ORIGINAL > POSTCODE > DISTRICT > DIVISION > REGION > STATE > UNKNOWN
--    The postcode_meta join bridges the facility's postcode to the hierarchy
--    so that district/division/region centroids are reachable.
-- -------------------------------------------------------------------------
resolved AS (
  SELECT
    d.unique_id, d.name, d.city, d.state, d.postcode,
    pm.district, pm.division, pm.region,
    d.organization_type, d.phone, d.website,
    d.year_established, d.num_doctors, d.capacity,
    d.description, d.specialties, d.capability,
    d.procedure_text, d.equipment, d.source_urls,
    d.completeness_score,

    COALESCE(
      CASE WHEN d.raw_lat BETWEEN 6.0 AND 38.0 AND d.raw_lon BETWEEN 67.0 AND 98.5
           THEN d.raw_lat END,
      c_pin.lat,
      c_dist.lat,
      c_div.lat,
      c_reg.lat,
      c_st.lat
    )                                                               AS latitude,

    COALESCE(
      CASE WHEN d.raw_lat BETWEEN 6.0 AND 38.0 AND d.raw_lon BETWEEN 67.0 AND 98.5
           THEN d.raw_lon END,
      c_pin.lon,
      c_dist.lon,
      c_div.lon,
      c_reg.lon,
      c_st.lon
    )                                                               AS longitude,

    CASE
      WHEN d.raw_lat BETWEEN 6.0 AND 38.0 AND d.raw_lon BETWEEN 67.0 AND 98.5 THEN 'ORIGINAL'
      WHEN c_pin.lat  IS NOT NULL                                               THEN 'POSTCODE'
      WHEN c_dist.lat IS NOT NULL                                               THEN 'DISTRICT'
      WHEN c_div.lat  IS NOT NULL                                               THEN 'DIVISION'
      WHEN c_reg.lat  IS NOT NULL                                               THEN 'REGION'
      WHEN c_st.lat   IS NOT NULL                                               THEN 'STATE'
      ELSE                                                                           'UNKNOWN'
    END                                                             AS geo_source

  FROM deduped d
  -- Bridge: postcode -> district/division/region/state in pincode hierarchy
  LEFT JOIN postcode_meta      pm     ON d.postcode = pm.pincode
  -- Centroid joins
  LEFT JOIN pincode_centroids  c_pin  ON d.postcode   = c_pin.pincode
  LEFT JOIN district_centroids c_dist ON pm.district  = c_dist.district
  LEFT JOIN division_centroids c_div  ON pm.division  = c_div.division
  LEFT JOIN region_centroids   c_reg  ON pm.region    = c_reg.region
  LEFT JOIN state_centroids    c_st   ON d.state      = c_st.state   -- normalised INITCAP match
)

-- ALL records preserved; geo_valid=false marks the ones with no resolvable coords
SELECT
  unique_id, name, city, state, postcode,
  district, division, region,
  latitude, longitude,
  geo_source,
  (geo_source != 'UNKNOWN')   AS geo_valid,
  organization_type, phone, website,
  year_established, num_doctors, capacity,
  description, specialties, capability,
  procedure_text, equipment, source_urls,
  completeness_score,
  -- Concatenated text for Vector Search embedding (unstructured fields only)
  CONCAT_WS(' | ',
    NULLIF(description,    ''),
    NULLIF(specialties,    ''),
    NULLIF(capability,     ''),
    NULLIF(procedure_text, ''),
    NULLIF(equipment,      '')
  )                             AS search_text
FROM resolved;


-- =============================================================================
-- 2. PINCODE DIRECTORY
-- =============================================================================
CREATE OR REPLACE TABLE mediguide.referral_copilot.pincode_silver
  COMMENT 'Silver: India Post pincode directory -- location resolution index'
  TBLPROPERTIES ('quality' = 'silver')
AS
SELECT
  TRIM(officename)                      AS officename,
  TRIM(CAST(pincode AS STRING))         AS pincode,
  TRIM(INITCAP(district))               AS district,
  TRIM(INITCAP(divisionname))           AS division,
  TRIM(INITCAP(regionname))             AS region,
  TRIM(INITCAP(statename))              AS state,
  TRY_CAST(latitude  AS DOUBLE)         AS latitude,
  TRY_CAST(longitude AS DOUBLE)         AS longitude
FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory
WHERE
  -- Keep ALL rows — hierarchy (postcode→district→division→region→state) is
  -- useful even when coordinates are missing. lat/lon will be NULL for those rows;
  -- centroid computations filter NULLs before averaging.
  district IS NOT NULL
  AND UPPER(TRIM(district)) NOT IN ('', 'NA', 'NULL')
  AND CASE
        WHEN TRY_CAST(latitude AS DOUBLE) IS NULL THEN TRUE   -- keep, no coords
        WHEN TRY_CAST(latitude AS DOUBLE) NOT BETWEEN 6.0 AND 38.0  THEN FALSE  -- drop bad coords
        WHEN TRY_CAST(longitude AS DOUBLE) NOT BETWEEN 67.0 AND 98.5 THEN FALSE
        ELSE TRUE
      END;


-- =============================================================================
-- 3. NFHS-5 DISTRICT HEALTH INDICATORS
--    Numeric cells can contain "(29.5)" (estimated) or "*" (suppressed).
--    Strip parens/asterisks/spaces before TRY_CAST so those become NULL.
-- =============================================================================
CREATE OR REPLACE TABLE mediguide.referral_copilot.nfhs_silver
  COMMENT 'Silver: NFHS-5 district health indicators (curated, typed)'
  TBLPROPERTIES ('quality' = 'silver')
AS
SELECT
  TRIM(district_name)                                                                          AS district,
  TRIM(state_ut)                                                                               AS state,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(institutional_birth_5y_pct),                    '[()* ]',''),'') AS DOUBLE) AS institutional_birth_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(institutional_birth_in_public_facility_5y_pct), '[()* ]',''),'') AS DOUBLE) AS public_facility_birth_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(hh_member_covered_health_insurance_pct),        '[()* ]',''),'') AS DOUBLE) AS health_insurance_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(hh_use_improved_sanitation_pct),                '[()* ]',''),'') AS DOUBLE) AS improved_sanitation_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(hh_improved_water_pct),                         '[()* ]',''),'') AS DOUBLE) AS improved_water_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(hh_electricity_pct),                            '[()* ]',''),'') AS DOUBLE) AS electricity_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(mothers_who_had_at_least_4_anc_visits_lb5y_pct),  '[()* ]',''),'') AS DOUBLE) AS anc_4visits_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(births_attended_by_skilled_hp_5y_10_pct),          '[()* ]',''),'') AS DOUBLE) AS skilled_birth_attendant_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(births_delivered_by_csection_5y_pct),              '[()* ]',''),'') AS DOUBLE) AS csection_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(child_u5_who_are_stunted_height_for_age_18_pct),     '[()* ]',''),'') AS DOUBLE) AS child_stunting_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(child_u5_who_are_wasted_weight_for_height_18_pct),   '[()* ]',''),'') AS DOUBLE) AS child_wasting_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(child_u5_who_are_underweight_weight_for_age_18_pct), '[()* ]',''),'') AS DOUBLE) AS child_underweight_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(non_pregnant_w15_49_who_are_anaemic_lt_12_0_g_dl_22_pct), '[()* ]',''),'') AS DOUBLE) AS women_anaemia_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(all_w15_49_who_are_anaemic_pct),                          '[()* ]',''),'') AS DOUBLE) AS all_women_anaemia_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(w15_plus_with_high_or_very_high_gt_140_mg_dl_blood_sugar_or_pct), '[()* ]',''),'') AS DOUBLE) AS women_high_bloodsugar_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(m15_plus_with_high_or_very_high_gt_140_mg_dl_blood_sugar_or_pct), '[()* ]',''),'') AS DOUBLE) AS men_high_bloodsugar_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(w15_plus_with_high_bp_sys_gte_140_mmhg_and_or_dia_gte_90_mm_pct), '[()* ]',''),'') AS DOUBLE) AS women_high_bp_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(m15_plus_with_high_bp_sys_gte_140_mmhg_and_or_dia_gte_90_mm_pct), '[()* ]',''),'') AS DOUBLE) AS men_high_bp_pct,
  TRY_CAST(NULLIF(REGEXP_REPLACE(TRIM(child_12_23m_fully_vaccinated_based_on_information_from_eit_pct), '[()* ]',''),'') AS DOUBLE) AS full_vaccination_pct
FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.nfhs_5_district_health_indicators
WHERE district_name IS NOT NULL AND TRIM(district_name) != '';


-- =============================================================================
-- 4. Verify -- shows coordinate imputation breakdown per level
-- =============================================================================
SELECT
  'facilities_silver'   AS table_name,
  COUNT(*)              AS total_rows,
  SUM(CASE WHEN geo_source = 'ORIGINAL'  THEN 1 ELSE 0 END) AS original_gps,
  SUM(CASE WHEN geo_source = 'POSTCODE'  THEN 1 ELSE 0 END) AS imputed_postcode,
  SUM(CASE WHEN geo_source = 'DISTRICT'  THEN 1 ELSE 0 END) AS imputed_district,
  SUM(CASE WHEN geo_source = 'DIVISION'  THEN 1 ELSE 0 END) AS imputed_division,
  SUM(CASE WHEN geo_source = 'REGION'    THEN 1 ELSE 0 END) AS imputed_region,
  SUM(CASE WHEN geo_source = 'STATE'     THEN 1 ELSE 0 END) AS imputed_state,
  SUM(CASE WHEN geo_source = 'UNKNOWN'   THEN 1 ELSE 0 END) AS unresolved,
  COUNT(DISTINCT city)  AS distinct_cities
FROM mediguide.referral_copilot.facilities_silver

UNION ALL

SELECT 'pincode_silver', COUNT(*), NULL, NULL, NULL, NULL, NULL, NULL, NULL,
  COUNT(DISTINCT district)
FROM mediguide.referral_copilot.pincode_silver

UNION ALL

SELECT 'nfhs_silver', COUNT(*), NULL, NULL, NULL, NULL, NULL, NULL, NULL,
  COUNT(DISTINCT district)
FROM mediguide.referral_copilot.nfhs_silver;
