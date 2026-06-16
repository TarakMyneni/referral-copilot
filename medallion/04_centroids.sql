-- =============================================================================
-- Referral Copilot -- Pre-compute location centroids
-- Run AFTER 01_ingest_to_silver.sql
--
-- Creates one table: mediguide.referral_copilot.location_centroids
--   level    : city | district | division | region | state
--   name_key : lowercase normalised name (used for lookup)
--   lat / lon: average coordinate for that group
--
-- Priority order in the app (finer wins):
--   city > district > division > region > state
-- =============================================================================

CREATE OR REPLACE TABLE mediguide.referral_copilot.location_centroids
USING DELTA
COMMENT 'Pre-computed lat/lon centroids for location resolution'
AS

-- City centroids from facilities (most precise)
SELECT
  'city'                         AS level,
  LOWER(TRIM(city))              AS name_key,
  AVG(CAST(latitude  AS DOUBLE)) AS lat,
  AVG(CAST(longitude AS DOUBLE)) AS lon
FROM mediguide.referral_copilot.facilities_silver
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
  AND geo_valid = true
GROUP BY LOWER(TRIM(city))

UNION ALL

-- District centroids from facilities (supplements pincode data)
SELECT
  'district'                     AS level,
  LOWER(TRIM(district))          AS name_key,
  AVG(CAST(latitude  AS DOUBLE)) AS lat,
  AVG(CAST(longitude AS DOUBLE)) AS lon
FROM mediguide.referral_copilot.facilities_silver
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
  AND geo_valid = true
  AND district IS NOT NULL
GROUP BY LOWER(TRIM(district))

UNION ALL

-- District centroids from India Post pincode directory
SELECT
  'district'                    AS level,
  LOWER(TRIM(district))         AS name_key,
  AVG(CAST(latitude  AS DOUBLE)) AS lat,
  AVG(CAST(longitude AS DOUBLE)) AS lon
FROM mediguide.referral_copilot.pincode_silver
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
GROUP BY LOWER(TRIM(district))

UNION ALL

-- Division centroids
SELECT
  'division'                    AS level,
  LOWER(TRIM(division))         AS name_key,
  AVG(CAST(latitude  AS DOUBLE)) AS lat,
  AVG(CAST(longitude AS DOUBLE)) AS lon
FROM mediguide.referral_copilot.pincode_silver
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
GROUP BY LOWER(TRIM(division))

UNION ALL

-- Region centroids
SELECT
  'region'                      AS level,
  LOWER(TRIM(region))           AS name_key,
  AVG(CAST(latitude  AS DOUBLE)) AS lat,
  AVG(CAST(longitude AS DOUBLE)) AS lon
FROM mediguide.referral_copilot.pincode_silver
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
GROUP BY LOWER(TRIM(region))

UNION ALL

-- State centroids (coarsest fallback)
SELECT
  'state'                       AS level,
  LOWER(TRIM(state))            AS name_key,
  AVG(CAST(latitude  AS DOUBLE)) AS lat,
  AVG(CAST(longitude AS DOUBLE)) AS lon
FROM mediguide.referral_copilot.pincode_silver
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
GROUP BY LOWER(TRIM(state));

SELECT level, COUNT(*) AS cnt FROM mediguide.referral_copilot.location_centroids GROUP BY level ORDER BY level;
