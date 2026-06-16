"""
Feedback store: reads/writes user_interactions Delta table.

Exposes:
  load(sdk_query_fn)                   — call once at app startup
  get_boost(care_need, facility_id)    — boost count for one facility
  get_all_boosts(care_need)            — {facility_id: count} for a need
  record_save(sdk_query_fn, ...)       — append a row + update local cache
"""

import re
from datetime import datetime, timezone

INTERACTIONS_TABLE = "mediguide.referral_copilot.user_interactions"

# {care_need: {facility_id: count}}
_cache: dict[str, dict[str, int]] = {}

_SAFE = re.compile(r"['\\\x00-\x1f]")   # chars to strip before SQL interpolation


def _sanitize(s: str) -> str:
    return _SAFE.sub("", str(s or ""))[:500]


def load(sdk_query_fn) -> None:
    """Read all 'saved' interactions into the in-memory cache."""
    global _cache
    _cache = {}
    try:
        cols, rows = sdk_query_fn(
            f"""
            SELECT care_need, facility_id, COUNT(*) AS cnt
            FROM {INTERACTIONS_TABLE}
            WHERE action = 'saved'
            GROUP BY care_need, facility_id
            """,
            wait="30s",
        )
        import pandas as pd
        df = pd.DataFrame(rows, columns=cols)
        for _, row in df.iterrows():
            need = str(row["care_need"])
            fid  = str(row["facility_id"])
            cnt  = int(row["cnt"])
            _cache.setdefault(need, {})[fid] = cnt
        total = sum(sum(v.values()) for v in _cache.values())
        print(f"[Feedback] Loaded {total} interactions across {len(_cache)} care needs.")
    except Exception as e:
        print(f"[Feedback] Could not load interactions (degrading to no-boost): {e}")


def get_boost(care_need: str, facility_id: str) -> int:
    return _cache.get(care_need, {}).get(facility_id, 0)


def get_all_boosts(care_need: str) -> dict:
    return dict(_cache.get(care_need, {}))


def record_save(sdk_query_fn, session_id: str, care_need: str,
                facility_id: str, facility_name: str) -> None:
    """Write one 'saved' row to Delta and update the local cache."""
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sid      = _sanitize(session_id)
    need     = _sanitize(care_need)
    fid      = _sanitize(facility_id)
    fname    = _sanitize(facility_name)

    try:
        sdk_query_fn(
            f"""
            INSERT INTO {INTERACTIONS_TABLE} VALUES (
              '{sid}', TIMESTAMP '{ts}', '{need}', '{fid}', '{fname}', 'saved'
            )
            """,
            wait="15s",
        )
    except Exception as e:
        print(f"[Feedback] Write failed (interaction not persisted): {e}")

    # Always update local cache so the boost is visible in the same session
    _cache.setdefault(need, {})
    _cache[need][fid] = _cache[need].get(fid, 0) + 1
