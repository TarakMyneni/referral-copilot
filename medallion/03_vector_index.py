# =============================================================================
# Referral Copilot -- Vector Search setup (databricks-sdk only)
# Run as a Databricks notebook AFTER 01_ingest_to_silver.sql has completed.
# =============================================================================

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    VectorIndexType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)

w = WorkspaceClient()

VS_ENDPOINT  = "referral_copilot_vs"
VS_INDEX     = "mediguide.referral_copilot.facilities_vector_index"
SOURCE_TABLE = "mediguide.referral_copilot.facilities_silver"
EMBED_MODEL  = "databricks-gte-large-en"

# ---------------------------------------------------------------------------
# Step 1: Create endpoint (skip if already exists)
# ---------------------------------------------------------------------------
try:
    w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
    print(f"Endpoint '{VS_ENDPOINT}' already exists — skipping create.")
except Exception:
    print(f"Creating endpoint '{VS_ENDPOINT}' …")
    w.vector_search_endpoints.create_endpoint(
        name          = VS_ENDPOINT,
        endpoint_type = EndpointType.STANDARD,
    )
    for _ in range(40):
        ep    = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
        state = (ep.endpoint_status.state.value
                 if ep.endpoint_status and ep.endpoint_status.state
                 else "PROVISIONING")
        print(f"  endpoint state: {state}")
        if state == "ONLINE":
            break
        time.sleep(20)
    else:
        raise RuntimeError("Endpoint did not reach ONLINE within timeout.")

# ---------------------------------------------------------------------------
# Step 2: Create Delta Sync index on search_text (skip if already exists)
# ---------------------------------------------------------------------------
try:
    w.vector_search_indexes.get_index(index_name=VS_INDEX)
    print(f"Index '{VS_INDEX}' already exists — skipping create.")
except Exception:
    print(f"Creating index '{VS_INDEX}' …")
    w.vector_search_indexes.create_index(
        name          = VS_INDEX,
        endpoint_name = VS_ENDPOINT,
        primary_key   = "unique_id",
        index_type    = VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec = DeltaSyncVectorIndexSpecRequest(
            source_table = SOURCE_TABLE,
            pipeline_type = PipelineType.TRIGGERED,
            embedding_source_columns = [
                EmbeddingSourceColumn(
                    name                          = "search_text",
                    embedding_model_endpoint_name = EMBED_MODEL,
                )
            ],
        ),
    )
    print("Index creation started. Waiting for ready …")
    for _ in range(40):
        idx   = w.vector_search_indexes.get_index(index_name=VS_INDEX)
        ready = idx.status.ready_for_query if idx.status else False
        print(f"  ready_for_query: {ready}")
        if ready:
            break
        time.sleep(20)
    else:
        print("Warning: index not ready within timeout — sync manually and retry.")

# ---------------------------------------------------------------------------
# Step 3: Wait for index to be ready, then trigger sync
# ---------------------------------------------------------------------------
print("Waiting for index to be ready …")
for _ in range(60):
    idx = w.vector_search_indexes.get_index(index_name=VS_INDEX)
    status = idx.status
    ready  = bool(status.ready) if status else False
    msg    = status.message[:80] if status and status.message else ""
    print(f"  ready={ready}  rows={getattr(status,'indexed_row_count',0)}  {msg}")
    if ready:
        break
    time.sleep(20)
else:
    print("Index still not ready after 20 min — check Vector Search UI and retry sync manually.")
    raise SystemExit(1)

print("Triggering sync …")
w.vector_search_indexes.sync_index(index_name=VS_INDEX)
print("Sync triggered (~5-10 min to complete).")

# ---------------------------------------------------------------------------
# Step 4: Smoke test
# ---------------------------------------------------------------------------
print("\nSmoke test — querying 'dialysis kidney renal care' …")
resp = w.vector_search_indexes.query_index(
    index_name  = VS_INDEX,
    query_text  = "dialysis kidney renal care",
    columns     = ["unique_id", "name", "city"],
    num_results = 3,
)
for row in (resp.result.data_array or []):
    print(" ", row)
print("✓ Done.")
