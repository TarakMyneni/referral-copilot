"""
Vector Search RAG skill — uses databricks-sdk (no deprecated packages).

Degrades gracefully: returns {} if the index is unavailable, letting the
supervisor fall back to keyword-only ranking.
"""

VS_INDEX = "mediguide.referral_copilot.facilities_vector_index"

_w = None


def _get_client():
    global _w
    if _w is None:
        from databricks.sdk import WorkspaceClient
        _w = WorkspaceClient()
    return _w


def semantic_search(query_text: str, top_k: int = 200) -> dict:
    """
    Returns {facility_id: similarity_score} for the top_k closest matches.
    Returns {} on any error (graceful degradation).
    """
    if not query_text or not query_text.strip():
        return {}
    try:
        w    = _get_client()
        resp = w.vector_search_indexes.query_index(
            index_name  = VS_INDEX,
            query_text  = query_text.strip(),
            columns     = ["unique_id"],
            num_results = top_k,
        )
        scores = {}
        for row in (resp.result.data_array or []):
            if len(row) >= 2 and row[1] is not None:
                scores[str(row[0])] = float(row[1])
        return scores
    except Exception as e:
        print(f"[RAG] Vector search unavailable: {e}")
        return {}
