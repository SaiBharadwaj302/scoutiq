"""
api/routers/similarity.py

GET /v1/players/similar/{player_id}
Returns the top-N most similar players based on PCA embeddings.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import SimilarityResponse, SimilarPlayer
from api.dependencies import get_similarity_data
from models.similarity.query import find_similar, get_player_stats

router = APIRouter(prefix="/v1/players", tags=["Player Similarity"])


@router.get(
    "/similar/{player_id}",
    response_model=SimilarityResponse,
    summary="Find Similar Players",
)
async def get_similar_players(
    player_id: int,
    top_n: int = Query(default=5, ge=1, le=20, description="Number of similar players"),
    similarity_data=Depends(get_similarity_data),
):
    """
    Find the most tactically similar players to a given player.

    Uses PCA embeddings of per-90 stats and cosine similarity.
    Useful for scouting transfer targets with a similar profile.

    - **player_id**: StatsBomb player ID
    - **top_n**: number of results (1-20, default 5)
    """
    embeddings, player_ids, player_names = similarity_data

    if player_id not in player_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Player {player_id} not found. "
                   f"They may not have enough playing time in the dataset."
        )

    try:
        results = find_similar(
            player_id=player_id,
            top_n=top_n,
            embeddings=embeddings,
            player_ids=player_ids,
            player_names=player_names,
        )

        idx = player_ids.index(player_id)
        query_name = player_names[idx]

        return SimilarityResponse(
            query_player_id=player_id,
            query_player_name=str(query_name),
            similar_players=[
                SimilarPlayer(
                    player_id=r["player_id"],
                    player_name=str(r["player_name"]),
                    similarity=r["similarity"],
                )
                for r in results
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{player_id}", summary="Get Player Per-90 Stats")
async def get_player_per90(player_id: int):
    """
    Return per-90 aggregated stats for a player.
    """
    try:
        stats = get_player_stats(player_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return stats
