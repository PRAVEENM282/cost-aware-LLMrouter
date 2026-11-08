"""
LLM Gateway Semantic Router API.
Exposes POST /api/v1/route for intelligent query routing and cost metering.
"""

from contextlib import asynccontextmanager
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.router import SemanticRouter
from app.tools import (
    evaluate_calculator,
    query_revenue_db,
    lookup_faq_cache,
    query_small_llm,
    query_large_llm,
)
from app.cost_meter import calculate_cost

# Pareto-optimal threshold default determined from empirical sweep
OPTIMAL_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.65"))
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# Global router instance loaded on startup
router: Optional[SemanticRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    if router is None:
        router = SemanticRouter(model_name=MODEL_NAME)
    yield


app = FastAPI(
    title="LLM Gateway Semantic Router",
    version="1.0.0",
    description="Cost-Aware Semantic Router for Intelligent LLM Dispatch",
    lifespan=lifespan,
)


class RouteRequest(BaseModel):
    query: str = Field(..., description="The user's input query")
    threshold: Optional[float] = Field(
        default=None,
        description="Confidence threshold for semantic routing (defaults to optimal)",
    )


class RouteResponse(BaseModel):
    answer: str = Field(..., description="The generated or retrieved response")
    path_taken: str = Field(..., description="The execution path taken (calculator|sql|cache|small_llm|large_llm)")
    cost: float = Field(..., description="Cost of the query in USD")


@app.get("/")
@app.get("/health")
def health_check():
    """Healthcheck endpoint for docker-compose and monitoring."""
    return {"status": "healthy", "model": MODEL_NAME, "optimal_threshold": OPTIMAL_THRESHOLD}


@app.post("/api/v1/route", response_model=RouteResponse)
def route_query(request: RouteRequest):
    """
    Intelligently dispatches query to deterministic tools, small models, or large LLM
    based on embedding similarity and confidence threshold.
    """
    global router
    if router is None:
        router = SemanticRouter(model_name=MODEL_NAME)

    threshold = request.threshold if request.threshold is not None else OPTIMAL_THRESHOLD

    # Determine route
    category, path_taken, similarity = router.route_details(request.query, threshold)

    # Execute path
    if path_taken == "calculator":
        answer = evaluate_calculator(request.query)
    elif path_taken == "sql":
        answer = query_revenue_db(request.query)
    elif path_taken == "cache":
        answer = lookup_faq_cache(request.query)
    elif path_taken == "small_llm":
        answer = query_small_llm(request.query)
    elif path_taken == "large_llm":
        answer = query_large_llm(request.query)
    else:
        # Default safety fallback
        path_taken = "large_llm"
        answer = query_large_llm(request.query)

    # Calculate exact cost
    cost = calculate_cost(path_taken, request.query, answer)

    return RouteResponse(
        answer=answer,
        path_taken=path_taken,
        cost=cost,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
