# src/nopin/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os

from nopin import NoPinocchio, load_config

app = FastAPI(
    title="NoPinocchio API",
    description="AI Confidence Estimation Service",
    version="1.0.0",
)


class AnalysisRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    timestamp: str


config_path = os.getenv("NOPIN_CONFIG", "configs/default.toml")
config = load_config(config_path)
np_service = NoPinocchio.from_config(config=config)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_confidence(request: AnalysisRequest):
    """Analyze confidence for a question."""
    try:
        result = np_service.analyze_question(question=request.question)

        confidence_score = result["confidence_score"]

        return AnalysisResponse(
            question=result["question"],
            answer=result["answer"],
            confidence_score=confidence_score,
            timestamp=result["timestamp"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
