# src/nopin/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import logging

from nopin import NoPinocchio, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    provider: str  # Add provider info to response
    model: str  # Add model info to response


config_path = os.getenv("NOPIN_CONFIG", "configs/default.toml")
logger.info(f"Loading configuration from: {config_path}")

config = load_config(config_path)
logger.info("Configuration loaded successfully!")
logger.info(f"LLM Provider: {config.llm.client}")
logger.info(f"Model: {config.llm.model}")

np_service = NoPinocchio.from_config(config=config)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_confidence(request: AnalysisRequest):
    """Analyze confidence for a question."""
    try:
        logger.info(
            f"Starting analysis for question: {request.question[:50]}{'...' if len(request.question) > 50 else ''}"
        )

        result = np_service.analyze_question(question=request.question)
        confidence_score = result["confidence_score"]

        logger.info(" Analysis complete!")
        logger.info(f" Confidence Score: {confidence_score:.3f}")
        logger.info(f" Provider: {config.llm.client}")
        logger.info(f"Model: {config.llm.model}")

        return AnalysisResponse(
            question=result["question"],
            answer=result["answer"],
            confidence_score=confidence_score,
            timestamp=result["timestamp"],
            provider=config.llm.client,
            model=config.llm.model,
        )

    except Exception as e:
        logger.error(f" Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
