# NoPinocchio 🤥

AI Confidence Estimation System - Detect when your AI might be lying

## Motivation

Large language models often provide confident-sounding answers even when they're uncertain or incorrect. NoPinocchio implements the confidence estimation algorithm from ["Language models (mostly) know what they know"](https://arxiv.org/abs/2308.16175) to detect when AI responses might be unreliable.

The system combines self-reflection and consistency checking to provide confidence scores, helping you identify when to trust AI outputs.

## How to Use

**As an API:** Send questions to the REST API and receive answers with confidence scores for integration into your applications.

**As a Chat Interface:** Interactive web demo where you can ask questions and see real-time confidence estimates.

## Demo

![](assets/NoPinDemo.gif)

## Installation

```bash
git clone https://github.com/MaxJoas/no_pinochio.git
cd no_pinochio
```

### 🐳 Docker (Recommended)

**Quick Start:**
```bash
docker compose up --build
```

**Services:**
- **API**: http://localhost:8000
- **Chat Demo**: http://localhost:7860

**Individual Services:**
```bash
# API only
docker compose up api

# Chat demo only
docker compose up demo
```

## Configuration

NoPinocchio supports multiple LLM providers. Configure via `configs/default.toml`:

```toml
[llm]
client = "mistral"  # or "ollama"
model = "mistral-medium-latest"
```

**Supported Providers:**
- **Mistral AI**: Requires API key in `.env` file
- **Ollama**: Requires local Ollama installation and running service

*OpenAI and Claude support coming soon.*

**Environment Setup:**
Create `.env` file with:
```env
MISTRAL_API_KEY=your_api_key_here
```

For Ollama, ensure the service is running:
```bash
ollama serve
```

**Commands:**
```bash
# Start both services
docker compose up -d

# View logs  
docker compose logs -f

# Stop
docker compose down
```

### 🐍 Python

**Requirements:** [uv](https://docs.astral.sh/uv/)

```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Quick Start

**API Server:**
```bash
uvicorn nopin.api:app --reload
```

**Chat Interface:**
```bash
python src/nopin/demo/app.py
```

**CLI:**
```bash
nopinochio --question "What is the capital of France?"
```

**API Usage:**
```python
import requests

response = requests.post("http://localhost:8000/analyze", 
    json={"question": "What is the capital of France?"})

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Confidence: {data['confidence_score']:.2f}")
```

```bash
# curl example
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

## Use Cases

**E-commerce Chatbots:** Route low-confidence product questions to human agents instead of providing potentially incorrect information.

**Educational Applications:** Flag uncertain AI explanations so students know when to seek additional verification.

**Research Assistance:** Identify when AI-generated insights need fact-checking before inclusion in reports.

**Customer Support:** Escalate queries where the AI lacks confidence to ensure customer satisfaction.

## Evaluation Study

Coming soon - comprehensive evaluation on confidence estimation accuracy.

## References

(https://arxiv.org/abs/2308.16175)
