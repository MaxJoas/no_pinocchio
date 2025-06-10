from typing import Dict, Union, ClassVar, List
from pathlib import Path
import tomli  # type: ignore
from pydantic import BaseModel, Field, field_validator, model_validator


class TemperatureConfig(BaseModel):
    """Temperature configuration for different clients."""

    min: float = Field(ge=0.0, le=2.0)
    max: float = Field(ge=0.0, le=2.0)

    @field_validator("max")
    @classmethod
    def max_must_be_greater_than_min(cls, v, info):
        if info.data.get("min") is not None and v < info.data["min"]:
            raise ValueError("max temperature must be >= min temperature")
        return v


class LLMConfig(BaseModel):
    """LLM client configuration."""

    client: str = Field(default="mistral")
    model: str = Field(default="mistral-medium-latest")
    temperature: Dict[str, TemperatureConfig] = Field(default_factory=dict)

    SUPPORTED_COMBINATIONS: ClassVar[Dict[str, Union[List[str], str]]] = {
        "mistral": [
            "mistral-medium-latest",
            "mistral-large-latest",
            "mistral-small-latest",
        ],
        "ollama": "dynamic",  # Ollama models are user-installed, validated at runtime
    }

    @field_validator("client")
    @classmethod
    def validate_client(cls, v):
        if v not in cls.SUPPORTED_COMBINATIONS:
            raise ValueError(
                f"Unsupported client: {v}. "
                f"Supported: {list(cls.SUPPORTED_COMBINATIONS.keys())}"
            )
        return v

    @model_validator(mode="after")
    def validate_model_client_combination(self):
        client = self.client
        model = self.model

        if client == "mistral":
            supported_models = self.SUPPORTED_COMBINATIONS.get(client, [])
            if model not in supported_models:
                raise ValueError(
                    f"Model '{model}' not supported for Mistral client. "
                    f"Supported models: {supported_models}"
                )
        elif client == "ollama":
            # For Ollama, we'll validate at runtime when actually using the model
            # since it depends on what the user has installed locally
            pass

        return self


class NLIConfig(BaseModel):
    """Natural Language Inference configuration."""

    model: str = Field(default="nli-deberta-v3-small")
    device: int = Field(default=-1)

    SUPPORTED_MODELS: ClassVar[List[str]] = [
        "nli-deberta-v3-small",
        "nli-deberta-v3-base",
    ]

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported NLI model: {v}. Supported: {cls.SUPPORTED_MODELS}"
            )
        return v


class NoPinocchioConfig(BaseModel):
    """Core algorithm configuration."""

    k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.5, ge=0.0, le=1.0)


class APIConfig(BaseModel):
    """API configuration. Name of the Environment Variable

    DONT PUT THE API KEY HERE!!!!!!!!!!!
    """

    mistral_api_key_env: str = Field(default="MISTRAL_API_KEY")
    timeout: int = Field(default=300, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Config(BaseModel):
    """Main configuration class."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    nli: NLIConfig = Field(default_factory=NLIConfig)
    nopinocchio: NoPinocchioConfig = Field(default_factory=NoPinocchioConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load and validate configuration from TOML file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Validated configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "rb") as f:
        config_data = tomli.load(f)

    return Config(**config_data)
