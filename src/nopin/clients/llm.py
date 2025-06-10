"""LLM client implementations."""
import os
import ollama
from dotenv import find_dotenv, load_dotenv
from mistralai import Mistral
from nopin.config.settings import Config


class LLMClient:
    """LLM client with support for multiple providers."""
    
    def __init__(self, *, config: Config):
        """Initialize LLM client.
        
        Args:
            config: Configuration object.
        """
        load_dotenv(find_dotenv())
        self._config = config
        self._client_name = config.llm.client
        self._model = config.llm.model
        self._temperature_config = config.llm.temperature.get(self._client_name)
        
        if self._client_name == "ollama":
            # Set Ollama host for Docker compatibility
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            
            # Configure ollama client
            ollama_client = ollama.Client(host=ollama_host)
            self._ollama_client = ollama_client
            
            try:
                response = self._ollama_client.list()
                available_models = [m['name'] for m in response['models']]
                if self._model not in available_models:
                    raise ValueError(
                        f"Model '{self._model}' not available. "
                        f"Available: {available_models}"
                    )
            except Exception as e:
                import warnings
                warnings.warn(f"Could not validate Ollama model '{self._model}': {e}")
        
        # Initialize client-specific objects
        if self._client_name == "mistral":
            api_key = os.environ.get(config.api.mistral_api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable {config.api.mistral_api_key_env} not set"
                )
            self._mistral_client = Mistral(api_key=api_key)

    def chat(self, *, question: str, temperature: str) -> str:
        """Generate a response to a question.
        
        Args:
            question: The input question.
            temperature: Temperature level ("min" or "max").
            
        Returns:
            Generated response text.
            
        Raises:
            ValueError: If temperature level is invalid.
        """
        if temperature not in ["min", "max"]:
            raise ValueError(f"Temperature must be 'min' or 'max', got: {temperature}")
        
        temp_value = getattr(self._temperature_config, temperature)
        
        if self._client_name == "mistral":
            return self._get_mistral_answer(question=question, temperature=temp_value)
        elif self._client_name == "ollama":
            return self._get_ollama_answer(question=question, temperature=temp_value)
        else:
            raise NotImplementedError(f"Client {self._client_name} not implemented")

    def _get_mistral_answer(self, *, question: str, temperature: float) -> str:
        """Get answer from Mistral API.
        
        Args:
            question: The input question.
            temperature: Sampling temperature.
            
        Returns:
            Generated response text.
        """
        chat_response = self._mistral_client.chat.complete(
            model=self._model,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": question,  # type: ignore
                },
            ],
        )
        return chat_response.choices[0].message.content  # type: ignore

    def _get_ollama_answer(self, *, question: str, temperature: float) -> str:
        """Get answer from Ollama.
        
        Args:
            question: The input question.
            temperature: Sampling temperature.
            
        Returns:
            Generated response text.
        """
        response = self._ollama_client.generate(
            model=self._model,
            prompt=question,
            options={
                "temperature": temperature,
            },
        )
        return response["response"]