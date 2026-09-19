"""Groq API client for Llama 3.3 70B.

Uses the official groq Python package as an optional LLM backend.
development and evaluation.

Usage:
    from llm.groq_client import GroqLLM

    llm = GroqLLM()
    answer = llm.generate("Explain type 2 diabetes.", system_prompt="Be concise.")
"""

from groq import Groq

from config.settings import GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_MODEL, GROQ_TEMPERATURE
from llm.base import BaseLLM


class GroqLLM(BaseLLM):
    """Groq-hosted Llama 3.3 70B client."""

    def __init__(
        self,
        model: str = GROQ_MODEL,
        api_key: str = GROQ_API_KEY,
    ) -> None:
        """Initialize the Groq client.

        Args:
            model: Model identifier on Groq.
            api_key: Groq API key (falls back to GROQ_API_KEY env var).
        """
        self.model = model
        self.client = Groq(api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = GROQ_MAX_TOKENS,
        temperature: float = GROQ_TEMPERATURE,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via the Groq chat completions API.

        Args:
            prompt: User message content.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.
            system_prompt: Optional system message.

        Returns:
            Generated text string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    llm = GroqLLM()
    result = llm.generate("What is type 2 diabetes? Answer in one sentence.")
    print(result)
