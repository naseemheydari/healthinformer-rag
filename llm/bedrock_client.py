"""AWS Bedrock client for Claude models via the Converse API.

Uses boto3's converse() endpoint which supports all current and future
Bedrock models without needing model-specific API version strings.

Usage:
    from llm.bedrock_client import BedrockLLM

    llm = BedrockLLM()
    answer = llm.generate("Explain type 2 diabetes.", system_prompt="Be concise.")
"""

import boto3

from config.settings import (
    AWS_REGION,
    BEDROCK_MAX_TOKENS,
    BEDROCK_MODEL_ID,
    BEDROCK_TEMPERATURE,
)
from llm.base import BaseLLM


class BedrockLLM(BaseLLM):
    """AWS Bedrock client using the Converse API."""

    def __init__(
        self,
        model_id: str = BEDROCK_MODEL_ID,
        region: str = AWS_REGION,
    ) -> None:
        """Initialize the Bedrock runtime client.

        Args:
            model_id: Bedrock model or inference profile ID.
            region: AWS region for the Bedrock endpoint.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def generate(
        self,
        prompt: str,
        max_tokens: int = BEDROCK_MAX_TOKENS,
        temperature: float = BEDROCK_TEMPERATURE,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via Bedrock Converse API.

        Args:
            prompt: User message content.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.
            system_prompt: Optional system message.

        Returns:
            Generated text string.
        """
        kwargs: dict = {
            "modelId": self.model_id,
            "messages": [
                {"role": "user", "content": [{"text": prompt}]},
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]


if __name__ == "__main__":
    llm = BedrockLLM()
    result = llm.generate("What is type 2 diabetes? Answer in one sentence.")
    print(result)
