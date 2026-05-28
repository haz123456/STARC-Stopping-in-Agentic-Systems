import functools
import logging
import time
from typing import Any, Dict, Optional


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def format_chat(message: str, system_message: str = "You are a helpful assistant.") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]


def call_api(func, limit: int = 5, pause: int = 10):
    retries = 0
    while True:
        try:
            return func()
        except Exception as exc:
            message = str(exc).lower()
            logger.info("Exception while using api: %s", exc)
            if "rate limit" in message or "rate_limit" in message or "quota" in message or "429" in message:
                logger.info("Rate limit exceeded, waiting %s seconds and retrying", pause)
                time.sleep(pause)
                continue
            if retries < limit:
                retries += 1
                logger.info("Retrying after error (%s/%s)", retries, limit)
                continue
            logger.info("Skipping generation after repeated errors")
            return None


class OpenAIModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        generation_max_length: int = 512,
    ) -> None:
        import openai

        if "azure/" in model_name:
            self.client = openai.AzureOpenAI()
            self.model_name = model_name.split("/", 1)[1]
        else:
            self.client = openai.OpenAI()
            self.model_name = model_name

        self.temperature = temperature
        self.generation_max_length = generation_max_length

    def generate(
        self,
        prompt: str,
        system_message: str = "You are a careful answer judge. Return JSON only.",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        messages = format_chat(prompt, system_message=system_message)
        func = functools.partial(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=self.generation_max_length,
            temperature=self.temperature,
            **kwargs,
        )
        response = call_api(func)
        if response is None or response.choices[0].message.content is None:
            return None
        return {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "input_text": messages,
        }
