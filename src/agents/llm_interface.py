import os
import json
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.1"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096


class LLMInterface:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        key = api_key or os.environ.get("OPENAI_API_KEY", "placeholder")
        url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(api_key=key, base_url=url, _enforce_credentials=False)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        system_with_instruction = (
            (system_prompt or "")
            + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just JSON."
        )
        raw = self.generate(prompt, system_prompt=system_with_instruction)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nRaw: {raw[:500]}")
            raise

    def generate_hypothesis(
        self,
        symbolic_results: dict,
        dataset_info: dict,
        iteration: int,
    ) -> dict:
        system_prompt = (
            "You are a scientific equation discovery assistant. "
            "Given the current symbolic regression results and dataset information, "
            "formulate hypotheses for auxiliary variables or transformations that "
            "could improve the model. Suggest new feature combinations, "
            "symmetries to test, or variable substitutions."
        )
        prompt = (
            f"Iteration: {iteration}\n\n"
            f"Symbolic regression results:\n{json.dumps(symbolic_results, indent=2)}\n\n"
            f"Dataset info:\n{json.dumps(dataset_info, indent=2)}\n\n"
            "Formulate hypotheses for auxiliary variables or transformations. "
            "Return JSON with keys: 'hypotheses' (list of dicts with 'name', 'expression', 'rationale'), "
            "'suggested_lib' (list of symbolic functions to add), "
            "'suggested_architecture' (optional dict with 'width' list)."
        )
        return self.generate_json(prompt, system_prompt=system_prompt)

    def interpret_results(
        self,
        discovered_equations: list,
        dataset_info: dict,
        metrics: dict,
    ) -> str:
        system_prompt = (
            "You are a scientific interpreter. Given discovered equations, "
            "dataset information, and validation metrics, provide a clear "
            "narrative explanation of what was discovered and its significance."
        )
        prompt = (
            f"Discovered equations:\n{json.dumps(discovered_equations, indent=2)}\n\n"
            f"Dataset info:\n{json.dumps(dataset_info, indent=2)}\n\n"
            f"Validation metrics:\n{json.dumps(metrics, indent=2)}\n\n"
            "Provide a detailed scientific interpretation of these findings. "
            "Explain the mathematical relationships, physical significance, "
            "and any novel insights discovered."
        )
        return self.generate(prompt, system_prompt=system_prompt)
