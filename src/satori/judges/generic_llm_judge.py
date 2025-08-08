import json
import re
from typing import Any, Dict, Optional, Union

from ..providers.base import BaseLLMProvider
from .base import BaseJudge, JudgeError


class GenericLLMJudge(BaseJudge):
  """Provider-agnostic judge that uses any `BaseLLMProvider` to score.

  The judge prompts the underlying model to return strict JSON with keys
  {"score": int 0-5, "explanation": str} and validates it.
  """

  def __init__(
    self,
    provider: BaseLLMProvider,
    low_temperature: float = 0.1,
    max_tokens: int = 1000,
    system_message: Optional[str] = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(**kwargs)
    self.provider = provider
    self.low_temperature = low_temperature
    self.max_tokens = max_tokens
    self.system_message = (
      system_message
      or "You are a precise evaluator that always responds with valid JSON."
    )

  async def score(
    self, input_text: str, expected_output: str, candidate_output: str, **kwargs
  ) -> Dict[str, Union[float, str]]:
    prompt = self._create_prompt(input_text, expected_output, candidate_output)
    try:
      response_text = await self.provider.generate(
        prompt,
        temperature=self.low_temperature,
        max_tokens=self.max_tokens,
        system_message=self.system_message,
      )
      parsed = self._parse_json_response(response_text)
      return self._validate_score_response(parsed)
    except Exception as e:
      raise self._handle_error(e)

  def _create_prompt(
    self, input_text: str, expected_output: str, candidate_output: str
  ) -> str:
    return (
      "You are an expert evaluator tasked with grading an AI model's response.\n\n"
      "## Task\n"
      "Compare the candidate output against the expected output for the given input.\n\n"
      "## Input\n"
      f"{input_text}\n\n"
      "## Expected Output\n"
      f"{expected_output}\n\n"
      "## Candidate Output\n"
      f"{candidate_output}\n\n"
      "## Evaluation Instructions\n"
      "Consider correctness, completeness, relevance, and clarity.\n\n"
      "## Scoring Scale\n"
      "0..5 (integers).\n\n"
      "## Output Format\n"
      'Return ONLY a JSON object with exactly: {"score": <0-5>, "explanation": <string>}'
    )

  def _parse_json_response(self, text: str) -> Dict[str, Any]:
    # Try direct JSON
    try:
      return json.loads(text)
    except Exception:
      pass

    # Try to extract the first JSON object substring
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
      candidate = match.group(0)
      return json.loads(candidate)

    raise JudgeError("Judge response was not valid JSON")

  def __repr__(self) -> str:
    return f"GenericLLMJudge(provider={self.provider})"
