import json
import os
from typing import Any, Dict, Optional, Union

from openai import AsyncOpenAI, OpenAI

from .base import BaseJudge, JudgeError


class OpenAIJudge(BaseJudge):
  """OpenAI-based judge for evaluating LLM responses."""

  def __init__(
    self, model: str = "gpt-4.1", api_key: Optional[str] = None, **kwargs
  ):
    """Initialize OpenAI judge.

    Args:
        model: Model to use for judging (default: gpt-4.1)
        api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var
        **kwargs: Additional configuration options
    """
    super().__init__(**kwargs)

    self.model = model

    # Get API key from parameter or environment
    self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not self.api_key:
      raise JudgeError(
        "OpenAI API key not found. Please provide it as a parameter "
        "or set the OPENAI_API_KEY environment variable."
      )

    # Extract OpenAI-specific configuration
    self.base_url = kwargs.get("base_url")
    self.organization = kwargs.get("organization")
    self.timeout = kwargs.get("timeout", 60.0)
    self.max_retries = kwargs.get("max_retries", 3)

    # Initialize async client (sync client created on demand)
    self.async_client = AsyncOpenAI(
      api_key=self.api_key,
      base_url=self.base_url,
      organization=self.organization,
      timeout=self.timeout,
    )
    self._sync_client: Optional[OpenAI] = None

  @property
  def sync_client(self) -> OpenAI:
    """Lazy initialization of sync client."""
    if self._sync_client is None:
      self._sync_client = OpenAI(
        api_key=self.api_key,
        base_url=self.base_url,
        organization=self.organization,
        timeout=self.timeout,
      )
    return self._sync_client

  async def score(
    self, input_text: str, expected_output: str, candidate_output: str, **kwargs
  ) -> Dict[str, Union[float, str]]:
    """Score a candidate output against the expected output.

    Args:
        input_text: The original input/prompt that generated the outputs
        expected_output: The reference/ground truth output
        candidate_output: The output to be evaluated
        **kwargs: Additional scoring parameters

    Returns:
        Dictionary with 'score' (float 0-5) and 'explanation' (str) keys

    Raises:
        JudgeError: If the scoring operation fails
    """
    prompt = self._create_judge_prompt(
      input_text, expected_output, candidate_output
    )

    # Try to get a valid response with retries
    for attempt in range(self.max_retries):
      try:
        response = await self._get_judge_response(prompt)
        parsed_response = self._parse_judge_response(response)
        return self._validate_score_response(parsed_response)
      except (json.JSONDecodeError, KeyError, ValueError) as e:
        if attempt == self.max_retries - 1:
          raise JudgeError(
            f"Failed to get valid judge response after {self.max_retries} attempts: {str(e)}"
          )
        continue
      except Exception as e:
        raise self._handle_error(e)

    # This should never be reached due to the raise in the loop, but satisfies type checker
    raise JudgeError("Unexpected error in score method")

  def score_sync(
    self, input_text: str, expected_output: str, candidate_output: str, **kwargs
  ) -> Dict[str, Union[float, str]]:
    """Synchronous wrapper for score method using sync client."""
    prompt = self._create_judge_prompt(
      input_text, expected_output, candidate_output
    )

    # Try to get a valid response with retries
    for attempt in range(self.max_retries):
      try:
        response = self._get_judge_response_sync(prompt)
        parsed_response = self._parse_judge_response(response)
        return self._validate_score_response(parsed_response)
      except (json.JSONDecodeError, KeyError, ValueError) as e:
        if attempt == self.max_retries - 1:
          raise JudgeError(
            f"Failed to get valid judge response after {self.max_retries} attempts: {str(e)}"
          )
        continue
      except Exception as e:
        raise self._handle_error(e)

    # This should never be reached due to the raise in the loop, but satisfies type checker
    raise JudgeError("Unexpected error in score_sync method")

  def _create_judge_prompt(
    self, input_text: str, expected_output: str, candidate_output: str
  ) -> str:
    """Create the prompt for the judge model using Chain-of-Thought reasoning.

    Args:
        input_text: The original input
        expected_output: The expected output
        candidate_output: The candidate output to evaluate

    Returns:
        The formatted judge prompt
    """
    prompt = f"""You are an expert evaluator tasked with grading the quality of an AI model's response.

## Task
Compare the candidate output against the expected output for the given input. Use Chain-of-Thought reasoning to analyze the response quality.

## Input
{input_text}

## Expected Output
{expected_output}

## Candidate Output
{candidate_output}

## Evaluation Instructions
1. First, analyze what the input is asking for
2. Compare the candidate output with the expected output
3. Consider these aspects:
   - Correctness: Is the information accurate?
   - Completeness: Does it cover all required points?
   - Relevance: Does it directly address the input?
   - Clarity: Is it well-structured and clear?
   - Additional value: Does it provide useful extra information?

## Scoring Scale
- 0: Completely wrong, off-topic, or harmful
- 1: Substantially incorrect with major errors
- 2: Partially correct but missing key information
- 3: Mostly correct with some notable issues
- 4: Almost perfect with minor issues
- 5: Fully correct, comprehensive, and well-reasoned

## Output Format
Provide your evaluation as a JSON object with exactly this structure:
{{
    "reasoning": "Step-by-step analysis of the response quality",
    "score": <integer from 0 to 5>,
    "explanation": "Brief summary of why this score was given"
}}

Think through your evaluation step-by-step before providing the final JSON response."""

    return prompt

  async def _get_judge_response(self, prompt: str) -> str:
    """Get response from the judge model asynchronously.

    Args:
        prompt: The judge prompt

    Returns:
        The model's response
    """
    try:
      response = await self.async_client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "system",
            "content": "You are a precise evaluator that always responds with valid JSON.",
          },
          {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # Low temperature for consistent judging
        max_tokens=1000,
        response_format={"type": "json_object"},  # Ensure JSON response
      )

      if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content or "{}"
      else:
        raise JudgeError("Empty response from judge model")

    except Exception as e:
      if "api_key" in str(e).lower():
        raise JudgeError("Invalid OpenAI API key for judge")
      elif "rate_limit" in str(e).lower() or "429" in str(e):
        raise JudgeError(
          "OpenAI rate limit exceeded for judge. Please wait and retry."
        )
      else:
        raise e

  def _get_judge_response_sync(self, prompt: str) -> str:
    """Get response from the judge model synchronously.

    Args:
        prompt: The judge prompt

    Returns:
        The model's response
    """
    try:
      response = self.sync_client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "system",
            "content": "You are a precise evaluator that always responds with valid JSON.",
          },
          {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # Low temperature for consistent judging
        max_tokens=1000,
        response_format={"type": "json_object"},  # Ensure JSON response
      )

      if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content or "{}"
      else:
        raise JudgeError("Empty response from judge model")

    except Exception as e:
      if "api_key" in str(e).lower():
        raise JudgeError("Invalid OpenAI API key for judge")
      elif "rate_limit" in str(e).lower() or "429" in str(e):
        raise JudgeError(
          "OpenAI rate limit exceeded for judge. Please wait and retry."
        )
      else:
        raise e

  def _parse_judge_response(self, response: str) -> Dict[str, Any]:
    """Parse the JSON response from the judge.

    Args:
        response: The JSON string response

    Returns:
        Parsed dictionary with score and explanation

    Raises:
        json.JSONDecodeError: If response is not valid JSON
        KeyError: If required keys are missing
    """
    parsed = json.loads(response)

    # Extract score and explanation (handling different response formats)
    if "score" in parsed and "explanation" in parsed:
      return {"score": parsed["score"], "explanation": parsed["explanation"]}
    elif "score" in parsed and "reasoning" in parsed:
      # Use reasoning as explanation if explanation is missing
      return {
        "score": parsed["score"],
        "explanation": parsed.get("explanation", parsed["reasoning"]),
      }
    else:
      raise KeyError(
        "Response missing required 'score' and 'explanation' fields"
      )

  def __repr__(self) -> str:
    """String representation of the judge."""
    return f"OpenAIJudge(model='{self.model}')"
