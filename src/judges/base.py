from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class BaseJudge(ABC):
  """Base class for all LLM judges."""

  def __init__(self, **kwargs):
    """Initialize judge with configuration.

    Args:
        **kwargs: Judge-specific configuration options
    """
    self.config = kwargs

  @abstractmethod
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
    pass

  def score_sync(
    self, input_text: str, expected_output: str, candidate_output: str, **kwargs
  ) -> Dict[str, Union[float, str]]:
    """Synchronous wrapper for score method.

    Args:
        input_text: The original input/prompt
        expected_output: The reference output
        candidate_output: The output to evaluate
        **kwargs: Additional scoring parameters

    Returns:
        Dictionary with 'score' and 'explanation' keys
    """
    import asyncio

    return asyncio.run(
      self.score(input_text, expected_output, candidate_output, **kwargs)
    )

  def _validate_score_response(
    self, response: Dict[str, Any]
  ) -> Dict[str, Union[float, str]]:
    """Validate that the score response has the required format.

    Args:
        response: The response dictionary to validate

    Returns:
        Validated response dictionary

    Raises:
        JudgeError: If the response format is invalid
    """
    if not isinstance(response, dict):
      raise JudgeError("Score response must be a dictionary")

    if "score" not in response:
      raise JudgeError("Score response must contain 'score' key")

    if "explanation" not in response:
      raise JudgeError("Score response must contain 'explanation' key")

    score = response["score"]
    if not isinstance(score, (int, float)) or not (0 <= score <= 5):
      raise JudgeError("Score must be a number between 0 and 5")

    if not isinstance(response["explanation"], str):
      raise JudgeError("Explanation must be a string")

    return {"score": float(score), "explanation": response["explanation"]}

  def _handle_error(self, error: Exception) -> Exception:
    """Handle and transform judge-specific errors.

    Args:
        error: The original error from the judge

    Returns:
        Transformed error with consistent interface
    """
    return JudgeError(f"Judge {self.__class__.__name__} failed: {str(error)}")


class JudgeError(Exception):
  """Exception raised when a judge operation fails."""

  pass
