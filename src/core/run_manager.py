import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from tenacity import (
  retry,
  retry_if_exception_type,
  stop_after_attempt,
  wait_exponential,
)

from ..io.csv_loader import CSVLoader
from ..judges.base import BaseJudge, JudgeError
from ..providers.base import BaseLLMProvider, ProviderError


@dataclass
class EvaluationResult:
  """Result of evaluating a single test case."""

  input_text: str
  expected_output: str
  candidate_output: str
  score: float
  explanation: str
  provider_name: str
  judge_name: str
  execution_time: float
  error: Optional[str] = None
  timestamp: datetime = field(default_factory=datetime.now)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "input": self.input_text,
      "expected": self.expected_output,
      "candidate": self.candidate_output,
      "score": self.score,
      "explanation": self.explanation,
      "provider": self.provider_name,
      "judge": self.judge_name,
      "execution_time": self.execution_time,
      "error": self.error,
      "timestamp": self.timestamp.isoformat() if self.timestamp else None,
    }


@dataclass
class BatchResults:
  """Results from a batch evaluation run."""

  results: List[EvaluationResult]
  total_time: float
  success_count: int
  failure_count: int
  average_score: float
  median_score: float
  std_score: float

  def to_dict(self) -> Dict[str, Any]:
    return {
      "results": [r.to_dict() for r in self.results],
      "summary": {
        "total_time": self.total_time,
        "success_count": self.success_count,
        "failure_count": self.failure_count,
        "average_score": self.average_score,
        "median_score": self.median_score,
        "std_score": self.std_score,
        "total_cases": len(self.results),
      },
    }


class RunManager:
  """Manages the execution of LLM evaluation runs."""

  def __init__(
    self,
    provider: BaseLLMProvider,
    judge: BaseJudge,
    max_concurrent: int = 5,
    retry_attempts: int = 3,
    rate_limit_delay: float = 1.0,
    fail_fast: bool = True,
  ):
    """Initialize the run manager.

    Args:
        provider: The LLM provider to generate responses
        judge: The judge to evaluate responses
        max_concurrent: Maximum concurrent requests (default: 5)
        retry_attempts: Number of retry attempts for failed requests
        rate_limit_delay: Delay between requests to avoid rate limiting
        fail_fast: If True, stop execution on first provider error (default: True)
    """
    self.provider = provider
    self.judge = judge
    self.max_concurrent = max_concurrent
    self.retry_attempts = retry_attempts
    self.rate_limit_delay = rate_limit_delay
    self.fail_fast = fail_fast

    self.semaphore = asyncio.Semaphore(max_concurrent)

    self.total_requests = 0
    self.successful_requests = 0
    self.failed_requests = 0

  async def run_batch(
    self, csv_path: str, progress_callback: Optional[callable] = None
  ) -> BatchResults:
    """Run evaluation on a batch of test cases from a CSV file.

    Args:
        csv_path: Path to the CSV file with test cases
        progress_callback: Optional callback for progress updates

    Returns:
        BatchResults object containing all evaluation results
    """
    start_time = time.time()

    loader = CSVLoader(csv_path)
    loader.load()
    test_cases = loader.get_rows()

    tasks = []
    for idx, test_case in enumerate(test_cases):
      task = self._evaluate_single_with_semaphore(
        test_case, idx, len(test_cases), progress_callback
      )
      tasks.append(task)

    if self.fail_fast:
      # When fail_fast is True, don't catch exceptions - let them propagate
      results = await asyncio.gather(*tasks)
      evaluation_results = results
    else:
      # When fail_fast is False, catch exceptions and convert to error results
      results = await asyncio.gather(*tasks, return_exceptions=True)
      evaluation_results = []
      for result in results:
        if isinstance(result, Exception):
          # Check for critical errors that should always stop execution
          if self._is_critical_error(result):
            if hasattr(self.provider, "close"):
              try:
                await self.provider.close()
              except Exception:
                pass
            raise result

          evaluation_results.append(
            EvaluationResult(
              input_text="Error",
              expected_output="Error",
              candidate_output="Error",
              score=0.0,
              explanation="Evaluation failed",
              provider_name=str(self.provider),
              judge_name=str(self.judge),
              execution_time=0.0,
              error=str(result),
            )
          )
        else:
          evaluation_results.append(result)

    if hasattr(self.provider, "close"):
      try:
        await self.provider.close()
      except Exception:
        pass

    successful_results = [r for r in evaluation_results if r.error is None]
    scores = [r.score for r in successful_results]

    total_time = time.time() - start_time

    return BatchResults(
      results=evaluation_results,
      total_time=total_time,
      success_count=len(successful_results),
      failure_count=len(evaluation_results) - len(successful_results),
      average_score=sum(scores) / len(scores) if scores else 0.0,
      median_score=self._calculate_median(scores),
      std_score=self._calculate_std(scores),
    )

  async def _evaluate_single_with_semaphore(
    self,
    test_case: Dict[str, Any],
    index: int,
    total: int,
    progress_callback: Optional[callable],
  ) -> EvaluationResult:
    """Evaluate a single test case with semaphore control.

    Args:
        test_case: Dictionary containing input and expected_output
        index: Index of the current test case
        total: Total number of test cases
        progress_callback: Optional callback for progress updates

    Returns:
        EvaluationResult for the test case
    """
    async with self.semaphore:
      if self.rate_limit_delay > 0 and index > 0:
        await asyncio.sleep(self.rate_limit_delay)

      result = await self._evaluate_single_with_retry(test_case)

      if progress_callback:
        progress_callback(index + 1, total, result)

      return result

  @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ProviderError, JudgeError)),
    reraise=True,
  )
  async def _evaluate_single_with_retry(
    self, test_case: Dict[str, Any]
  ) -> EvaluationResult:
    """Evaluate a single test case with retry logic.

    Args:
        test_case: Dictionary containing input and expected_output

    Returns:
        EvaluationResult for the test case
    """
    start_time = time.time()

    input_text = test_case.get("input", "")
    expected_output = test_case.get("expected_output", "")

    try:
      candidate_output = await self.provider.generate(input_text)

      judge_result = await self.judge.score(
        input_text=input_text,
        expected_output=expected_output,
        candidate_output=candidate_output,
      )

      execution_time = time.time() - start_time

      return EvaluationResult(
        input_text=input_text,
        expected_output=expected_output,
        candidate_output=candidate_output,
        score=float(judge_result["score"]),
        explanation=judge_result["explanation"],
        provider_name=str(self.provider),
        judge_name=str(self.judge),
        execution_time=execution_time,
        error=None,
      )

    except (ProviderError, JudgeError) as e:
      if self.fail_fast:
        raise e

      execution_time = time.time() - start_time

      return EvaluationResult(
        input_text=input_text,
        expected_output=expected_output,
        candidate_output="",
        score=0.0,
        explanation=f"Evaluation failed: {str(e)}",
        provider_name=str(self.provider),
        judge_name=str(self.judge),
        execution_time=execution_time,
        error=str(e),
      )
    except Exception as e:
      execution_time = time.time() - start_time

      return EvaluationResult(
        input_text=input_text,
        expected_output=expected_output,
        candidate_output="",
        score=0.0,
        explanation=f"Unexpected error: {str(e)}",
        provider_name=str(self.provider),
        judge_name=str(self.judge),
        execution_time=execution_time,
        error=str(e),
      )

  async def run_single(
    self, input_text: str, expected_output: str
  ) -> EvaluationResult:
    """Run evaluation on a single test case.

    Args:
        input_text: The input prompt
        expected_output: The expected output

    Returns:
        EvaluationResult for the test case
    """
    test_case = {"input": input_text, "expected_output": expected_output}

    return await self._evaluate_single_with_retry(test_case)

  def _calculate_median(self, scores: List[float]) -> float:
    """Calculate median of scores.

    Args:
        scores: List of scores

    Returns:
        Median score
    """
    if not scores:
      return 0.0

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    if n % 2 == 0:
      return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    else:
      return sorted_scores[n // 2]

  def _calculate_std(self, scores: List[float]) -> float:
    """Calculate standard deviation of scores.

    Args:
        scores: List of scores

    Returns:
        Standard deviation
    """
    if not scores or len(scores) < 2:
      return 0.0

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    return variance**0.5

  def _is_critical_error(self, error: Exception) -> bool:
    """Check if an error is critical and should stop execution regardless of fail_fast setting.

    Args:
        error: The exception to check

    Returns:
        True if the error is critical (API key, authentication, etc.)
    """
    error_str = str(error).lower()

    # Critical authentication/API key errors
    critical_patterns = [
      "invalid api key",
      "invalid openai api key",
      "api key not found",
      "authentication failed",
      "unauthorized",
      "invalid anthropic api key",
      "invalid google api key",
    ]

    return any(pattern in error_str for pattern in critical_patterns)

  def __repr__(self) -> str:
    """String representation of the run manager."""
    return (
      f"RunManager(provider={self.provider}, judge={self.judge}, "
      f"max_concurrent={self.max_concurrent})"
    )
