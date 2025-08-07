"""Result writer module for saving evaluation results in various formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..core.run_manager import BatchResults


class ResultWriter:
  """Write evaluation results to various output formats."""

  def __init__(self, output_path: Optional[str] = None):
    """Initialize result writer.

    Args:
        output_path: Optional path for output file. If not provided,
                    results will be printed to console.
    """
    self.output_path = Path(output_path) if output_path else None

  def write(self, batch_results: BatchResults, format: str = "csv") -> None:
    """Write batch results to file or console.

    Args:
        batch_results: The batch results to write
        format: Output format ('csv', 'json', 'jsonl')
    """
    if format == "csv":
      self.write_csv(batch_results)
    elif format == "json":
      self.write_json(batch_results)
    elif format == "jsonl":
      self.write_jsonl(batch_results)
    else:
      raise ValueError(f"Unsupported format: {format}")

  def write_csv(self, batch_results: BatchResults) -> None:
    """Write results to CSV format.

    Args:
        batch_results: The batch results to write
    """
    # Prepare data for DataFrame
    data = []
    for result in batch_results.results:
      data.append(
        {
          "input": result.input_text,
          "expected": result.expected_output,
          "candidate": result.candidate_output,
          "score": result.score,
          "explanation": result.explanation,
          "provider": result.provider_name,
          "judge": result.judge_name,
          "execution_time": f"{result.execution_time:.2f}",
          "error": result.error or "",
          "timestamp": result.timestamp.isoformat() if result.timestamp else "",
        }
      )

    df = pd.DataFrame(data)

    if self.output_path:
      # Ensure parent directory exists
      self.output_path.parent.mkdir(parents=True, exist_ok=True)

      # Save to CSV
      df.to_csv(self.output_path, index=False)
      print(f"Results saved to {self.output_path}")
    else:
      # Print to console
      print("\n" + df.to_string())

  def write_json(self, batch_results: BatchResults) -> None:
    """Write results to JSON format.

    Args:
        batch_results: The batch results to write
    """
    output_data = {
      "metadata": {
        "total_cases": len(batch_results.results),
        "success_count": batch_results.success_count,
        "failure_count": batch_results.failure_count,
        "total_time": batch_results.total_time,
        "timestamp": datetime.now().isoformat(),
      },
      "statistics": {
        "average_score": batch_results.average_score,
        "median_score": batch_results.median_score,
        "std_deviation": batch_results.std_score,
        "score_distribution": self._calculate_score_distribution(batch_results),
      },
      "results": [result.to_dict() for result in batch_results.results],
    }

    if self.output_path:
      # Ensure parent directory exists
      self.output_path.parent.mkdir(parents=True, exist_ok=True)

      # Save to JSON
      with open(self.output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
      print(f"Results saved to {self.output_path}")
    else:
      # Print to console
      print(json.dumps(output_data, indent=2, default=str))

  def write_jsonl(self, batch_results: BatchResults) -> None:
    """Write results to JSONL (JSON Lines) format.

    Args:
        batch_results: The batch results to write
    """
    if self.output_path:
      # Ensure parent directory exists
      self.output_path.parent.mkdir(parents=True, exist_ok=True)

      # Save to JSONL
      with open(self.output_path, "w") as f:
        # Write metadata as first line
        metadata = {
          "type": "metadata",
          "total_cases": len(batch_results.results),
          "success_count": batch_results.success_count,
          "failure_count": batch_results.failure_count,
          "average_score": batch_results.average_score,
          "median_score": batch_results.median_score,
          "std_deviation": batch_results.std_score,
          "total_time": batch_results.total_time,
          "timestamp": datetime.now().isoformat(),
        }
        f.write(json.dumps(metadata, default=str) + "\n")

        # Write each result as a separate line
        for result in batch_results.results:
          result_data = {"type": "result", **result.to_dict()}
          f.write(json.dumps(result_data, default=str) + "\n")

      print(f"Results saved to {self.output_path}")
    else:
      # Print to console
      metadata = {
        "type": "metadata",
        "total_cases": len(batch_results.results),
        "success_count": batch_results.success_count,
        "failure_count": batch_results.failure_count,
        "average_score": batch_results.average_score,
        "median_score": batch_results.median_score,
        "std_deviation": batch_results.std_score,
        "total_time": batch_results.total_time,
      }
      print(json.dumps(metadata, default=str))

      for result in batch_results.results:
        result_data = {"type": "result", **result.to_dict()}
        print(json.dumps(result_data, default=str))

  def print_summary(self, batch_results: BatchResults) -> None:
    """Print a summary of the results to console.

    Args:
        batch_results: The batch results to summarize
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal test cases: {len(batch_results.results)}")
    print(f"Successful: {batch_results.success_count}")
    print(f"Failed: {batch_results.failure_count}")
    print(
      f"Success rate: {(batch_results.success_count / len(batch_results.results) * 100):.1f}%"
    )

    print("\nScore Statistics:")
    print(f"  Average: {batch_results.average_score:.2f}")
    print(f"  Median: {batch_results.median_score:.2f}")
    print(f"  Std Dev: {batch_results.std_score:.2f}")

    # Score distribution
    distribution = self._calculate_score_distribution(batch_results)
    print("\nScore Distribution:")
    for score, count in distribution.items():
      percentage = (
        (count / batch_results.success_count * 100)
        if batch_results.success_count > 0
        else 0
      )
      bar = "█" * int(percentage / 2)  # Scale to 50 chars max
      print(f"  Score {score}: {bar} {count} ({percentage:.1f}%)")

    print(f"\nTotal execution time: {batch_results.total_time:.2f} seconds")
    avg_time = (
      batch_results.total_time / len(batch_results.results)
      if batch_results.results
      else 0
    )
    print(f"Average time per case: {avg_time:.2f} seconds")

    # Show worst performing cases
    if batch_results.success_count > 0:
      worst_cases = sorted(
        [r for r in batch_results.results if r.error is None],
        key=lambda x: x.score,
      )[:3]

      if worst_cases:
        print("\nLowest Scoring Cases:")
        for i, result in enumerate(worst_cases, 1):
          print(f"\n  {i}. Score: {result.score}")
          print(
            f"     Input: {result.input_text[:100]}{'...' if len(result.input_text) > 100 else ''}"
          )
          print(
            f"     Explanation: {result.explanation[:150]}{'...' if len(result.explanation) > 150 else ''}"
          )

    print("\n" + "=" * 60)

  def _calculate_score_distribution(
    self, batch_results: BatchResults
  ) -> Dict[int, int]:
    """Calculate the distribution of scores.

    Args:
        batch_results: The batch results to analyze

    Returns:
        Dictionary mapping score values to counts
    """
    distribution = {i: 0 for i in range(6)}

    for result in batch_results.results:
      if result.error is None:
        score_int = int(result.score)
        if 0 <= score_int <= 5:
          distribution[score_int] += 1

    return distribution

  def save_detailed_report(
    self, batch_results: BatchResults, report_path: str
  ) -> None:
    """Save a detailed HTML report of the evaluation.

    Args:
        batch_results: The batch results to report
        report_path: Path for the HTML report file
    """
    # This is a bonus feature - creates an HTML report
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Satori Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: white; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
                th {{ background: #f5f5f5; }}
                .score-0 {{ background: #ffcccc; }}
                .score-1 {{ background: #ffd9cc; }}
                .score-2 {{ background: #ffe6cc; }}
                .score-3 {{ background: #fff3cc; }}
                .score-4 {{ background: #e6ffcc; }}
                .score-5 {{ background: #ccffcc; }}
            </style>
        </head>
        <body>
            <h1>Satori Evaluation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="stats">
                    <div class="stat-box">
                        <strong>Total Cases:</strong> {len(batch_results.results)}
                    </div>
                    <div class="stat-box">
                        <strong>Average Score:</strong> {batch_results.average_score:.2f}
                    </div>
                    <div class="stat-box">
                        <strong>Success Rate:</strong> {(batch_results.success_count / len(batch_results.results) * 100):.1f}%
                    </div>
                    <div class="stat-box">
                        <strong>Total Time:</strong> {batch_results.total_time:.2f}s
                    </div>
                </div>
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Input</th>
                    <th>Expected</th>
                    <th>Candidate</th>
                    <th>Score</th>
                    <th>Explanation</th>
                </tr>
        """

    for result in batch_results.results:
      score_class = f"score-{int(result.score)}" if result.error is None else ""
      html_content += f"""
                <tr class="{score_class}">
                    <td>{result.input_text[:100]}...</td>
                    <td>{result.expected_output[:100]}...</td>
                    <td>{result.candidate_output[:100]}...</td>
                    <td>{result.score:.1f}</td>
                    <td>{result.explanation[:200]}...</td>
                </tr>
            """

    html_content += """
            </table>
        </body>
        </html>
        """

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(html_content)
    print(f"Detailed HTML report saved to {report_path}")
