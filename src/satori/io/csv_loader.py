from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from rich.console import Console

console = Console()


class CSVLoader:
  """Load and validate CSV datasets for LLM evaluation.

  By default expects columns named "input" and "expected_output" but can be
  configured to read any column names via constructor parameters. Internally,
  the loader normalizes to canonical column names "input" and
  "expected_output" so downstream code remains unchanged.
  """

  def __init__(
    self,
    file_path: str,
    input_col: str = "input",
    expected_col: str = "expected_output",
  ):
    """Initialize CSV loader with file path.

    Args:
        file_path: Path to the CSV file

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    self.file_path = Path(file_path)
    if not self.file_path.exists():
      raise FileNotFoundError(f"CSV file not found: {file_path}")

    self._input_col = input_col
    self._expected_col = expected_col
    self.df: Optional[pd.DataFrame] = None

  def load(self) -> pd.DataFrame:
    """Load CSV file and validate its structure.

    Returns:
        Loaded and validated DataFrame

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    try:
      self.df = pd.read_csv(self.file_path)
    except Exception as e:
      raise ValueError(f"Failed to read CSV file: {str(e)}")

    required = [self._input_col, self._expected_col]
    missing_columns = [c for c in required if c not in self.df.columns]
    if missing_columns:
      raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
        + ". Available columns: "
        + ", ".join(map(str, self.df.columns))
      )

    # Normalize to canonical names used throughout the pipeline
    rename_map = {
      self._input_col: "input",
      self._expected_col: "expected_output",
    }
    self.df = self.df.rename(columns=rename_map)

    self.df = self._handle_missing_data(self.df)

    for col in ("input", "expected_output"):
      self.df[col] = self.df[col].astype(str)

    return self.df

  def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data in the DataFrame.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with missing data handled
    """
    for col in ("input", "expected_output"):
      missing_count = df[col].isna().sum()
      if missing_count > 0:
        console.print(
          f"Warning: Found {missing_count} missing values in column '{col}'"
        )
        df[col] = df[col].fillna("")

    empty_mask = (df["input"] == "") & (df["expected_output"] == "")
    if empty_mask.any():
      num_removed = empty_mask.sum()
      console.print(f"Removing {num_removed} completely empty rows")
      df = df[~empty_mask]

    df = df.reset_index(drop=True)

    return df

  def get_rows(self) -> List[Dict[str, Any]]:
    """Get all rows as a list of dictionaries.

    Returns:
        List of dictionaries, each representing a row

    Raises:
        ValueError: If data hasn't been loaded yet
    """
    if self.df is None:
      raise ValueError("Data not loaded. Call load() first.")

    return cast(List[Dict[str, Any]], self.df.to_dict("records"))

  def get_row(self, index: int) -> Dict[str, Any]:
    """Get a specific row by index.

    Args:
        index: Row index (0-based)

    Returns:
        Dictionary representing the row

    Raises:
        ValueError: If data hasn't been loaded or index is out of bounds
    """
    if self.df is None:
      raise ValueError("Data not loaded. Call load() first.")

    if index < 0 or index >= len(self.df):
      raise ValueError(f"Index {index} out of bounds (0-{len(self.df) - 1})")

    return self.df.iloc[index].to_dict()

  def __len__(self) -> int:
    """Get the number of rows in the dataset.

    Returns:
        Number of rows, or 0 if not loaded
    """
    if self.df is None:
      return 0
    return len(self.df)

  def __repr__(self) -> str:
    """String representation of the loader."""
    status = "loaded" if self.df is not None else "not loaded"
    rows = len(self) if self.df is not None else "unknown"
    return (
      f"CSVLoader(file='{self.file_path.name}', status={status}, rows={rows})"
    )
