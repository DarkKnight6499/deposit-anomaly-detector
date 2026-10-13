"""Tiny helpers to write DataFrames as Markdown/CSV without extra dependencies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(v, digits: int) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{digits}f}"
    if isinstance(v, pd.Timestamp):
        return "" if pd.isna(v) else v.strftime("%Y-%m-%d")
    return "" if v is None or v is pd.NaT else str(v)


def to_markdown(df: pd.DataFrame, digits: int = 3, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def write_outputs(df: pd.DataFrame, directory: Path, stem: str, title: str, digits: int = 3,
                  index: bool = False, note: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (df.reset_index() if index else df).to_csv(directory / f"{stem}.csv", index=False)
    body = f"# {title}\n\n" + (note + "\n\n" if note else "") + to_markdown(df, digits, index) + "\n"
    (directory / f"{stem}.md").write_text(body, encoding="utf-8")
