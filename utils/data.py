"""Utilities to read Excel by ID and maintain a CSV of original rows and predictions.

Requirements implemented:
1) Read the Excel, find a line by ID, and return the line.
2) Write a new CSV that contains the original line and a line of predictions based on the predicted column.
3) Each time the ID already exists in the CSV and the value of that column is filled, replace the value; if empty, add the new value.
4) The CSV follows the structure (column order/names) of the original Excel.

Notes:
- Uses pandas for I/O.
- ID column can be an index (int) or a column name (str). Default assumes the first column.
- For step (2), we create a CSV with the header copied from the Excel. The prediction is written into the predicted column for an extra row labeled with a special marker unless in-place update is desired.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import numpy as np

default_csv_path_output = os.path.join('data', 'output.csv')
default_csv_path= os.path.join('data', 'Resultados_Patricia-Rodada-03.xlsx')

def _resolve_id_col(df: pd.DataFrame, id_col: int | str = 0) -> str:
    """Return the column name for id_col which may be index or name."""
    return df.columns[id_col] if isinstance(id_col, int) else id_col


def _read_table(path: str) -> pd.DataFrame:
    """Read a table from either CSV or Excel based on file extension."""
    lower = path.lower()
    if lower.endswith((".csv", ".txt")):
        return pd.read_csv(path)
    # default to Excel for .xlsx/.xls and other spreadsheet formats
    return pd.read_excel(path)


essential_marker_col = "__row_type__"  # distinguishes 'original' vs 'prediction' rows


def read_row_by_id(source_path: str, id_value: Any, id_col: int | str = 0) -> Optional[Dict[str, Any]]:
    """Read an Excel or CSV file and return the row as a dict where id column equals id_value.

    Returns None if not found.
    """
    df = _read_table(source_path)
    key = _resolve_id_col(df, id_col)
    row = df[df[key] == id_value]
    if row.empty:
        return None
    return row.iloc[0].to_dict() # basically converts series to a object

