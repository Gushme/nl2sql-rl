from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def evaluation_db(tmp_path: Path) -> Path:
    path = tmp_path / "evaluation.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE items(id INTEGER PRIMARY KEY, category TEXT, score REAL);
        INSERT INTO items VALUES
            (1, 'a', 1.0),
            (2, 'b', 2.0),
            (3, 'a', 3.0);
        """
    )
    connection.close()
    return path
