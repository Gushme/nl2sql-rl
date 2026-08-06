from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def agent_db(tmp_path: Path) -> Path:
    path = tmp_path / "agent.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE departments(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE employees(
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER REFERENCES departments(id),
            salary REAL
        );
        INSERT INTO departments VALUES (1, '研发'), (2, '销售');
        INSERT INTO employees VALUES
            (1, 'Alice', 1, 100.0),
            (2, 'Bob', 1, 120.0),
            (3, 'Carol', 2, 90.0);
        """
    )
    connection.close()
    return path
