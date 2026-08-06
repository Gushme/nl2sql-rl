from nl2sql_rl.data.split import choose_validation_databases, normalize_question


def test_validation_split_is_deterministic_and_database_disjoint() -> None:
    counts = {f"db_{index}": 100 + index for index in range(12)}
    first = choose_validation_databases(counts, seed=42)
    second = choose_validation_databases(dict(reversed(list(counts.items()))), seed=42)
    assert first == second
    assert first
    assert set(first).issubset(counts)
    assert set(first) != set(counts)
    assert sum(counts[db_id] for db_id in first) >= 100


def test_question_normalization_removes_case_punctuation_and_spacing() -> None:
    assert normalize_question("  How MANY rows? ") == normalize_question("how many rows")
