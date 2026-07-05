def precision_recall(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    """Set-based retrieval precision/recall over paper ids."""
    retrieved_set, expected_set = set(retrieved), set(expected)
    overlap = len(retrieved_set & expected_set)
    precision = overlap / len(retrieved_set) if retrieved_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    return precision, recall
