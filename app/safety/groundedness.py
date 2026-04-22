def groundedness_check(query: str, retrieved_chunks: list, answer: str) -> bool:
    if not retrieved_chunks:
        return False
    combined = " ".join(retrieved_chunks)
    answer_words = set(answer.lower().split())
    chunk_words = set(combined.lower().split())
    overlap = len(answer_words & chunk_words)
    return overlap > 5