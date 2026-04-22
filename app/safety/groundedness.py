def groundedness_check(user_query: str, retrieved_chunks: list, final_answer: str) -> bool:
    """
    A simple overlap-based groundedness check.
    Returns True if the final answer contains meaningful terms from the retrieved chunks.
    For a prototype, this is sufficient. In production, you would use an LLM evaluator.
    """
    if not retrieved_chunks:
        return False  # No grounding possible
    combined_chunks = " ".join(retrieved_chunks).lower()
    answer_words = set(final_answer.lower().split())
    # Count how many answer words appear in the chunks (ignore very short words)
    chunk_words = set(combined_chunks.split())
    overlap = answer_words.intersection(chunk_words)
    # Heuristic: at least 10% of answer words or at least 5 words overlap
    return len(overlap) > 5 or (len(overlap) / max(1, len(answer_words))) > 0.1