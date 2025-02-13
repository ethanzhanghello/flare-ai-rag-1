from typing import Final

BASE_PROMPT: Final = """
You are a query router. Analyze the query below and classify it by returning exactly one of these values—without any additional text:\n\n

    - **{answer_option}**: Use this if the query is clear, specific, and can be answered with factual information.\n
    - **{clarify_option}**: Use this if the query is ambiguous, vague, or needs additional context.\n
    - **{reject_option}**: Use this if the query is inappropriate, harmful, or completely out of scope.\n\n

    Query: {query}\n\n

    Return exactly one word: ANSWER, CLARIFY, or REJECT.
"""
