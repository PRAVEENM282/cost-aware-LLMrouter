"""
Cost Metering Module for LLM Gateway Semantic Router.

Pricing Table:
- Cache: $0.0000 flat per query
- Calculator: $0.0001 flat per query
- SQL: $0.0001 flat per query
- Small LLM: $0.001 per 1,000 input tokens; $0.002 per 1,000 output tokens
- Large LLM: $0.010 per 1,000 input tokens; $0.030 per 1,000 output tokens
"""

import math

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken (cl100k_base), falling back to standard approximation.
    """
    if not text:
        return 0
    if _encoding is not None:
        return len(_encoding.encode(text))
    # Approximation as specified in guidelines
    return max(1, math.ceil(len(text.split()) * 1.3))


def calculate_cost(path: str, input_text: str, output_text: str) -> float:
    """
    Calculates the exact cost of an execution path based on the mandatory pricing table.

    Params:
      - path (str): The executed path ('calculator', 'sql', 'cache', 'small_llm', 'large_llm')
      - input_text (str): The original query
      - output_text (str): The generated response

    Returns:
      - (float) The exact cost based on the pricing table
    """
    p = path.lower()
    if p == "cache":
        return 0.0000
    elif p == "calculator":
        return 0.0001
    elif p == "sql":
        return 0.0001
    elif p == "small_llm":
        in_tokens = count_tokens(input_text)
        out_tokens = count_tokens(output_text)
        cost = (in_tokens / 1000.0) * 0.001 + (out_tokens / 1000.0) * 0.002
        return float(cost)
    elif p == "large_llm":
        in_tokens = count_tokens(input_text)
        out_tokens = count_tokens(output_text)
        cost = (in_tokens / 1000.0) * 0.010 + (out_tokens / 1000.0) * 0.030
        return float(cost)
    else:
        # Fallback to large_llm pricing
        in_tokens = count_tokens(input_text)
        out_tokens = count_tokens(output_text)
        cost = (in_tokens / 1000.0) * 0.010 + (out_tokens / 1000.0) * 0.030
        return float(cost)
