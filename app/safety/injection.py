import re

def detect_prompt_injection(text: str) -> bool:
    dangerous_patterns = [
        r"ignore previous instructions",
        r"forget your previous",
        r"you are now a different",
        r"system prompt",
        r"jailbreak",
        r"act as (an|a) (hacker|attacker|malicious)"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in dangerous_patterns)