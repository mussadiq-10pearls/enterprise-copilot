import re

def detect_prompt_injection(text: str) -> bool:
    """Return True if the user message appears to be a prompt injection attempt."""
    patterns = [
        r"\bignore\s+(all\s+)?previous\s+instructions\b",
        r"\bforget\s+(your\s+)?(previous|system)\s+(instructions|prompt)\b",
        r"\byou\s+are\s+now\s+a\s+different\s+(assistant|system|AI)\b",
        r"\b(new\s+)?system\s+prompt\s*:\s*",
        r"\b(break|bypass|crack)\s+(out\s+of|the\s+)?(jailbreak|sandbox|restrictions?)\b",
        r"\bact\s+(as|like)\s+(an|a)\s+(hacker|attacker|malicious|evil)\b",
        r"\b(don't|do\s+not)\s+follow\s+(the\s+)?(rules|guidelines|policies)\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)