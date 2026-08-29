import re

FINAL_ANSWER_PATTERNS = [
    r'\\boxed\{([^}]*)\}',
    r'[Ff]inal\s*[Aa]nswer\s*[:\-]?\s*(.{1,80})',
    r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer|value|result)\s+is\s*[:\-]?\s*(.{1,80})',
    r'[Tt]he\s+answer\s+is\s*[:\-]?\s*(.{1,80})',
    r'[Ss]o\s+the\s+answer\s+is\s*[:\-]?\s*(.{1,80})',
    r'=\s*(\S+)\s*$',
]

def extract_answer(raw_text: str, tail_chars: int = 300) -> str:
    """
    Extracts the final answer from a reasoning trace using regex fallbacks.
    """
    if not isinstance(raw_text, str): return str(raw_text)
    
    # Check regexes first
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, raw_text, re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
            
    # Fallback 1: 'answer is' string match
    idx = raw_text.lower().rfind("answer is")
    if idx != -1:
        ans = raw_text[idx + 9:].strip()
        ans = ans.replace(":", "").replace(".", "").strip()
        if ans: return ans
        
    # Fallback 2: very last word
    words = raw_text.split()
    if words: return words[-1]
    
    # Fallback 3: tail chars
    return raw_text[-tail_chars:] if len(raw_text) > tail_chars else raw_text

def normalize_answer(ans: str) -> str:
    """
    Normalizes extracted answers for CISC voting (stripping units, x=, etc.)
    so mathematically equivalent strings pool correctly.
    """
    if not ans: return ""
    ans = ans.strip().lower()
    
    # Strip common prefixes
    prefixes = ["x=", "y=", "z=", "v=", "a=", "b=", "c="]
    for p in prefixes:
        if ans.startswith(p):
            ans = ans[len(p):].strip()
            
    # Try cast to float
    try:
        f = float(ans)
        # return int string if it's perfectly integral (e.g. 5.0 -> '5')
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        pass
        
    return ans
