import re
import unicodedata

# ── Unicode math canonicalization map ──
_UNICODE_MAP = {
    '\u221a': 'sqrt',   # √
    '\u00d7': '*',      # ×
    '\u00f7': '/',      # ÷
    '\u03c0': 'pi',     # π
    '\u00b2': '^2',     # ²
    '\u00b3': '^3',     # ³
    '\u2070': '^0',     # ⁰
    '\u00b9': '^1',     # ¹
    '\u2074': '^4',     # ⁴
    '\u2075': '^5',     # ⁵
    '\u2076': '^6',     # ⁶
    '\u2077': '^7',     # ⁷
    '\u2078': '^8',     # ⁸
    '\u2079': '^9',     # ⁹
    '\u2013': '-',      # en-dash
    '\u2014': '-',      # em-dash
    '\u2212': '-',      # minus sign
    '\u00bd': '1/2',    # ½
    '\u00bc': '1/4',    # ¼
    '\u00be': '3/4',    # ¾
    '\u2153': '1/3',    # ⅓
    '\u2154': '2/3',    # ⅔
    '\u00b0': '',       # degree symbol (strip)
}

# ── Units to strip (longest-first to avoid partial matches) ──
_UNITS = sorted([
    'square units', 'sq units', 'cubic units', 'cu units',
    'units', 'unit',
    'centimeters', 'centimeter', 'cm²', 'cm2', 'cm',
    'meters', 'meter', 'mm', 'm²', 'm2', 'm',
    'inches', 'inch', 'in',
    'feet', 'foot', 'ft',
    'degrees', 'degree', 'deg', '°',
    'radians', 'radian', 'rad',
    'percent', '%',
    'dollars', 'dollar', '$',
], key=len, reverse=True)

# ── Regex patterns — ORDER MATTERS ──
# Most specific patterns first; capture group must NOT include filler words.
FINAL_ANSWER_PATTERNS = [
    # LaTeX \boxed{...}
    r'\\boxed\{([^}]*)\}',
    # "The answer is 15" / "So the answer is 15" — verb consumed OUTSIDE group
    r'(?:[Ss]o\s+)?[Tt]he\s+answer\s+is\s*[:\-]?\s*(.{1,80})',
    # "Therefore, the answer/value/result is 15"
    r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer|value|result)\s+is\s*[:\-]?\s*(.{1,80})',
    # "Final Answer: 15" / "Final answer - 15" — colon/dash separator required
    r'[Ff]inal\s*[Aa]nswer\s*[:\-=]\s*(.{1,80})',
    # "Final answer is 15" — "is" consumed outside group
    r'[Ff]inal\s*[Aa]nswer\s+is\s*[:\-]?\s*(.{1,80})',
    # Last equation: "= 15"
    r'=\s*(\S+)\s*$',
]


def _canonicalize_unicode(text: str) -> str:
    """Replace unicode math symbols with ASCII equivalents."""
    for uc, repl in _UNICODE_MAP.items():
        text = text.replace(uc, repl)
    # NFKD normalization for any remaining exotic chars
    text = unicodedata.normalize('NFKD', text)
    return text


def extract_answer(raw_text: str, tail_chars: int = 300) -> str:
    """
    Extracts the final answer from a reasoning trace using regex fallbacks.
    """
    if not isinstance(raw_text, str):
        return str(raw_text)

    # Check regexes first
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, raw_text, re.IGNORECASE | re.DOTALL))
        if matches:
            ans = matches[-1].group(1).strip()
            # Trim trailing sentence fragments (stop at period or newline)
            ans = re.split(r'[.\n]', ans)[0].strip()
            if ans:
                return ans

    # Fallback 1: 'answer is' string match
    idx = raw_text.lower().rfind("answer is")
    if idx != -1:
        ans = raw_text[idx + 9:].strip()  # len("answer is") == 9
        ans = re.split(r'[.\n]', ans)[0].strip()
        ans = ans.replace(":", "").strip()
        if ans:
            return ans

    # Fallback 2: very last word (strip trailing punctuation)
    words = raw_text.split()
    if words:
        return words[-1].rstrip('.,;:!?')

    # Fallback 3: tail chars
    return raw_text[-tail_chars:] if len(raw_text) > tail_chars else raw_text


def normalize_answer(ans: str) -> str:
    """
    Normalizes extracted answers for CISC voting (stripping units, variables, unicode)
    so mathematically equivalent strings pool correctly.
    """
    if not ans:
        return ""

    # Unicode canonicalization first
    ans = _canonicalize_unicode(ans)

    ans = ans.strip().lower()

    # Strip any single-letter variable assignment: "h = 10" → "10"
    ans = re.sub(r'^[a-z]\s*=\s*', '', ans)

    # Strip trailing units
    for unit in _UNITS:
        if ans.endswith(unit):
            ans = ans[:-len(unit)].strip()
            break  # only strip one unit

    # Strip trailing punctuation
    ans = ans.rstrip('.,;:!?')

    # Try cast to float for canonical form
    try:
        f = float(ans)
        # return int string if it's perfectly integral (e.g. 5.0 -> '5')
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        pass

    return ans
