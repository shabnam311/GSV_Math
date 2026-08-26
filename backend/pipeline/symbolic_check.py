import re
import sympy

def verify_equations(reasoning_text: str):
    """
    Extracts basic  op b = c equations and verifies them with sympy.
    Returns: True if consistent, False if inconsistent, None if no equations found.
    """
    equation_pattern = re.compile(r'([0-9\.\-]+)\s*([\+\-\*\/])\s*([0-9\.\-]+)\s*=\s*([0-9\.\-]+)')
    matches = equation_pattern.findall(reasoning_text)
    
    if not matches:
        return None
        
    for match in matches:
        left_a, op, left_b, right_c = match
        expr_str = f"{left_a} {op} {left_b}"
        
        try:
            left_val = sympy.sympify(expr_str)
            right_val = sympy.sympify(right_c)
            
            # Simple float comparison
            if abs(float(left_val) - float(right_val)) > 1e-4:
                return False # Found an arithmetic contradiction
        except Exception:
            continue # Ignore unparseable equations
            
    return True # All extracted equations are sound
