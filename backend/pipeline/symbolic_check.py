import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def verify_equations(reasoning_text: str):
    """
    Extracts algebraic equations and verifies them with sympy.
    
    Strategy:
    - Pure numeric equations (e.g., "2 * 3 + 1 = 7"): verify via simplify(lhs - rhs) == 0
    - Equations with free symbols (e.g., "x + 5 = 12"): skip (return None for that equation)
      because simplify(lhs - rhs) checks identity-for-all-values, not solvability.
    - Division by zero: explicitly flag as False (arithmetic error in reasoning)
    
    Returns: True if consistent, False if inconsistent, None if no verifiable equations found.
    """
    if not reasoning_text:
        return None
        
    # Match equations with variables, multiple operators, parentheses
    equation_pattern = re.compile(r'([a-zA-Z0-9\.\-\+\*\/\(\)\s\^]+)=([a-zA-Z0-9\.\-\+\*\/\(\)\s\^]+)')
    matches = equation_pattern.findall(reasoning_text)
    
    if not matches:
        return None
        
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    verified_any = False
    
    for match in matches:
        left_str, right_str = match
        left_str = left_str.strip()
        right_str = right_str.strip()
        
        # Skip trivial/empty matches
        if not left_str or not right_str:
            continue
            
        try:
            left_val = parse_expr(left_str, transformations=transformations)
            right_val = parse_expr(right_str, transformations=transformations)
            
            diff = sympy.simplify(left_val - right_val)
            
            # Check for free symbols — if present, this is a variable equation
            # (e.g., "x + 5 = 12"). We cannot verify these with simplify(lhs-rhs)==0
            # because that checks algebraic identity for ALL values, not solvability.
            # Skip these rather than mislabeling them as contradictions.
            if diff.free_symbols:
                print(f"Skipping variable equation '{left_str} = {right_str}' (has free symbols: {diff.free_symbols})")
                continue
            
            if diff != 0:
                return False # Found an arithmetic contradiction
                
            verified_any = True
        except ZeroDivisionError:
            print("ZeroDivisionError in symbolic check.")
            return False # Flag as hallucinated math error
        except Exception as e:
            print(f"Failed to parse equation '{left_str} = {right_str}': {e}")
            continue # Skip unparseable
            
    if verified_any:
        return True
    return None
