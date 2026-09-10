import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def verify_equations(reasoning_text: str):
    """
    Extracts algebraic equations and verifies them with sympy.
    Returns: True if consistent, False if inconsistent, None if no verifiable equations found.
    """
    if not reasoning_text:
        return None
        
    # Match equations with variables, multiple operators, parentheses
    # E.g., "x + 5 = 12", "(5 * 6) / 2 = 15"
    equation_pattern = re.compile(r'([a-zA-Z0-9\.\-\+\*\/\(\)\s]+)=([a-zA-Z0-9\.\-\+\*\/\(\)\s]+)')
    matches = equation_pattern.findall(reasoning_text)
    
    if not matches:
        return None
        
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    verified_any = False
    
    for match in matches:
        left_str, right_str = match
        # Clean up whitespace
        left_str = left_str.strip()
        right_str = right_str.strip()
        
        # Skip trivial matches or empty strings
        if not left_str or not right_str or len(left_str) < 1 or len(right_str) < 1:
            continue
            
        try:
            left_val = parse_expr(left_str, transformations=transformations)
            right_val = parse_expr(right_str, transformations=transformations)
            
            # Symbolic comparison: simplify(lhs - rhs) == 0
            diff = sympy.simplify(left_val - right_val)
            
            if diff != 0:
                return False # Found a contradiction
                
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
