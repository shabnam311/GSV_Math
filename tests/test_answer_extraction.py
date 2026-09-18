"""
Tests for the fixed answer_extraction module.
Verifies all the bugs identified in the 40-point audit are fixed.

Run: python -m pytest tests/test_answer_extraction.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline.answer_extraction import extract_answer, normalize_answer


class TestExtractAnswer:
    """Audit D.15: 'The final answer is 15' must extract '15', not 'is 15'."""

    def test_the_final_answer_is(self):
        text = "After calculating, the final answer is 15"
        assert extract_answer(text) == "15"

    def test_the_answer_is(self):
        text = "So the answer is 42."
        result = extract_answer(text)
        assert result == "42", f"Got: {result}"

    def test_final_answer_colon(self):
        text = "Final Answer: 15"
        assert extract_answer(text) == "15"

    def test_final_answer_dash(self):
        text = "Final Answer - 25"
        assert extract_answer(text) == "25"

    def test_boxed_latex(self):
        text = r"Therefore \boxed{42} is correct."
        assert extract_answer(text) == "42"

    def test_therefore_the_answer_is(self):
        text = "Therefore, the answer is 7"
        assert extract_answer(text) == "7"

    def test_equals_sign_last_line(self):
        text = "x + 5 = 20\n= 15"
        assert extract_answer(text) == "15"

    def test_trailing_period_stripped(self):
        """Extract should stop at period (not include it in the answer)."""
        text = "The answer is 99. This is confirmed."
        result = extract_answer(text)
        assert result == "99", f"Got: {result}"

    def test_fallback_last_word(self):
        text = "some reasoning blah blah 42"
        assert extract_answer(text) == "42"

    def test_fallback_strips_punctuation(self):
        text = "some reasoning blah blah 42."
        result = extract_answer(text)
        assert result == "42", f"Got: {result}"


class TestNormalizeAnswer:
    """Audit D.17-D.20: unicode, units, variable prefixes."""

    def test_unicode_sqrt(self):
        result = normalize_answer("\u221a16")
        assert result == "sqrt16", f"Got: {result}"

    def test_unicode_superscript(self):
        result = normalize_answer("3\u00b2")
        assert result == "3^2", f"Got: {result}"

    def test_unicode_pi(self):
        result = normalize_answer("2\u03c0")
        assert result == "2pi", f"Got: {result}"

    def test_strip_units_cm(self):
        result = normalize_answer("15 cm")
        assert result == "15", f"Got: {result}"

    def test_strip_units_square_units(self):
        result = normalize_answer("25 square units")
        assert result == "25", f"Got: {result}"

    def test_strip_units_degrees(self):
        result = normalize_answer("90 degrees")
        assert result == "90", f"Got: {result}"

    def test_variable_prefix_h(self):
        """Audit D.19: 'h = 10' should normalize to '10', not left as 'h = 10'."""
        result = normalize_answer("h = 10")
        assert result == "10", f"Got: {result}"

    def test_variable_prefix_x(self):
        result = normalize_answer("x=5")
        assert result == "5", f"Got: {result}"

    def test_variable_prefix_r(self):
        result = normalize_answer("r = 3.5")
        assert result == "3.5", f"Got: {result}"

    def test_float_to_int(self):
        result = normalize_answer("5.0")
        assert result == "5", f"Got: {result}"

    def test_float_preserved(self):
        result = normalize_answer("3.14")
        assert result == "3.14", f"Got: {result}"

    def test_equivalent_unicode_forms(self):
        """sqrt and unicode sqrt should normalize to the same string."""
        a = normalize_answer("\u221a(a\u00b2+b\u00b2)")
        b = normalize_answer("sqrt(a^2+b^2)")
        assert a == b, f"Got: '{a}' vs '{b}'"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
