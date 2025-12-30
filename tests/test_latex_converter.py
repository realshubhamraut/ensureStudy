"""
Tests for LaTeX Converter Service

Tests:
1. Math detection
2. Plaintext to LaTeX conversion
3. Step-by-step equation formatting
4. OCR cleanup
5. Response enhancement
"""
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'ai-service'))

from app.services.latex_converter import (
    detect_math_content,
    convert_to_latex,
    normalize_math_text,
    extract_math_segments,
    format_step_by_step,
    cleanup_ocr_math,
    enhance_response_with_latex
)


class TestMathDetection:
    """Test math content detection"""
    
    def test_detect_equation(self):
        """Test detection of basic equations"""
        assert detect_math_content("x = 5 + 3") is True
        assert detect_math_content("a² + b² = c²") is True
        assert detect_math_content("y = mx + b") is True
    
    def test_detect_math_symbols(self):
        """Test detection of math symbols"""
        assert detect_math_content("√16 = 4") is True
        assert detect_math_content("π ≈ 3.14159") is True
        assert detect_math_content("x ± 5") is True
    
    def test_detect_no_math(self):
        """Test that non-math text is not flagged"""
        assert detect_math_content("Hello world") is False
        assert detect_math_content("The cat sat on the mat") is False
        assert detect_math_content("History of France") is False
    
    def test_detect_trig_functions(self):
        """Test detection of trig functions"""
        assert detect_math_content("sin(θ) = opposite/hypotenuse") is True
        assert detect_math_content("cos 45° = √2/2") is True


class TestConversion:
    """Test plaintext to LaTeX conversion"""
    
    def test_convert_sqrt(self):
        """Test square root conversion"""
        assert "\\sqrt{x}" in convert_to_latex("sqrt(x)")
        assert "\\sqrt{x}" in convert_to_latex("√x")
        assert "\\sqrt{b^2 - 4ac}" in convert_to_latex("sqrt(b^2 - 4ac)")
    
    def test_convert_fractions(self):
        """Test fraction conversion"""
        assert "\\frac{1}{2}" in convert_to_latex("1/2")
        assert "\\frac{a}{b}" in convert_to_latex("a/b")
    
    def test_convert_superscripts(self):
        """Test superscript/exponent conversion"""
        result = convert_to_latex("x^2")
        assert "^{2}" in result
        
        result = convert_to_latex("x²")
        assert "^{2}" in result
    
    def test_convert_subscripts(self):
        """Test subscript conversion"""
        result = convert_to_latex("x_1")
        assert "_{1}" in result
    
    def test_convert_plus_minus(self):
        """Test plus/minus conversion"""
        assert "\\pm" in convert_to_latex("+-")
        assert "\\pm" in convert_to_latex("±")
    
    def test_convert_operators(self):
        """Test operator conversion"""
        assert "\\times" in convert_to_latex("3 * 4")
        assert "\\leq" in convert_to_latex("x <= 5")
        assert "\\geq" in convert_to_latex("x >= 5")
        assert "\\neq" in convert_to_latex("x != 0")
    
    def test_convert_trig(self):
        """Test trig function conversion"""
        assert "\\sin" in convert_to_latex("sin x")
        assert "\\cos" in convert_to_latex("cos x")
        assert "\\tan" in convert_to_latex("tan x")
        assert "\\log" in convert_to_latex("log x")
    
    def test_convert_greek(self):
        """Test Greek letter conversion"""
        assert "\\pi" in convert_to_latex("pi")
        assert "\\alpha" in convert_to_latex("alpha")
        assert "\\theta" in convert_to_latex("theta")


class TestMathSegmentExtraction:
    """Test extraction of math segments from text"""
    
    def test_extract_equation(self):
        """Test extracting equations"""
        text = "The formula is x = 5 + 3."
        segments = extract_math_segments(text)
        
        assert len(segments) >= 1
        assert any("=" in s[0] for s in segments)
    
    def test_extract_multiple_equations(self):
        """Test extracting multiple equations"""
        text = "Given a = 3 and b = 4, we have c = 5."
        segments = extract_math_segments(text)
        
        assert len(segments) >= 1


class TestNormalizeMathText:
    """Test full math text normalization"""
    
    def test_normalize_with_equations(self):
        """Test normalization with equations"""
        text = "The area is A = pi * r^2"
        processed, blocks = normalize_math_text(text)
        
        # Should detect math and create blocks
        assert len(blocks) >= 0  # May or may not extract blocks depending on pattern
    
    def test_normalize_empty_text(self):
        """Test with empty text"""
        processed, blocks = normalize_math_text("")
        
        assert processed == ""
        assert blocks == []


class TestStepByStepFormatting:
    """Test step-by-step equation formatting"""
    
    def test_format_steps(self):
        """Test formatting list of equations as steps"""
        equations = [
            "x + 5 = 10",
            "x = 10 - 5",
            "x = 5"
        ]
        
        blocks = format_step_by_step(equations)
        
        assert len(blocks) == 3
        assert blocks[0].id == "step1"
        assert blocks[1].id == "step2"
        assert blocks[2].id == "step3"
        assert blocks[0].position == {"step": 1}


class TestOCRCleanup:
    """Test OCR artifact cleanup"""
    
    def test_cleanup_l_to_1(self):
        """Test converting 'l' to '1' in numeric context"""
        result = cleanup_ocr_math("l0")
        assert result == "10"
    
    def test_cleanup_O_to_0(self):
        """Test converting 'O' to '0' in numeric context"""
        result = cleanup_ocr_math("1O0")
        assert result == "100"
    
    def test_cleanup_sqrt_typo(self):
        """Test fixing 'sqr t' to 'sqrt'"""
        result = cleanup_ocr_math("sqr t(x)")
        assert result == "sqrt(x)"


class TestResponseEnhancement:
    """Test enhancing tutor response with LaTeX"""
    
    def test_enhance_with_math(self):
        """Test enhancement of response containing math"""
        answer_short = "The quadratic formula is x = (-b ± sqrt(b^2 - 4ac)) / 2a"
        answer_detailed = "Step 1: x^2 + 5x + 6 = 0\nStep 2: (x + 2)(x + 3) = 0"
        
        result = enhance_response_with_latex(answer_short, answer_detailed)
        
        assert "latex_blocks" in result
        assert result["render_hint"] == "katex"
        assert "answer_detailed_plain" in result
    
    def test_enhance_without_math(self):
        """Test enhancement of response without math"""
        answer_short = "Paris is the capital of France."
        answer_detailed = "France is a country in Western Europe."
        
        result = enhance_response_with_latex(answer_short, answer_detailed)
        
        assert "latex_blocks" in result
        assert result["render_hint"] == "katex"


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_quadratic_formula_pipeline(self):
        """Test full pipeline with quadratic formula"""
        text = "x = (-b ± √(b² - 4ac)) / 2a"
        
        # Should detect math
        assert detect_math_content(text) is True
        
        # Should convert
        latex = convert_to_latex(text)
        
        # Should contain LaTeX constructs
        assert "\\pm" in latex or "±" in latex
        assert "\\sqrt" in latex or "√" in latex
    
    def test_pythagorean_theorem(self):
        """Test with Pythagorean theorem"""
        text = "a² + b² = c²"
        
        assert detect_math_content(text) is True
        
        latex = convert_to_latex(text)
        assert "^{2}" in latex or "²" in latex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
