"""
LaTeX Converter Service - Convert plaintext math to LaTeX

Implements:
- Math detection heuristics
- Plaintext to LaTeX conversion rules
- Step-by-step equation formatting
- LaTeX block extraction with positions
"""
import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LatexBlock:
    """A LaTeX equation block"""
    id: str
    latex: str
    position: dict  # {"within_text_index": N}
    plaintext_fallback: str


# ============================================================================
# Math Detection
# ============================================================================

# Characters and patterns that indicate math content
MATH_INDICATORS = [
    '=', '+', '-', '×', '÷', '±', '≠', '≈', '≤', '≥',
    '^', '√', '∑', '∏', '∫', 'π', 'θ', 'α', 'β', 'γ',
    'sin', 'cos', 'tan', 'log', 'ln', 'exp',
    'lim', 'dx', 'dy', 'dt',
    '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁰',
    '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'
]

MATH_PATTERNS = [
    r'\d+\s*[\+\-\*/\^]\s*\d+',  # Simple arithmetic
    r'[a-z]\s*=\s*',              # Variable assignment
    r'[a-z]\^2',                   # Squared variables
    r'sqrt\s*\(',                  # sqrt function
    r'\d+/\d+',                    # Fractions
    r'[a-z]_\d',                   # Subscripts
]


def detect_math_content(text: str) -> bool:
    """
    Detect if text contains mathematical content.
    
    Uses:
    - Symbol frequency check
    - Pattern matching
    
    Args:
        text: Text to analyze
        
    Returns:
        True if math content detected
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Check for math indicators
    indicator_count = sum(1 for ind in MATH_INDICATORS if ind in text)
    if indicator_count >= 2:
        return True
    
    # Check patterns
    for pattern in MATH_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False


def get_math_density(text: str) -> float:
    """
    Calculate math symbol density in text.
    
    Returns:
        Float between 0 and 1 indicating math density
    """
    if not text:
        return 0.0
    
    math_chars = sum(1 for char in text if char in "=+-×÷±²³√∑∏∫^/")
    return min(math_chars / len(text), 1.0)


# ============================================================================
# Conversion Rules
# ============================================================================

CONVERSION_RULES = [
    # Greek letters
    (r'\balpha\b', r'\\alpha'),
    (r'\bbeta\b', r'\\beta'),
    (r'\bgamma\b', r'\\gamma'),
    (r'\bdelta\b', r'\\delta'),
    (r'\btheta\b', r'\\theta'),
    (r'\bpi\b', r'\\pi'),
    (r'\bsigma\b', r'\\sigma'),
    (r'\bomega\b', r'\\omega'),
    (r'\binfinity\b', r'\\infty'),
    (r'\b∞\b', r'\\infty'),
    
    # Square root
    (r'sqrt\(([^)]+)\)', r'\\sqrt{\1}'),
    (r'√\(([^)]+)\)', r'\\sqrt{\1}'),
    (r'√(\w+)', r'\\sqrt{\1}'),
    
    # Fractions - complex ones
    (r'\(([^)]+)\)/\(([^)]+)\)', r'\\frac{\1}{\2}'),
    (r'(\d+)/(\d+)', r'\\frac{\1}{\2}'),
    (r'([a-z])/([a-z])', r'\\frac{\1}{\2}'),
    
    # Plus/minus
    (r'\+-', r'\\pm'),
    (r'±', r'\\pm'),
    
    # Comparison operators
    (r'<=', r'\\leq'),
    (r'>=', r'\\geq'),
    (r'!=', r'\\neq'),
    (r'≤', r'\\leq'),
    (r'≥', r'\\geq'),
    (r'≠', r'\\neq'),
    (r'≈', r'\\approx'),
    
    # Superscripts (exponents)
    (r'(\w)\^(\d+)', r'\1^{\2}'),
    (r'(\w)\^(\([^)]+\))', r'\1^{\2}'),
    (r'(\w)²', r'\1^{2}'),
    (r'(\w)³', r'\1^{3}'),
    
    # Subscripts
    (r'(\w)_(\d+)', r'\1_{\2}'),
    (r'([a-z])₀', r'\1_{0}'),
    (r'([a-z])₁', r'\1_{1}'),
    (r'([a-z])₂', r'\1_{2}'),
    
    # Multiplication
    (r'\*', r'\\times'),
    (r'×', r'\\times'),
    (r'·', r'\\cdot'),
    
    # Division
    (r'÷', r'\\div'),
    
    # Trig functions
    (r'\bsin\b', r'\\sin'),
    (r'\bcos\b', r'\\cos'),
    (r'\btan\b', r'\\tan'),
    (r'\blog\b', r'\\log'),
    (r'\bln\b', r'\\ln'),
    
    # Limits and integrals
    (r'\blim\b', r'\\lim'),
    (r'\bsum\b', r'\\sum'),
    (r'∑', r'\\sum'),
    (r'∏', r'\\prod'),
    (r'∫', r'\\int'),
]


# ============================================================================
# Main Conversion Functions
# ============================================================================

def normalize_math_text(text: str) -> Tuple[str, List[LatexBlock]]:
    """
    Convert plaintext math to LaTeX.
    
    Applies conversion rules and extracts LaTeX blocks.
    
    Args:
        text: Text potentially containing math
        
    Returns:
        (processed_text, list of LatexBlock)
    """
    if not text:
        return text, []
    
    latex_blocks = []
    block_count = 0
    
    # Find mathematical segments
    segments = extract_math_segments(text)
    
    processed_text = text
    
    for segment, start_idx in segments:
        # Convert segment to LaTeX
        latex = convert_to_latex(segment)
        
        if latex != segment:  # Only if conversion happened
            block_count += 1
            block_id = f"lb{block_count}"
            
            latex_blocks.append(LatexBlock(
                id=block_id,
                latex=latex,
                position={"within_text_index": start_idx},
                plaintext_fallback=segment
            ))
    
    logger.info(f"[MATH] detected_formula_count={len(segments)} latex_blocks={len(latex_blocks)}")
    
    return processed_text, latex_blocks


def convert_to_latex(text: str) -> str:
    """
    Apply conversion rules to convert plaintext math to LaTeX.
    
    Args:
        text: Math expression in plaintext
        
    Returns:
        LaTeX formatted string
    """
    result = text
    
    # Apply each conversion rule
    for pattern, replacement in CONVERSION_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def extract_math_segments(text: str) -> List[Tuple[str, int]]:
    """
    Extract segments that appear to be mathematical.
    
    Args:
        text: Full text to analyze
        
    Returns:
        List of (segment, start_index) tuples
    """
    segments = []
    
    # Pattern 1: Equations with = sign
    equation_pattern = r'[a-zA-Z0-9\s\+\-\*\/\^\(\)√±]+\s*=\s*[a-zA-Z0-9\s\+\-\*\/\^\(\)√±]+'
    for match in re.finditer(equation_pattern, text):
        segment = match.group().strip()
        if len(segment) > 3 and detect_math_content(segment):
            segments.append((segment, match.start()))
    
    # Pattern 2: Standalone expressions with math operators
    expr_pattern = r'(?:^|\s)([a-zA-Z]\s*[\+\-\*\/\^]\s*[a-zA-Z0-9]+(?:\s*[\+\-\*\/\^]\s*[a-zA-Z0-9]+)*)'
    for match in re.finditer(expr_pattern, text):
        segment = match.group(1).strip()
        if segment and detect_math_content(segment):
            # Avoid duplicates
            if not any(s[0] == segment for s in segments):
                segments.append((segment, match.start()))
    
    return segments


def convert_quadratic_formula(text: str) -> str:
    """
    Special handling for quadratic formula patterns.
    
    Detects patterns like:
    - x = (-b ± √(b² - 4ac)) / 2a
    - x = (-b ± sqrt(b^2 - 4ac)) / (2a)
    """
    quadratic_patterns = [
        # Pattern: x = (-b ± √(b² - 4ac)) / 2a
        (r'x\s*=\s*\(-b\s*±\s*√?\(?b[²2]\s*-\s*4ac\)?\)\s*/\s*2a',
         r'x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}'),
        
        # Pattern with sqrt
        (r'x\s*=\s*\(-b\s*\+-\s*sqrt\(b\^?2\s*-\s*4ac\)\)\s*/\s*\(?2a\)?',
         r'x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}'),
    ]
    
    result = text
    for pattern, replacement in quadratic_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def format_step_by_step(equations: List[str]) -> List[LatexBlock]:
    """
    Format a list of equations as numbered LaTeX blocks.
    
    Args:
        equations: List of equation strings
        
    Returns:
        List of numbered LatexBlock objects
    """
    blocks = []
    
    for i, eq in enumerate(equations):
        latex = convert_to_latex(eq.strip())
        blocks.append(LatexBlock(
            id=f"step{i+1}",
            latex=latex,
            position={"step": i + 1},
            plaintext_fallback=eq.strip()
        ))
    
    return blocks


def extract_step_equations(text: str) -> List[str]:
    """
    Extract step-by-step equations from text.
    
    Looks for patterns like:
    - Step 1: x = 5
    - 1. x = 5
    - (1) x = 5
    """
    patterns = [
        r'(?:Step\s*\d+[:\.]?\s*)([^\n]+)',
        r'(?:\d+[\.)\]]\s*)([^\n]*=\s*[^\n]+)',
    ]
    
    equations = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            eq = match.group(1).strip()
            if detect_math_content(eq):
                equations.append(eq)
    
    return equations


# ============================================================================
# Response Enhancement
# ============================================================================

def enhance_response_with_latex(
    answer_short: str,
    answer_detailed: Optional[str]
) -> dict:
    """
    Enhance tutor response with LaTeX blocks.
    
    Args:
        answer_short: Short answer text
        answer_detailed: Detailed answer text
        
    Returns:
        Dict with latex_blocks, render_hint, and plaintext versions
    """
    all_blocks = []
    
    # Process short answer
    _, short_blocks = normalize_math_text(answer_short)
    all_blocks.extend(short_blocks)
    
    # Process detailed answer
    if answer_detailed:
        _, detailed_blocks = normalize_math_text(answer_detailed)
        # Renumber blocks
        for i, block in enumerate(detailed_blocks):
            block.id = f"lb{len(short_blocks) + i + 1}"
        all_blocks.extend(detailed_blocks)
        
        # Check for step-by-step
        step_equations = extract_step_equations(answer_detailed)
        if step_equations:
            step_blocks = format_step_by_step(step_equations)
            all_blocks.extend(step_blocks)
    
    return {
        "latex_blocks": [
            {
                "id": b.id,
                "latex": b.latex,
                "position": b.position,
                "plaintext_fallback": b.plaintext_fallback
            }
            for b in all_blocks
        ],
        "render_hint": "katex",
        "answer_detailed_plain": answer_detailed  # Fallback
    }


# ============================================================================
# OCR Math Cleanup
# ============================================================================

def cleanup_ocr_math(text: str) -> str:
    """
    Clean up common OCR artifacts in math expressions.
    
    Args:
        text: OCR output with potential errors
        
    Returns:
        Cleaned text
    """
    # Common OCR errors
    corrections = [
        (r'l(\d)', r'1\1'),  # l -> 1
        (r'O(\d)', r'0\1'),  # O -> 0
        (r'(\d)O', r'\g<1>0'),  # O -> 0
        (r'\bl\b', '1'),     # lone l -> 1
        (r'\bO\b', '0'),     # lone O -> 0
        (r'x\s+2', 'x^2'),   # x 2 -> x^2
        (r'(\w)\s*\^\s*(\d)', r'\1^\2'),  # normalize superscripts
        (r'sqr\s*t', 'sqrt'),  # sqr t -> sqrt
    ]
    
    result = text
    for pattern, replacement in corrections:
        result = re.sub(pattern, replacement, result)
    
    return result
