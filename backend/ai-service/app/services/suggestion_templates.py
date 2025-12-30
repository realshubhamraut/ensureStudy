"""
Suggestion Templates Library

Curated templates for generating follow-up question candidates.
Templates are grouped by intent categories for diverse suggestions.

Extensible: Add new templates without code changes by editing this file.
"""
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class QuestionTemplate:
    """A question template with metadata."""
    template: str  # Template with {topic} placeholder
    intent: str    # Category: example, application, practice, etc.
    priority: int = 1  # Higher = more likely to be selected


# ============================================================================
# Template Library by Intent
# ============================================================================

TEMPLATES: Dict[str, List[str]] = {
    "example": [
        "Can you give an example of {topic}?",
        "Show me a worked example involving {topic}.",
        "What's a simple example that illustrates {topic}?",
        "Can you demonstrate {topic} with a concrete case?",
    ],
    
    "application": [
        "How is {topic} used in real life?",
        "What are practical applications of {topic}?",
        "Where would I encounter {topic} in everyday situations?",
        "How do professionals use {topic}?",
    ],
    
    "proof": [
        "Why does {topic} work that way?",
        "Can you explain the reasoning behind {topic}?",
        "What's the logic behind {topic}?",
        "How can we prove or verify {topic}?",
    ],
    
    "practice": [
        "How can I practice problems on {topic}?",
        "What exercises would help me master {topic}?",
        "Can you suggest practice questions for {topic}?",
        "What problems should I solve to understand {topic} better?",
    ],
    
    "pitfalls": [
        "What are common mistakes with {topic}?",
        "What should I avoid when working with {topic}?",
        "What errors do students often make with {topic}?",
        "What are the tricky parts of {topic}?",
    ],
    
    "prerequisites": [
        "What prerequisite knowledge do I need for {topic}?",
        "What should I learn before studying {topic}?",
        "What background is required to understand {topic}?",
        "What concepts lead up to {topic}?",
    ],
    
    "stepbystep": [
        "Can you show a step-by-step solution for {topic}?",
        "Walk me through solving a {topic} problem.",
        "Break down the process of {topic} into steps.",
        "What's the procedure for working with {topic}?",
    ],
    
    "intuition": [
        "What's the intuition behind {topic}?",
        "How should I think about {topic}?",
        "What's the big picture of {topic}?",
        "Can you give me a mental model for {topic}?",
    ],
    
    "comparison": [
        "How does {topic} compare to similar concepts?",
        "What's the difference between {topic} and related ideas?",
        "When should I use {topic} instead of alternatives?",
    ],
    
    "history": [
        "What's the history or origin of {topic}?",
        "Who discovered or invented {topic}?",
        "How did {topic} develop over time?",
    ],
    
    "visualization": [
        "Can you visualize or diagram {topic}?",
        "What does {topic} look like graphically?",
        "How can I picture {topic} in my mind?",
    ],
    
    "edgecases": [
        "What are the edge cases or limits of {topic}?",
        "When does {topic} not apply?",
        "What are the boundary conditions for {topic}?",
    ],
    
    "causeseffects": [
        "What were the main causes of {topic}?",
        "What were the consequences of {topic}?",
        "What led to {topic}?",
        "What was the impact of {topic}?",
    ],
    
    "timeline": [
        "What are the key events in {topic}?",
        "Can you give a timeline of {topic}?",
        "When did {topic} happen and what came after?",
    ],
}


# ============================================================================
# Generic Fallback Templates
# ============================================================================

GENERIC_TEMPLATES = [
    "Can you explain this more simply?",
    "What's the most important thing to remember?",
    "How would this appear on an exam?",
    "What are the key takeaways?",
    "Can you summarize the main points?",
    "What should I focus on to understand this?",
]


# ============================================================================
# Template Selection Functions
# ============================================================================

def get_templates_for_intent(intent: str) -> List[str]:
    """Get all templates for a specific intent."""
    return TEMPLATES.get(intent, [])


def get_all_intents() -> List[str]:
    """Get list of all available intent categories."""
    return list(TEMPLATES.keys())


def get_diverse_templates(max_per_intent: int = 1) -> List[QuestionTemplate]:
    """
    Get a diverse set of templates (max_per_intent from each category).
    
    Returns templates as QuestionTemplate objects with metadata.
    """
    result = []
    for intent, templates in TEMPLATES.items():
        for i, template in enumerate(templates[:max_per_intent]):
            result.append(QuestionTemplate(
                template=template,
                intent=intent,
                priority=max_per_intent - i  # First template in category has highest priority
            ))
    return result


def instantiate_template(template: str, topic: str) -> str:
    """Fill in the {topic} placeholder in a template."""
    return template.format(topic=topic)


def get_generic_fallbacks() -> List[str]:
    """Get generic fallback suggestions when extraction yields too few candidates."""
    return GENERIC_TEMPLATES.copy()


# ============================================================================
# Template Statistics
# ============================================================================

def get_template_stats() -> dict:
    """Get statistics about the template library."""
    return {
        "total_intents": len(TEMPLATES),
        "total_templates": sum(len(t) for t in TEMPLATES.values()),
        "templates_per_intent": {k: len(v) for k, v in TEMPLATES.items()},
        "generic_fallbacks": len(GENERIC_TEMPLATES),
    }
