"""
Flowchart Generator Service with Gemini AI

Generates Mermaid flowchart code dynamically using Google Gemini API.
Falls back to templates if API is unavailable.
"""
import os
import re
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# ============================================================================
# Gemini API Integration
# ============================================================================

_gemini_client = None

def _get_gemini_client():
    """Lazy-load Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("[FLOWCHART] GEMINI_API_KEY not set, using templates only")
            return None
        
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=api_key)
            logger.info("[FLOWCHART] Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"[FLOWCHART] Failed to initialize Gemini: {e}")
            return None
    
    return _gemini_client


def _generate_with_gemini(topic: str, context: str = "") -> Optional[str]:
    """
    Generate a Mermaid flowchart using Gemini API.
    
    Args:
        topic: The topic to generate a flowchart for
        context: Optional additional context from the answer
        
    Returns:
        Mermaid flowchart code or None if generation fails
    """
    client = _get_gemini_client()
    if client is None:
        return None
    
    prompt = f'''Create a Mermaid flowchart for: "{topic}"

CRITICAL LABEL RULES:
- Keep ALL node labels SHORT - maximum 20-25 characters
- Use abbreviations if needed
- NEVER use parentheses () in labels
- NEVER use emojis
- For dates: write "1789" not "in 1789"

MERMAID SYNTAX:
- Start with: graph TD
- Use square brackets: [Short Label]
- Use --> for arrows
- Subgraph: subgraph ID [Title] ... end

STRUCTURE:
- 8-12 nodes total
- Group with subgraphs: Causes, Events, Effects
- Show cause-effect relationships

STYLING:
- Causes: fill:#fff3e0,stroke:#ff9800
- Events: fill:#e8f5e9,stroke:#4caf50  
- Effects: fill:#e3f2fd,stroke:#2196f3

{f"CONTEXT: {context[:300]}" if context else ""}

OUTPUT: Only valid Mermaid code, no markdown blocks.

EXAMPLE:
graph TD
    subgraph Causes [Root Causes]
        C1[Financial Crisis]
        C2[Social Inequality]
        C3[Political Issues]
    end
    
    C1 --> T[Trigger Event]
    C2 --> T
    C3 --> T
    
    T --> E1[Key Event 1]
    E1 --> E2[Key Event 2]
    
    subgraph Effects [Outcomes]
        O1[Short-term Effect]
        O2[Long-term Impact]
    end
    
    E2 --> O1
    E2 --> O2
    
    style Causes fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style Effects fill:#e3f2fd,stroke:#2196f3,stroke-width:2px'''
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Extract the flowchart code
        flowchart = response.text.strip()
        
        # Clean up: remove markdown code blocks if present
        flowchart = re.sub(r'^```mermaid\s*', '', flowchart)
        flowchart = re.sub(r'^```\s*', '', flowchart)
        flowchart = re.sub(r'\s*```$', '', flowchart)
        flowchart = flowchart.strip()
        
        # Sanitize: Remove emojis and problematic characters
        # Remove all emoji characters
        flowchart = re.sub(r'[\U0001F300-\U0001F9FF]', '', flowchart)
        flowchart = re.sub(r'[\u2600-\u26FF]', '', flowchart)
        flowchart = re.sub(r'[\u2700-\u27BF]', '', flowchart)
        
        # Fix parentheses in labels: [Text (1789)] -> [Text - 1789]
        flowchart = re.sub(r'\[([^\]]*)\(([^)]*)\)([^\]]*)\]', r'[\1- \2\3]', flowchart)
        
        # Remove any remaining special characters that might break parsing
        # Keep alphanumeric, common punctuation, arrows, brackets
        
        # Validate it starts with a valid Mermaid directive
        if flowchart.startswith(('graph ', 'flowchart ', 'sequenceDiagram', 'classDiagram', 'stateDiagram')):
            logger.info(f"[FLOWCHART] ✅ Generated Gemini flowchart for: {topic[:50]}")
            return flowchart
        else:
            logger.warning(f"[FLOWCHART] Invalid Mermaid output from Gemini: {flowchart[:100]}")
            return None
            
    except Exception as e:
        logger.error(f"[FLOWCHART] Gemini API error: {e}")
        return None


# ============================================================================
# Main Entry Point
# ============================================================================

def generate_concept_flowchart(question: str, answer: str, subject: Optional[str] = None) -> Optional[str]:
    """
    Generate a Mermaid flowchart to visualize the concept explained in the answer.
    
    Uses Gemini AI for dynamic generation, falls back to templates if unavailable.
    
    Args:
        question: The user's question
        answer: The AI's answer
        subject: Optional subject hint
        
    Returns:
        Mermaid flowchart code or None if not applicable
    """
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Check if the question/answer is about a process, law, or concept that benefits from a flowchart
    flowchart_keywords = [
        'law', 'process', 'how does', 'explain', 'steps', 'cycle',
        'principle', 'theory', 'mechanism', 'what is', 'define',
        'newton', 'force', 'motion', 'energy', 'reaction', 'photosynthesis',
        'digestion', 'respiration', 'circuit', 'algorithm', 'function',
        'revolution', 'war', 'history', 'cause', 'effect', 'machine learning',
        'neural network', 'data', 'programming', 'biology', 'chemistry', 'physics'
    ]
    
    should_generate = any(keyword in question_lower or keyword in answer_lower for keyword in flowchart_keywords)
    
    if not should_generate:
        return None
    
    # Try Gemini API first for dynamic generation
    gemini_result = _generate_with_gemini(question, answer[:500])
    if gemini_result:
        return gemini_result
    
    # Fall back to template-based generation
    logger.info(f"[FLOWCHART] Using template fallback for: {question[:50]}")
    return _generate_topic_flowchart(question, answer, subject)


# ============================================================================
# Template-based Fallback (when Gemini is unavailable)
# ============================================================================

def _generate_topic_flowchart(question: str, answer: str, subject: Optional[str]) -> str:
    """Generate a topic-specific flowchart using templates."""
    question_lower = question.lower()
    
    # Newton's Laws
    if 'newton' in question_lower and ('first' in question_lower or 'inertia' in question_lower):
        return """graph TD
    Start((Object's State)) --> Motion[In Motion]
    Start --> Rest[At Rest]
    
    Rest --> NoForce1[Net Force = 0]
    NoForce1 --> StayRest[Remains at Rest]
    
    Motion --> NoForce2[Net Force = 0]
    NoForce2 --> ConstV[Constant Velocity<br/>Same Speed & Direction]
    
    Rest --> ExtForce1[External Unbalanced Force]
    ExtForce1 --> Accel1[Change in Motion]
    
    Motion --> ExtForce2[External Unbalanced Force]
    ExtForce2 --> Accel2[Change in Motion]
    
    subgraph LawOfInertia [Newton's First Law - Law of Inertia]
    StayRest
    ConstV
    end
    
    style LawOfInertia fill:#e8f5e9,stroke:#4caf50,stroke-width:2px"""
    
    if 'newton' in question_lower and 'second' in question_lower:
        return """graph TD
    F((Force Applied)) --> Mass[Object with Mass m]
    Mass --> Accel[Acceleration Produced]
    
    Accel --> Formula[F = m × a]
    
    subgraph Newton2 [Newton's Second Law]
    Formula --> MoreF[More Force → More Acceleration]
    Formula --> MoreM[More Mass → Less Acceleration]
    end
    
    style Newton2 fill:#e3f2fd,stroke:#2196f3,stroke-width:2px"""
    
    if 'newton' in question_lower and 'third' in question_lower:
        return """graph LR
    A[Object A] -->|Action Force| B[Object B]
    B -->|Reaction Force<br/>Equal & Opposite| A
    
    subgraph Newton3 [Newton's Third Law]
    direction TB
    Ex1[Push wall → Wall pushes back]
    Ex2[Rocket exhaust down → Rocket goes up]
    end
    
    style Newton3 fill:#fff3e0,stroke:#ff9800,stroke-width:2px"""
    
    # Photosynthesis
    if 'photosynthesis' in question_lower:
        return """graph TD
    Sun[☀️ Sunlight] --> Leaf[🌿 Leaf/Chloroplast]
    CO2[CO₂ from Air] --> Leaf
    H2O[💧 Water from Roots] --> Leaf
    
    Leaf --> Process[Photosynthesis]
    
    Process --> Glucose[🍬 Glucose<br/>C₆H₁₂O₆]
    Process --> O2[🌬️ Oxygen Released]
    
    Glucose --> Energy[Energy for Plant Growth]
    
    subgraph Equation [Chemical Equation]
    Eq[6CO₂ + 6H₂O + Light → C₆H₁₂O₆ + 6O₂]
    end
    
    style Equation fill:#c8e6c9,stroke:#4caf50,stroke-width:2px"""
    
    # Cell Division / Mitosis
    if 'mitosis' in question_lower or 'cell division' in question_lower:
        return """graph TD
    Interphase[Interphase<br/>Cell Grows & DNA Replicates] --> Prophase
    
    Prophase[Prophase<br/>Chromosomes Condense] --> Metaphase
    Metaphase[Metaphase<br/>Chromosomes Align at Center] --> Anaphase
    Anaphase[Anaphase<br/>Chromosomes Separate] --> Telophase
    Telophase[Telophase<br/>Nuclear Membrane Forms] --> Cytokinesis
    Cytokinesis[Cytokinesis<br/>Cell Divides] --> TwoCells[Two Identical Daughter Cells]
    
    style Interphase fill:#e3f2fd,stroke:#2196f3
    style TwoCells fill:#c8e6c9,stroke:#4caf50"""
    
    # French Revolution
    if 'french revolution' in question_lower or ('french' in question_lower and 'revolution' in question_lower):
        return """graph TD
    Causes[🏛️ Causes of French Revolution]
    
    Causes --> Financial[💰 Financial Crisis<br/>Heavy debts, poor economy]
    Causes --> Social[👥 Social Inequality<br/>Three Estates System]
    Causes --> Political[⚖️ Political Issues<br/>Absolute monarchy]
    Causes --> Ideas[💡 Enlightenment Ideas<br/>Liberty, equality, democracy]
    
    Financial --> Unrest[Popular Unrest]
    Social --> Unrest
    Political --> Unrest
    Ideas --> Unrest
    
    Unrest --> Bastille[🏰 Storming of Bastille<br/>July 14, 1789]
    Bastille --> DeclarationRights[📜 Declaration of Rights of Man]
    DeclarationRights --> EndMonarchy[👑 End of Absolute Monarchy]
    EndMonarchy --> Republic[🇫🇷 French Republic]
    
    subgraph Effects [Key Outcomes]
    Republic --> Liberty[Liberty & Equality]
    Republic --> Nationalism[Rise of Nationalism]
    Republic --> Napoleon[Napoleon's Rise]
    end
    
    style Causes fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style Effects fill:#e3f2fd,stroke:#2196f3,stroke-width:2px"""
    
    # Water Cycle
    if 'water cycle' in question_lower:
        return """graph TD
    Ocean[🌊 Ocean/Water Bodies] -->|Evaporation| Vapor[☁️ Water Vapor]
    Vapor -->|Condensation| Clouds[🌥️ Clouds]
    Clouds -->|Precipitation| Rain[🌧️ Rain/Snow]
    Rain -->|Collection| Ground[🏔️ Ground/Rivers]
    Ground -->|Runoff| Ocean
    
    Ground -->|Infiltration| Underground[💧 Underground Water]
    Underground -->|Springs| Ground
    
    style Ocean fill:#e3f2fd,stroke:#2196f3
    style Clouds fill:#eceff1,stroke:#607d8b"""
    
    # Default concept flowchart for explanation questions
    return """graph TD
    Question[❓ Your Question] --> Analysis[📚 Finding Relevant Information]
    Analysis --> Context[📖 Context from Study Materials]
    Context --> Answer[✅ Generated Answer]
    
    Answer --> KeyConcepts[🔑 Key Concepts]
    Answer --> Examples[💡 Examples]
    Answer --> Practice[📝 Practice Problems]
    
    subgraph Learning [Understanding the Concept]
    KeyConcepts
    Examples
    Practice
    end
    
    style Learning fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px"""
