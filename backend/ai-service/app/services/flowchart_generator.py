"""
Flowchart Generator Service

Generates Mermaid flowchart code from AI tutor responses to visualize concepts.
"""
from typing import Optional


def generate_concept_flowchart(question: str, answer: str, subject: Optional[str] = None) -> Optional[str]:
    """
    Generate a Mermaid flowchart to visualize the concept explained in the answer.
    
    For now, this uses simple template-based generation based on keywords.
    In production, this could use an LLM to generate proper flowcharts.
    
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
        'digestion', 'respiration', 'circuit', 'algorithm', 'function'
    ]
    
    should_generate = any(keyword in question_lower or keyword in answer_lower for keyword in flowchart_keywords)
    
    if not should_generate:
        return None
    
    # Generate flowchart based on detected topic
    return _generate_topic_flowchart(question, answer, subject)


def _generate_topic_flowchart(question: str, answer: str, subject: Optional[str]) -> str:
    """Generate a topic-specific flowchart."""
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
