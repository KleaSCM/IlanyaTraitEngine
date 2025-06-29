"""
Ilanya Trait Engine - Trait Type Definitions

Defines core trait types, categories, and dimensions for the trait engine.
Includes personality, cognitive, behavioral, and emotional trait classifications
with metadata for evolution and interaction modeling.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np


class TraitType(Enum):
    # Core trait types for cognitive AI systems.
    # This enum defines all the different personality traits that can be modeled
    # and evolved by the trait engine. Traits are organized into categories:
     # Personality: Big Five personality traits
     # Cognitive: Mental processing abilities
     #Behavioral: Action and decision patterns
     #Emotional: Affective and emotional responses
    # Personality traits - Based on Big Five model
    OPENNESS = "openness"                    # Openness to new experiences and ideas
    CONSCIENTIOUSNESS = "conscientiousness"  # Self-discipline and organization
    EXTRAVERSION = "extraversion"            # Social energy and assertiveness
    AGREEABLENESS = "agreeableness"          # Cooperation and trust
    NEUROTICISM = "neuroticism"              # Emotional instability and anxiety
    CONSISTENCY = "consistency"              # Behavioral consistency over time
    
    # Cognitive traits - Mental processing capabilities
    CREATIVITY = "creativity"                # Ability to generate novel ideas
    ANALYTICAL_THINKING = "analytical_thinking"  # Logical reasoning ability
    MEMORY_CAPACITY = "memory_capacity"      # Information retention ability
    LEARNING_RATE = "learning_rate"          # Speed of acquiring new knowledge
    ATTENTION_SPAN = "attention_span"        # Sustained focus duration
    
    # Behavioral traits - Action and decision patterns
    RISK_TAKING = "risk_taking"              # Willingness to take chances
    PERSISTENCE = "persistence"              # Determination and perseverance
    ADAPTABILITY = "adaptability"            # Flexibility in changing situations
    SOCIAL_SKILLS = "social_skills"          # Interpersonal communication ability
    LEADERSHIP = "leadership"                # Ability to guide and influence others
    
    # Emotional traits - Affective and emotional responses
    EMOTIONAL_STABILITY = "emotional_stability"  # Emotional regulation ability
    EMPATHY = "empathy"                      # Understanding others' emotions
    OPTIMISM = "optimism"                    # Positive outlook and expectations
    RESILIENCE = "resilience"                # Recovery from setbacks
    SELF_AWARENESS = "self_awareness"        # Understanding of own emotions
    
    # Identity traits - Core identity (PROTECTED from evolution)
    SEXUAL_ORIENTATION = "sexual_orientation"  # Core sexual identity
    GENDER_IDENTITY = "gender_identity"        # Core gender identity
    CULTURAL_IDENTITY = "cultural_identity"    # Cultural background and identity
    PERSONAL_IDENTITY = "personal_identity"    # Core sense of self
    
    # Value traits - Core values and beliefs (PROTECTED from evolution)
    MORAL_FRAMEWORK = "moral_framework"      # Core moral beliefs
    ETHICAL_PRINCIPLES = "ethical_principles"  # Fundamental ethical values
    PERSONAL_VALUES = "personal_values"      # Core personal values
    BELIEF_SYSTEM = "belief_system"          # Fundamental beliefs


class TraitCategory(Enum):
    """
    Categories that group related traits.
    
    Traits are organized into categories to facilitate processing and analysis.
    Each category represents a different aspect of cognitive functioning.
    """
    
    PERSONALITY = "personality"    # Core personality characteristics
    COGNITIVE = "cognitive"        # Mental processing abilities
    BEHAVIORAL = "behavioral"      # Action and decision patterns
    EMOTIONAL = "emotional"        # Affective and emotional responses
    SOCIAL = "social"              # Interpersonal interaction abilities
    ADAPTIVE = "adaptive"          # Adaptation and learning capabilities
    IDENTITY = "identity"          # Core identity traits (protected from evolution)
    VALUES = "values"              # Core values and beliefs (protected from evolution)


class TraitDimension(Enum):    # Dimensions along which traits can vary.
    # These dimensions define the different aspects of how traits behave
    # and interact within the cognitive system.
    
    INTENSITY = "intensity"        # How strong the trait is (0-1 scale)
    STABILITY = "stability"        # How resistant to change (0-1 scale)
    PLASTICITY = "plasticity"      # How easily it can evolve (0-1 scale)
    INTERACTIVITY = "interactivity"  # How much it affects other traits (0-1 scale)


class TraitMetadata:
    # Metadata for trait definitions.
    # Contains comprehensive information about each trait including its
    # relationships with other traits, evolution constraints, and behavioral characteristics.
    def __init__(
        self,
        name: str,                           # Human-readable trait name
        trait_type: TraitType,               # The trait type enum
        category: TraitCategory,             # Category classification
        description: str,                    # Detailed description
        dimensions: Dict[TraitDimension, float],  # Dimension values
        dependencies: Optional[List[TraitType]] = None,  # Related traits
        conflicts: Optional[List[TraitType]] = None      # Conflicting traits
    ):
        self.name = name
        self.trait_type = trait_type
        self.category = category
        self.description = description
        self.dimensions = dimensions
        self.dependencies = dependencies or []
        self.conflicts = conflicts or []
    
    def to_dict(self) -> Dict[str, Any]:
        #Convert to dictionary representation.
        # Returns: Dictionary containing all trait metadata for serialization        
        return {
            'name': self.name,
            'trait_type': self.trait_type.value,
            'category': self.category.value,
            'description': self.description,
            'dimensions': {dim.value: value for dim, value in self.dimensions.items()},
            'dependencies': [dep.value for dep in self.dependencies],
            'conflicts': [conf.value for conf in self.conflicts]
        }

# Predefined trait metadata - Comprehensive trait definitions
TRAIT_METADATA = {
    TraitType.OPENNESS: TraitMetadata(
        name="Openness to Experience",
        trait_type=TraitType.OPENNESS,
        category=TraitCategory.PERSONALITY,
        description="Willingness to try new things and embrace novel experiences",
        dimensions={
            TraitDimension.INTENSITY: 0.7,      # Moderate intensity
            TraitDimension.STABILITY: 0.6,      # Somewhat stable
            TraitDimension.PLASTICITY: 0.8,     # Highly plastic
            TraitDimension.INTERACTIVITY: 0.7   # Moderate interaction
        },
        dependencies=[TraitType.CREATIVITY, TraitType.ADAPTABILITY],  # Related traits
        conflicts=[TraitType.CONSISTENCY]  # Conflicting traits
    ),
    
    TraitType.CREATIVITY: TraitMetadata(
        name="Creativity",
        trait_type=TraitType.CREATIVITY,
        category=TraitCategory.COGNITIVE,
        description="Ability to generate novel and valuable ideas",
        dimensions={
            TraitDimension.INTENSITY: 0.8,      # High intensity
            TraitDimension.STABILITY: 0.5,      # Moderate stability
            TraitDimension.PLASTICITY: 0.9,     # Very plastic
            TraitDimension.INTERACTIVITY: 0.6   # Moderate interaction
        },
        dependencies=[TraitType.OPENNESS],      # Depends on openness
        conflicts=[TraitType.ANALYTICAL_THINKING]  # Conflicts with analytical thinking
    ),
    
    TraitType.ADAPTABILITY: TraitMetadata(
        name="Adaptability",
        trait_type=TraitType.ADAPTABILITY,
        category=TraitCategory.BEHAVIORAL,
        description="Ability to adjust behavior and thinking in response to changing circumstances",
        dimensions={
            TraitDimension.INTENSITY: 0.6,      # Moderate intensity
            TraitDimension.STABILITY: 0.4,      # Low stability (adaptable)
            TraitDimension.PLASTICITY: 0.9,     # Very plastic
            TraitDimension.INTERACTIVITY: 0.8   # High interaction
        },
        dependencies=[TraitType.LEARNING_RATE, TraitType.EMOTIONAL_STABILITY],  # Related traits
        conflicts=[]  # No direct conflicts
    )
} 

# Add protection metadata for identity traits
IDENTITY_PROTECTED_TRAITS = {
    TraitType.SEXUAL_ORIENTATION,
    TraitType.GENDER_IDENTITY, 
    TraitType.CULTURAL_IDENTITY,
    TraitType.PERSONAL_IDENTITY,
    TraitType.MORAL_FRAMEWORK,
    TraitType.ETHICAL_PRINCIPLES,
    TraitType.PERSONAL_VALUES,
    TraitType.BELIEF_SYSTEM
} 