"""
Ilanya Trait Engine - Trait State Management

Trait state and cognitive state models for tracking trait evolution over time.
Provides TraitState and CognitiveState classes for monitoring trait changes,
stability, and overall cognitive state in real-time AI systems.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
from dataclasses import dataclass, field
from datetime import datetime
from .trait_types import TraitType, TraitCategory


@dataclass
class TraitState:
    """Represents the state of a trait at a specific point in time."""
    
    trait_type: TraitType
    current_value: float
    previous_value: Optional[float] = None
    change_rate: float = 0.0
    stability_score: float = 1.0
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute derived values."""
        if not 0.0 <= self.current_value <= 1.0:
            raise ValueError(f"Trait value must be between 0 and 1, got {self.current_value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Compute change rate if previous value is available
        if self.previous_value is not None:
            self.change_rate = self.current_value - self.previous_value
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.tensor([
            self.current_value,
            self.change_rate,
            self.stability_score,
            self.confidence
        ], dtype=torch.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'trait_type': self.trait_type.value,
            'current_value': self.current_value,
            'previous_value': self.previous_value,
            'change_rate': self.change_rate,
            'stability_score': self.stability_score,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class CognitiveState:
    """Represents the overall cognitive state including all traits."""
    
    trait_states: Dict[TraitType, TraitState]
    overall_stability: float = 1.0
    cognitive_load: float = 0.0
    attention_focus: float = 1.0
    emotional_state: float = 0.5  # Neutral
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute derived values."""
        if not 0.0 <= self.overall_stability <= 1.0:
            raise ValueError(f"Overall stability must be between 0 and 1, got {self.overall_stability}")
        if not 0.0 <= self.cognitive_load <= 1.0:
            raise ValueError(f"Cognitive load must be between 0 and 1, got {self.cognitive_load}")
        if not 0.0 <= self.attention_focus <= 1.0:
            raise ValueError(f"Attention focus must be between 0 and 1, got {self.attention_focus}")
        if not 0.0 <= self.emotional_state <= 1.0:
            raise ValueError(f"Emotional state must be between 0 and 1, got {self.emotional_state}")
    
    def get_trait_values(self) -> np.ndarray:
        """Get array of current trait values."""
        return np.array([state.current_value for state in self.trait_states.values()])
    
    def get_trait_change_rates(self) -> np.ndarray:
        """Get array of trait change rates."""
        return np.array([state.change_rate for state in self.trait_states.values()])
    
    def get_trait_stabilities(self) -> np.ndarray:
        """Get array of trait stability scores."""
        return np.array([state.stability_score for state in self.trait_states.values()])
    
    def get_trait_confidences(self) -> np.ndarray:
        """Get array of trait confidences."""
        return np.array([state.confidence for state in self.trait_states.values()])
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        trait_values = self.get_trait_values()
        trait_changes = self.get_trait_change_rates()
        trait_stabilities = self.get_trait_stabilities()
        trait_confidences = self.get_trait_confidences()
        
        cognitive_state = np.array([
            self.overall_stability,
            self.cognitive_load,
            self.attention_focus,
            self.emotional_state
        ])
        
        return torch.tensor(
            np.concatenate([
                trait_values,
                trait_changes,
                trait_stabilities,
                trait_confidences,
                cognitive_state
            ]),
            dtype=torch.float32
        )
    
    def get_trait_state(self, trait_type: TraitType) -> Optional[TraitState]:
        """Get state for a specific trait."""
        return self.trait_states.get(trait_type)
    
    def update_trait_state(self, trait_type: TraitType, new_value: float, confidence: float = 1.0):
        """Update the state of a specific trait."""
        current_state = self.trait_states.get(trait_type)
        previous_value = current_state.current_value if current_state else None
        
        new_state = TraitState(
            trait_type=trait_type,
            current_value=new_value,
            previous_value=previous_value,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.trait_states[trait_type] = new_state
    
    def get_traits_by_category(self, category: TraitCategory) -> Dict[TraitType, TraitState]:
        """Get trait states filtered by category."""
        from .trait_types import TRAIT_METADATA
        return {
            trait_type: state for trait_type, state in self.trait_states.items()
            if trait_type in TRAIT_METADATA and TRAIT_METADATA[trait_type].category == category
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'trait_states': {trait_type.value: state.to_dict() for trait_type, state in self.trait_states.items()},
            'overall_stability': self.overall_stability,
            'cognitive_load': self.cognitive_load,
            'attention_focus': self.attention_focus,
            'emotional_state': self.emotional_state,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        } 