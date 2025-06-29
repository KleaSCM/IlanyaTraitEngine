"""
Ilanya Trait Engine - Trait Data Structures

Core data structures for representing traits as vectors and matrices.
Provides TraitVector, TraitMatrix, TraitData classes and TraitDataBuilder
for constructing and manipulating trait data for neural network processing.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from dataclasses import dataclass, field
from .trait_types import TraitType, TraitDimension, TraitCategory


@dataclass
class TraitVector:
    # Vector representation of a single trait.
    
    # Represents a single personality trait with its current value, confidence,
    # and additional metadata. This is the fundamental unit for trait processing
    # in the neural network.
    
    trait_type: TraitType                    # Type of trait (e.g., OPENNESS, CREATIVITY)
    value: float                             # Current trait value (0-1 scale)
    confidence: float                        # Confidence in the measurement (0-1 scale)
    dimensions: Dict[TraitDimension, float] = field(default_factory=dict)  # Additional trait dimensions
    metadata: Dict[str, Any] = field(default_factory=dict)                 # Custom metadata
    
    def __post_init__(self):
        # Validate trait vector data after initialization.
        
        # Ensures that trait values and confidence scores are within valid ranges.
        # Raises ValueError if validation fails.
        
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Trait value must be between 0 and 1, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def to_tensor(self) -> torch.Tensor:
    
        # Convert to PyTorch tensor for neural network processing.
        
        # Returns:
        #     Tensor containing [value, confidence] for neural network input
        
        return torch.tensor([self.value, self.confidence], dtype=torch.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        # Convert to dictionary representation for serialization.
        
        # Returns:
        #     Dictionary containing all trait vector data
        
        return {
            'trait_type': self.trait_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'dimensions': {dim.value: val for dim, val in self.dimensions.items()},
            'metadata': self.metadata
        }


@dataclass
class TraitMatrix:
    # Matrix representation of multiple traits and their interactions.
    
    # Contains a collection of trait vectors and an interaction matrix that
    # represents how traits influence each other. This is the primary data
    # structure for batch processing multiple traits.
    
    traits: Dict[TraitType, TraitVector]     # Dictionary of trait vectors
    interaction_matrix: Optional[np.ndarray] = None  # NxN matrix of trait interactions
    timestamp: Optional[float] = None        # When this matrix was created
    
    def __post_init__(self):
        # Initialize interaction matrix if not provided.
        
        # Creates a zero matrix of appropriate size if no interaction matrix
        # is provided during initialization.
        
        if self.interaction_matrix is None:
            num_traits = len(self.traits)
            self.interaction_matrix = np.zeros((num_traits, num_traits))
    
    def get_trait_values(self) -> np.ndarray:

        # Get array of trait values for processing.
        # Returns: Numpy array containing all trait values in order
        
        return np.array([trait.value for trait in self.traits.values()])
    
    def get_trait_confidences(self) -> np.ndarray:
        
        # Get array of trait confidences for processing.
        
        # Returns:
        #     Numpy array containing all trait confidence scores in order
        
        return np.array([trait.confidence for trait in self.traits.values()])
    
    def get_trait_types(self) -> List[TraitType]:
        
        # Get list of trait types in order.
        
        # Returns:
        #     List of trait types in the order they appear in the matrix
        
        return list(self.traits.keys())
    
    def to_tensor(self) -> torch.Tensor:
            # Convert to PyTorch tensor for neural network input.
        
        # Concatenates trait values and confidences into a single tensor.
        
        # Returns:
        #     Tensor containing [values, confidences] for neural network processing
        
        values = self.get_trait_values()
        confidences = self.get_trait_confidences()
        return torch.tensor(np.concatenate([values, confidences]), dtype=torch.float32)
    
    def update_trait(self, trait_type: TraitType, value: float, confidence: float):
        
        # Update a specific trait in the matrix.
        
        # Args:
        #     trait_type: The trait to update
        #     value: New trait value (0-1)
        #     confidence: New confidence score (0-1)
        
        if trait_type in self.traits:
            # Update existing trait
            self.traits[trait_type].value = value
            self.traits[trait_type].confidence = confidence
        else:
            # Create new trait vector
            self.traits[trait_type] = TraitVector(trait_type, value, confidence)
    
    def get_trait(self, trait_type: TraitType) -> Optional[TraitVector]:
    
        # Get a specific trait vector from the matrix.
        
        # Args:
        #     trait_type: The trait to retrieve
            
        # Returns:
        #     TraitVector if found, None otherwise
        
        return self.traits.get(trait_type)
    
    def to_dict(self) -> Dict[str, Any]:
        
        # Convert to dictionary representation for serialization.
        # Returns:
        #     Dictionary containing all trait matrix data
        return {
            'traits': {trait_type.value: trait.to_dict() for trait_type, trait in self.traits.items()},
            'interaction_matrix': self.interaction_matrix.tolist() if self.interaction_matrix is not None else None,
            'timestamp': self.timestamp
        }

@dataclass
class TraitData:

    # Complete trait data structure with metadata and processing information.
    
    # Wraps a TraitMatrix with additional context information including source,
    # processing metadata, and version information. This is the top-level data
    # structure used throughout the trait engine.
    
    trait_matrix: TraitMatrix                 # Core trait data
    source: str = "unknown"                   # Data source identifier
    processing_metadata: Dict[str, Any] = field(default_factory=dict)  # Processing context
    version: str = "1.0"                      # Data format version
    
    def get_input_vector(self) -> torch.Tensor:
    
        # Get input vector for neural network processing.
        
        # Returns:
        #     Tensor ready for neural network input
        return self.trait_matrix.to_tensor()
    
    def get_trait_count(self) -> int:
        
        # Get number of traits in the data.
        # Returns: Total number of traits in the matrix
        return len(self.trait_matrix.traits)
    
    def get_traits_by_category(self, category: TraitCategory) -> Dict[TraitType, TraitVector]:
        
        # Get traits filtered by category.
        # Args:     category: The category to filter by        
        # Returns:Dictionary of traits belonging to the specified category
    
        from .trait_types import TRAIT_METADATA
        return {
            trait_type: trait for trait_type, trait in self.trait_matrix.traits.items()
            if trait_type in TRAIT_METADATA and TRAIT_METADATA[trait_type].category == category
        }
    
    def to_dict(self) -> Dict[str, Any]:    
        # Convert to dictionary representation for serialization.
        # Returns:Dictionary containing all trait data and metadata
    
        return {
            'trait_matrix': self.trait_matrix.to_dict(),
            'source': self.source,
            'processing_metadata': self.processing_metadata,
            'version': self.version
        }

class TraitDataBuilder:
    # Builder class for creating trait data structures.
    # Provides a fluent interface for constructing TraitData objects step by step.
    # This makes it easier to create complex trait data structures programmatically.
    
    def __init__(self):
        """Initialize an empty builder."""
        self.traits: Dict[TraitType, TraitVector] = {}  # Accumulated traits
        self.interaction_matrix: Optional[np.ndarray] = None  # Optional interaction matrix
        self.source: str = "builder"  # Default source
        self.processing_metadata: Dict[str, Any] = {}  # Processing metadata
    
    def add_trait(self, trait_type: TraitType, value: float, confidence: float = 1.0) -> 'TraitDataBuilder':
        # Add a trait to the builder.
        # Args:
        #     trait_type: The type of trait to add
        #     value: Trait value (0-1)
        #     confidence: Confidence score (0-1), defaults to 1.0
        # Returns:Self for method chaining
        
        self.traits[trait_type] = TraitVector(trait_type, value, confidence)
        return self
    
    def set_interaction_matrix(self, matrix: np.ndarray) -> 'TraitDataBuilder':
        # Set the interaction matrix.
        # Args:matrix: NxN numpy array representing trait interactions
        # Returns: Self for method chaining
        self.interaction_matrix = matrix
        return self
    
    def set_source(self, source: str) -> 'TraitDataBuilder':
        
        # Set the data source.
        # Args: source: Source identifier string
        # Returns:Self for method chaining
        
        self.source = source
        return self
    
    def add_metadata(self, key: str, value: Any) -> 'TraitDataBuilder':
        # Add processing metadata.
        # Args: key: Metadata key 
        # value: Metadata value
        # Returns: Self for method chaining
        self.processing_metadata[key] = value
        return self
    
    def build(self) -> TraitData:
        # Build the final TraitData object.
        # Returns: Complete TraitData object with all accumulated information
        trait_matrix = TraitMatrix(self.traits, self.interaction_matrix)
        return TraitData(trait_matrix, self.source, self.processing_metadata) 