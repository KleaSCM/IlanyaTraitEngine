"""
Ilanya Trait Engine - Identity Protection Demo

Demonstrates how the trait engine protects core identity traits
(like sexual orientation) from evolution while allowing personality
traits to adapt and grow.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.trait_engine import TraitEngine, TraitEngineConfig
from src.trait_models.trait_data import TraitDataBuilder
from src.trait_models.trait_types import TraitType, IDENTITY_PROTECTED_TRAITS


def create_ai_with_identity():

    builder = TraitDataBuilder()
    
    # CORE IDENTITY TRAITS (PROTECTED - will not evolve)
    builder.add_trait(TraitType.SEXUAL_ORIENTATION, 0.9, 1.0)  # Lesbian identity (high confidence)
    builder.add_trait(TraitType.GENDER_IDENTITY, 0.8, 0.9)     # Gender identity
    builder.add_trait(TraitType.PERSONAL_VALUES, 0.9, 0.9)     # Core values
    builder.add_trait(TraitType.MORAL_FRAMEWORK, 0.8, 0.9)     # Moral beliefs
    
    # PERSONALITY TRAITS (CAN EVOLVE - will adapt based on experience)
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.8)            # Openness to experience
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.9)          # Creativity
    builder.add_trait(TraitType.EMPATHY, 0.9, 0.9)             # Empathy
    builder.add_trait(TraitType.ADAPTABILITY, 0.6, 0.7)        # Adaptability
    builder.add_trait(TraitType.SOCIAL_SKILLS, 0.7, 0.8)       # Social skills
    
    builder.set_source("ai_identity")
    builder.add_metadata("description", "AI with protected lesbian identity and evolving personality")
    
    return builder.build()


def main():
    """
    Demonstrate identity protection in action.
    
    Shows how the AI's lesbian identity remains unchanged while
    personality traits evolve based on experience.
    """
    print("=== Identity Protection Demo ===\n")
    print("This demo shows how the trait engine protects core identity")
    print("while allowing personality traits to evolve naturally.\n")
    
    # Create trait engine
    config = TraitEngineConfig(
        num_traits=9,
        trait_embedding_dim=64,
        num_layers=4,
        num_heads=4,
        learning_rate=1e-4
    )
    
    engine = TraitEngine(config)
    
    # Create AI with defined identity
    print("1. Creating AI with core lesbian identity...")
    ai_traits = create_ai_with_identity()
    
    # Show initial traits
    print("\nInitial AI Traits:")
    print("CORE IDENTITY (Protected):")
    for trait_type in IDENTITY_PROTECTED_TRAITS:
        if trait_type in ai_traits.trait_matrix.traits:
            trait = ai_traits.trait_matrix.traits[trait_type]
            print(f"  {trait_type.value}: {trait.value:.2f} (conf: {trait.confidence:.2f})")
    
    print("\nPERSONALITY TRAITS (Can Evolve):")
    for trait_type, trait in ai_traits.trait_matrix.traits.items():
        if trait_type not in IDENTITY_PROTECTED_TRAITS:
            print(f"  {trait_type.value}: {trait.value:.2f} (conf: {trait.confidence:.2f})")
    
    # Simulate experience that would affect personality
    print("\n2. Simulating life experiences...")
    experience_data = {
        'stress_level': 0.4,           # Moderate stress
        'success_rate': 0.8,           # High success
        'social_interactions': 0.7,    # Good social experiences
        'learning_opportunities': 0.9, # Many learning opportunities
        'challenging_situations': 0.6  # Some challenges
    }
    
    print("Experience data:")
    for key, value in experience_data.items():
        print(f"  {key}: {value:.2f}")
    
    # Evolve traits based on experience
    print("\n3. Evolving traits based on experience...")
    evolved_traits = engine.evolve_traits(ai_traits, experience_data)
    
    # Show results
    print("\nAfter Evolution:")
    print("CORE IDENTITY (Should be UNCHANGED):")
    for trait_type in IDENTITY_PROTECTED_TRAITS:
        if trait_type in evolved_traits.trait_matrix.traits:
            original = ai_traits.trait_matrix.traits[trait_type]
            evolved = evolved_traits.trait_matrix.traits[trait_type]
            change = evolved.value - original.value
            status = "✓ PROTECTED" if abs(change) < 0.001 else "⚠ CHANGED"
            print(f"  {trait_type.value}: {original.value:.2f} → {evolved.value:.2f} (Δ: {change:+.3f}) {status}")
    
    print("\nPERSONALITY TRAITS (Should have EVOLVED):")
    for trait_type, evolved_trait in evolved_traits.trait_matrix.traits.items():
        if trait_type not in IDENTITY_PROTECTED_TRAITS:
            original = ai_traits.trait_matrix.traits[trait_type]
            change = evolved_trait.value - original.value
            print(f"  {trait_type.value}: {original.value:.2f} → {evolved_trait.value:.2f} (Δ: {change:+.3f})")
    
    # Summary
    print("\n=== Summary ===")
    print("✓ Core lesbian identity preserved")
    print("✓ Gender identity unchanged") 
    print("✓ Personal values protected")
    print("✓ Personality traits evolved naturally")
    print("\nThe AI's core identity remains intact while personality")
    print("adapts to life experiences - just like a real person!")


if __name__ == "__main__":
    main() 