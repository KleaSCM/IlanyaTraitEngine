
# Command-line interface for the Ilanya Trait Engine.


import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trait_engine import TraitEngine, TraitEngineConfig
from trait_models.trait_data import TraitDataBuilder
from trait_models.trait_types import TraitType


def create_sample_data():
    """Create sample trait data for CLI demo."""
    builder = TraitDataBuilder()
    
    # Add a few key traits
    builder.add_trait(TraitType.OPENNESS, 0.7, 0.9)
    builder.add_trait(TraitType.CREATIVITY, 0.8, 0.8)
    builder.add_trait(TraitType.ADAPTABILITY, 0.6, 0.7)
    builder.add_trait(TraitType.EMOTIONAL_STABILITY, 0.7, 0.8)
    builder.add_trait(TraitType.LEARNING_RATE, 0.9, 0.9)
    
    builder.set_source("cli_demo")
    return builder.build()


def demo_command(args):
    """Run the demo command."""
    print("=== Ilanya Trait Engine CLI Demo ===\n")
    
    # Initialize engine
    print("Initializing trait engine...")
    config = TraitEngineConfig(
        num_traits=5,
        trait_embedding_dim=32,
        num_layers=2,
        num_heads=2
    )
    engine = TraitEngine(config)
    
    # Create sample data
    print("Creating sample trait data...")
    trait_data = create_sample_data()
    
    # Process traits
    print("Processing traits...")
    results = engine.process_traits(trait_data)
    
    # Show results
    print("\nResults:")
    for trait_type, predicted_trait in results['predicted_traits'].items():
        original = trait_data.trait_matrix.traits[trait_type]
        print(f"  {trait_type.value}: {original.value:.3f} → {predicted_trait.value:.3f}")
    
    print("\nDemo completed successfully!")


def process_command(args):
    """Process trait data from file."""
    print(f"Processing trait data from: {args.input}")
    # Implementation would go here
    print("Processing not yet implemented")


def train_command(args):
    """Train the model."""
    print(f"Training model with data from: {args.data_dir}")
    # Implementation would go here
    print("Training not yet implemented")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ilanya Trait Engine - Neural Network-based Trait Processing"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demonstration")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process trait data")
    process_parser.add_argument("input", help="Input trait data file")
    process_parser.add_argument("--output", help="Output file path")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("data_dir", help="Directory containing training data")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        demo_command(args)
    elif args.command == "process":
        process_command(args)
    elif args.command == "train":
        train_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 