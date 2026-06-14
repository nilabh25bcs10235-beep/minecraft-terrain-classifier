#!/usr/bin/env python3
"""
Basic usage example for Minecraft Terrain Classifier v2

This shows how to use the classifier and analyzer for real projects.
"""

from minecraft_terrain_classifier_v2 import MinecraftTerrainClassifier, WorldTerrainAnalyzer
import torch


def main():
    print("Loading Minecraft Terrain Classifier v2...")

    # Load the model
    model = MinecraftTerrainClassifier()
    try:
        model.load_state_dict(torch.load("minecraft_terrain_v2.pth"))
    except FileNotFoundError:
        print("Model weights not found. Please run the main script to generate them first.")
        return

    model.eval()
    print("Model loaded successfully!")

    # Example: Simulate some locations from a Minecraft world
    # In real use, replace this with data extracted from chunks
    sample_features = [
        [0.1, 0.2, 0.35, 0.25],   # Likely Plains
        [0.3, -0.4, 0.72, 0.85],  # Likely Mountains
        [-0.2, 0.5, 0.18, 0.15],  # Likely Water/Lake
        [0.0, 0.0, 0.48, 0.55],   # Likely Hills
    ]

    analyzer = WorldTerrainAnalyzer(model)
    results = analyzer.analyze_locations(sample_features)

    print("\nAnalysis Results:")
    for i, res in enumerate(results):
        print(f"  Location {i+1}: {res['class_name']} (confidence: {res['confidence']:.2f})")

    # Get buildability suggestions
    suggestions = analyzer.suggest_build_locations(results, min_buildability=60)
    print(f"\nSuggested build locations: {len(suggestions)}")
    for s in suggestions:
        print(f"  - {s['class_name']} | Buildability: {s['buildability_score']}")


if __name__ == "__main__":
    main()