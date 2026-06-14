# Minecraft Terrain Classifier

A lightweight PyTorch-based toolkit for **terrain classification** in Minecraft. Built for mod developers, world analysis tools, and procedural generation.

## What it does

- Classifies terrain into 4 types: **Plains, Hills, Mountains, Water/Lake**
- Returns probabilities + confidence score
- Includes a `WorldTerrainAnalyzer` with:
  - Terrain distribution statistics
  - Buildability scoring
  - Smart suggestions for structure placement
- Exports to **TorchScript** for easy use in Java/Kotlin mods

## Installation

```bash
git clone https://github.com/nilabh25bcs10235-beep/minecraft-terrain-classifier.git
cd minecraft-terrain-classifier
pip install -r requirements.txt





Quick Start
from minecraft_terrain_classifier_v2 import MinecraftTerrainClassifier, WorldTerrainAnalyzer
import torch

# Load model
model = MinecraftTerrainClassifier()
model.load_state_dict(torch.load("minecraft_terrain_v2.pth"))
model.eval()

# Analyze locations (replace with real data from your world)
features = [
    [0.1, 0.2, 0.35, 0.25],   # Example: likely Plains
    [0.3, -0.4, 0.72, 0.85],  # Example: likely Mountains
]

analyzer = WorldTerrainAnalyzer(model)
results = analyzer.analyze_locations(features)

# Get best build locations
suggestions = analyzer.suggest_build_locations(results, min_buildability=65)





For Minecraft Mods (Java / Kotlin)
Use the TorchScript export:
Module model = Module.load("minecraft_terrain_v2_scripted.pt");
// Pass normalized features: [x, y, height, slope]




Using with Real Minecraft Data
To use this with actual worlds, extract these 4 normalized features per location:
Feature
Description
How to get it
Normalization
x
X coordinate
Chunk/block position
Divide by ~6–10
y
Z coordinate
Chunk/block position
Divide by ~6–10
height
Elevation
Heightmap or block Y
Divide by 11
slope
Local steepness
Height difference to neighbors
Divide by 3.5





Project Structure
minecraft-terrain-classifier/
├── minecraft_terrain_classifier_v2.py   # Main production code (recommended)
├── examples/
│   └── basic_usage.py
├── archive/                             # Previous experimental versions
├── requirements.txt
└── README.md
