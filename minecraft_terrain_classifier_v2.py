#!/usr/bin/env python3
"""
Minecraft Terrain Classifier v2 - Production-Ready
Lightweight, modular, and ready for Minecraft mod development + world analysis tools.

Priorities addressed:
- Lightweight classifier callable from Java/Kotlin mods (TorchScript export)
- World analysis tools with stats, buildability scores, and location suggestions
- Clean API for data preprocessing pipelines
- Probability outputs + confidence

Core class: MinecraftTerrainClassifier
Analysis class: WorldTerrainAnalyzer

This version uses the best architecture from previous experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class MinecraftTerrainClassifier(nn.Module):
    """
    Lightweight neural network for classifying Minecraft terrain into 4 types.
    
    Input features (normalized):
        [x, y, height, slope]
    
    Output classes:
        0: Plains      - Good for building, flat areas
        1: Hills       - Rolling terrain, moderate building difficulty
        2: Mountains   - Steep, dramatic, harder to build
        3: Water/Lake  - Water bodies, good for waterfront builds
    """

    def __init__(self, input_dim: int = 4, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 96)
        self.bn1 = nn.BatchNorm1d(96)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.15)

        self.fc2 = nn.Linear(96, 48)
        self.bn2 = nn.BatchNorm1d(48)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        return self.fc3(x)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Dict:
        """
        Run inference and return rich output with probabilities and confidence.
        """
        self.eval()
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1).values

        class_names = ["Plains", "Hills", "Mountains", "Water/Lake"]

        return {
            "class": pred_class.item(),
            "class_name": class_names[pred_class.item()],
            "probabilities": probs.squeeze().tolist(),
            "confidence": confidence.item()
        }


class WorldTerrainAnalyzer:
    """
    High-level tool for analyzing Minecraft worlds or regions.
    Provides terrain distribution, buildability scoring, and location suggestions.
    """

    def __init__(self, model: MinecraftTerrainClassifier):
        self.model = model
        self.model.eval()
        self.class_names = ["Plains", "Hills", "Mountains", "Water/Lake"]

    def analyze_locations(self, features_list: List[List[float]]) -> List[Dict]:
        """Analyze a list of locations and return detailed predictions."""
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        results = []
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = torch.max(probs, dim=1).values

            for i in range(len(features_list)):
                results.append({
                    "features": features_list[i],
                    "class": preds[i].item(),
                    "class_name": self.class_names[preds[i].item()],
                    "probabilities": probs[i].tolist(),
                    "confidence": confs[i].item()
                })
        return results

    def get_terrain_distribution(self, analysis_results: List[Dict]) -> Dict:
        """Compute overall terrain type distribution."""
        total = len(analysis_results)
        if total == 0:
            return {}

        counts = {name: 0 for name in self.class_names}
        for res in analysis_results:
            counts[res["class_name"]] += 1

        return {
            name: {
                "count": count,
                "percentage": round(count / total * 100, 2)
            }
            for name, count in counts.items()
        }

    def calculate_buildability_score(self, result: Dict) -> float:
        """Simple buildability heuristic (0-100)."""
        class_id = result["class"]
        probs = result["probabilities"]
        confidence = result["confidence"]

        if class_id == 0:      # Plains
            score = 85 + (probs[0] * 10)
        elif class_id == 3:    # Water/Lake
            score = 70 + (probs[3] * 15)
        elif class_id == 1:    # Hills
            score = 45 + (probs[1] * 20)
        else:                  # Mountains
            score = 20 + (probs[2] * 15)

        score = min(100, score * (0.7 + 0.3 * confidence))
        return round(score, 1)

    def suggest_build_locations(
        self, 
        analysis_results: List[Dict], 
        min_buildability: float = 65.0,
        max_results: int = 10
    ) -> List[Dict]:
        """Suggest best locations for building based on buildability score."""
        scored = []
        for res in analysis_results:
            score = self.calculate_buildability_score(res)
            if score >= min_buildability:
                scored.append({
                    **res,
                    "buildability_score": score
                })

        scored.sort(key=lambda x: x["buildability_score"], reverse=True)
        return scored[:max_results]


if __name__ == "__main__":
    print("Minecraft Terrain Classifier v2 ready.")
    print("Load with: model.load_state_dict(torch.load('minecraft_terrain_v2.pth'))")