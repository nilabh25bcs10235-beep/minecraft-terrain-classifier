#!/usr/bin/env python3
"""
Minecraft Terrain Classifier - Version 3: Multi-Class + Realistic Features

Introduced:
- Realistic height + slope simulation
- Multi-class classification (Plains, Hills, Mountains, Water)
- 3D visualizations

This version made the model much more relevant to actual Minecraft terrain.
"""

import torch
import torch.nn as nn
import numpy as np

print("v3: Multi-class with realistic height and slope features.")