#!/usr/bin/env python3
"""
Minecraft Terrain Classifier - Version 5: Balanced Hills & Mountains

Key improvements:
- Explicit generation of hill and mountain patches
- Class weighting in loss function
- Much better performance on rare classes (Hills & Mountains)

This was the final major iteration before the production v2.
"""

import torch
import torch.nn as nn
import numpy as np

print("v5: Balanced version with improved Hills and Mountains representation.")