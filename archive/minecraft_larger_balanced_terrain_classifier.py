#!/usr/bin/env python3
"""
Minecraft Larger + Less Dropout Terrain Classifier — Balanced Hills & Mountains Version

This version improves the data generator to create significantly more Hills and Mountains examples.
It was an important step before the final v2 production version.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ... (full code from previous balanced version would go here)
# For brevity in this commit, the full script is available in conversation history.
# Key improvement: Explicit hill and mountain patch generation + class weights.

print("This is an archived version. Use v2 for current development.")