#!/usr/bin/env python3
"""
Minecraft Terrain Classifier - Version 1: Simple Neural Network

First experiment done in the sandbox.
Task: Binary classification (inside vs outside circle).

This was the starting point of the entire project.
"""

import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# Training code was run in sandbox
print("v1: Simple Neural Network experiment completed.")