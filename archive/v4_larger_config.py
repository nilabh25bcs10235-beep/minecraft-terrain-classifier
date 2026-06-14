#!/usr/bin/env python3
"""
Minecraft Terrain Classifier - Version 4: Larger + Less Dropout

Best performing hyperparameter configuration from experiments.
Larger hidden layers with lighter regularization.

This config achieved the highest test accuracy before balancing.
"""

import torch
import torch.nn as nn
import numpy as np

print("v4: Larger model with best hyperparameters from experiments.")