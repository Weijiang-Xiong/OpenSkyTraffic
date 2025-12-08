""" We implment a information-centric spatio-temporal traffic forecasting model based on transformer architecture.
    Suppose both the input and output data has shape (N, T, P, C), where N is the batch size, T is the temporal length, 
    P is the number of spatial points, and C is the number of channels/features.
    However, among the P locations in the input, only a subset of them are observed or monitored, denoted as M (M << P),
    while the rest are unobserved. 
    The monitored locations are indicated by a binary mask of shape (N, T, P), where 1 indicates monitored and 0 indicates unmonitored.
    What the model will have as input is the full-shape data (N, T, P, C) with unmonitored locations filled with zeros, and a monitoring 
    mask shaped (N, T, P).

    We need our model to really make use of the provided information (and thus we call it information-centric) to give meaningful predictions.

    To achieve this, we design such a transformer-based architecture:
        1. The overall architecture follows Time - Space - Time design, i.e., first model temporal dependencies for each spatial point independently,
"""

import torch
import torch.nn as nn
