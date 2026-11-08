"""Scenario-balanced, positive-enriched sampling.

A few scenarios hold most of the anomaly pixels in a training set. With a plain shuffle and
pixel-mean losses those scenarios dominate every update. This sampler gives each scenario the
SAME expected share of draws and, inside a scenario, enriches the frames that show the anomaly
(``pos_share_within`` of the scenario's draws; the rest are its anomaly-free frames).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def scenario_weights(df: pd.DataFrame, pos_share_within: float = 0.7) -> np.ndarray:
    """Per-frame sampling weights (sum to 1). ``df`` needs ``scenario_id`` and ``anomaly_pixels``."""
    if not 0 < pos_share_within < 1:
        raise ValueError("pos_share_within must be in (0, 1)")
    if df.empty:
        raise ValueError("Cannot build sampling weights for an empty frame table")
    positive = (df["anomaly_pixels"].to_numpy() > 0)
    scenario = df["scenario_id"].to_numpy()
    scenarios = pd.unique(scenario)
    share = 1.0 / len(scenarios)

    weights = np.zeros(len(df), dtype=np.float64)
    for s in scenarios:
        in_s = scenario == s
        pos, neg = in_s & positive, in_s & ~positive
        if pos.any() and neg.any():
            weights[pos] = share * pos_share_within / pos.sum()
            weights[neg] = share * (1 - pos_share_within) / neg.sum()
        else:  # all-positive or all-empty scenario: spread its share evenly
            weights[in_s] = share / in_s.sum()
    return weights / weights.sum()


def make_scenario_balanced_sampler(df: pd.DataFrame, pos_share_within: float = 0.7,
                                   num_samples: int | None = None,
                                   seed: int = 42) -> WeightedRandomSampler:
    """Sampler over the rows of ``df`` in order (positional indices)."""
    weights = scenario_weights(df.reset_index(drop=True), pos_share_within)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=num_samples or len(df),
                                 replacement=True, generator=generator)
