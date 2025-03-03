#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

# Standard Library
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from itertools import accumulate

# Third Party
import torch
import torch.nn.functional as F
from tensordict import TensorDict

# CuRobo
from curobo.rollout.rollout_base import Goal, Observation
from curobo.geom.transform import quaternion_to_matrix
# Local Folder
from .cost_base import CostBase, CostConfig


@dataclass
class ValueCostConfig(CostConfig):
    num_modules: int = 50
    discount_factor: float = 0.99

    def __post_init__(self):
        return super().__post_init__()


class ValueCost(CostBase, ValueCostConfig):
    def __init__(self, config: ValueCostConfig):
        ValueCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)

        self.value_func = None

    def set_value_fn(self, value_fn):
        self.value_func = value_fn
        # self.value_func.eval()


    def forward(self, state_batch, ee_pos_batch, ee_quat_batch, observation: Observation, goal: Goal):
        B, T, _ = ee_pos_batch.shape

        features = observation.features.unsqueeze(0).repeat(B*T, 1)
        pcd_mean = observation.pcd_mean
        pcd_size = observation.pcd_size.unsqueeze(0).repeat(B*T, 1)

        batch = TensorDict({
            'ee_position': ee_pos_batch - pcd_mean.repeat(B, T, 1),
            'ee_orientation': quaternion_to_matrix(ee_quat_batch.reshape(B*T, -1))[:, :2, :3].reshape(B, T, -1),
            'features': features.reshape(B, T, -1),
            'pcd_size': pcd_size.reshape(B, T, -1),
        }, device=self.tensor_args.device, batch_size=B)

        with torch.no_grad():
            value = self.value_func.predict_value(batch)

        if value.shape[1] > 1: 
            cost = value[:, :, :T]
        else:
            cost = value

        cost = cost.squeeze(-1)
        # print((self.weight * cost).max())
        return self.weight * cost