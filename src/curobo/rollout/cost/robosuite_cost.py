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
from curobo.geom.transform import quaternion_to_matrix, axis_angle_from_quat, compute_pose_error
# Local Folder
from .cost_base import CostBase, CostConfig


@dataclass
class RobosuiteCostConfig(CostConfig):
    task: str = "lift"

    def __post_init__(self):
        return super().__post_init__()


class RobosuiteLiftCost(CostBase, RobosuiteCostConfig):
    def __init__(self, config: RobosuiteCostConfig):
        RobosuiteCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)

    def set_grasp_callback(self, grasp_callback):
        self.grasp_callback = grasp_callback


    def forward(self, state_batch, ee_pos_batch, ee_quat_batch, action_batch, observation: Observation):
        B, T, _ = ee_pos_batch.shape
        dist = torch.norm(ee_pos_batch-observation.object_pos, dim=-1)
        reaching_cost = torch.tanh(15 * dist)
        grasping = torch.logical_and(dist < 0.04, state_batch.velocity[:, :, 7] < 0.) * observation.grasped
        grasping = grasping.float()



        reaching_cost *= (1-observation.grasped)
        lift = observation.grasped * (1-torch.tanh(torch.norm(0.15 - ee_pos_batch[:, :, -1:], dim=-1) * 5))
        # print(lift)
        # not grasped -> 1 -> 5
        # grasped (far away) -> 0.3
        # grasped (close) -> 0.9

        # lift = torch.logical_and(grasped, )
        cost = reaching_cost + (1 - grasping) * 5. + (1 - lift) * 20.
        return self.weight * cost