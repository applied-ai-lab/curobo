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
import matplotlib.pyplot as plt

# Third Party
import torch
import torch.nn.functional as F
from tensordict import TensorDict

# CuRobo
from curobo.rollout.rollout_base import Goal, Observation
from curobo.geom.transform import quaternion_to_matrix, axis_angle_from_quat, compute_pose_error
from curobo.util.rotation_transformer import RotationTransformer
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

        self.rotation_transformer = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")

        self.value_func = None

    def set_value_fn(self, value_fn):
        self.value_func = value_fn
        self.value_func.eval()


    def forward(self, state_batch, ee_pos_batch, ee_quat_batch, observation: Observation, goal: Goal):
        B, T, _ = ee_pos_batch.shape
    
        if ee_pos_batch.shape[1] == 1:
            return torch.zeros(B, T, 1, device=ee_pos_batch.device)

        pos_error, axis_angle_error = compute_pose_error(
            observation.left_ee_pos.repeat(B*T, 1),
            self.rotation_transformer.inverse(observation.left_ee_rot).repeat(B*T, 1),
            ee_pos_batch.reshape(-1, 3),
            ee_quat_batch.reshape(-1, 4),
        )
        pos_error = pos_error.reshape(B, T, -1)
        
        
        axis_angle_error = axis_angle_error.reshape(B, T, -1)

        left_gripper_qpos = observation.left_gripper_qpos.reshape(-1, observation.stack_states, 2)
        left_gripper_qpos = left_gripper_qpos.repeat(B*T, 1, 1).reshape(B, T, -1)

        #gripper_qpos = state_batch.position[:, :, 7:]
        #gripper_qpos = torch.clamp(gripper_qpos, 0, 0.04)
        #gripper_qpos = torch.cat([
        #    gripper_qpos,
        #    -1*gripper_qpos
        #], dim=-1)
        
        ee_rot_batch = self.rotation_transformer.forward(ee_quat_batch)

        

        object_to_left_ee_pos = ee_pos_batch - observation.object_pos.unsqueeze(1).repeat(B, T, 1)
        # object_to_left_ee_pos = observation.object_to_left_ee_pos.unsqueeze(1).repeat(B, T, 1)

        # action = action.reshape(B*T, -1)
        
        states = TensorDict(
            dict(
                left_ee_pos=ee_pos_batch.reshape(B*T, -1),
                left_ee_rot=ee_rot_batch.reshape(B*T, -1),
                left_gripper_qpos=left_gripper_qpos.reshape(B*T, -1),
                # left_gripper_qpos=observation.left_gripper_qpos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1),
                object_pos=observation.object_pos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_pos is not None else None,
                object_rot=observation.object_rot.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_rot is not None else None,
                object_to_left_ee_pos=object_to_left_ee_pos.reshape(B*T, -1),
                # object_to_left_ee_pos=observation.object_to_left_ee_pos.unsqueeze(1).repeat(B, T, 1) if observation.object_to_left_ee_pos is not None else None,
                # object_to_left_ee_quat=observation.object_to_left_ee_quat.unsqueeze(1).repeat(B, T, 1) if observation.object_to_left_ee_quat is not None else None,
            ),
            batch_size=torch.tensor([B*T])
        )        

        batch = TensorDict(
            states=states,
            batch_size=torch.tensor([B*T])
        )

        with torch.no_grad():
            value = self.value_func(batch)
            value = value.reshape(value.shape[0], B, T, 1)
            
            value *= -1
            value += 1.
            value = torch.clamp(value, min=0.)

        if value.shape[1] > 1: 
            cost = value[:, :, :T]
        else:
            cost = value

        cost = cost.squeeze(-1)
        # print((self.weight * cost).max())
        return self.weight * cost
