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

        pos_error, axis_angle_error = compute_pose_error(
            ee_pos_batch[:, :-1].reshape(-1, 3),
            ee_quat_batch[:, :-1].reshape(-1, 4),
            ee_pos_batch[:, 1:].reshape(-1, 3),
            ee_quat_batch[:, 1:].reshape(-1, 4),
        )
        pos_error = pos_error.reshape(B, T-1, -1)
        axis_angle_error = axis_angle_error.reshape(B, T-1, -1)
        action = torch.cat([pos_error, axis_angle_error], dim=-1)
        action /= 0.05
        action = torch.clamp(action, -1, 1)            

        gripper_qpos = state_batch.position[:, :, -1:]
        gripper_qpos = torch.cat([
            gripper_qpos,
            -1*gripper_qpos
        ], dim=-1)
        
        gripper_action = torch.where((gripper_qpos[:,1:, 0] - gripper_qpos[:, :-1, 0]) > 0, -1., 1.)
        
        action = torch.cat([action, gripper_action.unsqueeze(-1)], dim=-1)
        
        
        left_ee_pos = observation.left_ee_pos.reshape(-1, observation.stack_states, 3)
        left_ee_quat = observation.left_ee_quat.reshape(-1, observation.stack_states, 4)
        left_gripper_qpos = observation.left_gripper_qpos.reshape(-1, observation.stack_states, 2)

        left_ee_pos = torch.cat([
            ee_pos_batch.unsqueeze(2),
            left_ee_pos.unsqueeze(1).repeat(B, T, 1, 1)[:, :, :-1],
        ], dim=2).reshape(B, T, -1)

        left_ee_quat = torch.cat([
            ee_quat_batch.unsqueeze(2),
            left_ee_quat.unsqueeze(1).repeat(B, T, 1, 1)[:, :, :-1],
        ], dim=2).reshape(B, T, -1)

        gripper_qpos = torch.cat([
            gripper_qpos.unsqueeze(2),
            left_gripper_qpos.unsqueeze(1).repeat(B, T, 1, 1)[:, :, :-1],
        ], dim=2)


        observation.object_to_left_ee_pos = left_ee_pos.reshape(B*T, -1) - observation.object_pos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1)

        states = TensorDict(
            dict(
                left_ee_pos=left_ee_pos.reshape(B*T, -1),
                left_ee_quat=left_ee_quat.reshape(B*T, -1),
                left_gripper_qpos=gripper_qpos.reshape(B*T, -1),
                # left_gripper_qpos=observation.left_gripper_qpos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1),
                object_pos=observation.object_pos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_pos is not None else None,
                object_quat=observation.object_quat.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_quat is not None else None,
                object_to_left_ee_pos=observation.object_to_left_ee_pos,
                # object_to_left_ee_pos=observation.object_to_left_ee_pos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_to_left_ee_pos is not None else None,
                object_to_left_ee_quat=observation.object_to_left_ee_quat.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_to_left_ee_quat is not None else None,
            )
        )
        
        states = TensorDict(
            dict(
                left_ee_pos=left_ee_pos.reshape(B*T, -1),
                left_ee_quat=left_ee_quat.reshape(B*T, -1),
                left_gripper_qpos=gripper_qpos.reshape(B*T, -1),
                # left_gripper_qpos=observation.left_gripper_qpos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1),
                object_pos=observation.object_pos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_pos is not None else None,
                object_quat=observation.object_quat.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_quat is not None else None,
                object_to_left_ee_pos=observation.object_to_left_ee_pos,
                # object_to_left_ee_pos=observation.object_to_left_ee_pos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_to_left_ee_pos is not None else None,
                object_to_left_ee_quat=observation.object_to_left_ee_quat.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1) if observation.object_to_left_ee_quat is not None else None,
            )
        )        

        batch = TensorDict(
            states=states,
        )
         
        with torch.no_grad():
            value = self.value_func(batch)
            value = value.reshape(value.shape[0], B, T, 1)

        if value.shape[1] > 1: 
            cost = value[:, :, :T]
        else:
            cost = value

        cost = cost.squeeze(-1)
        # print((self.weight * cost).max())
        return self.weight * cost