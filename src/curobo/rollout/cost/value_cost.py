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
from curobo.geom.transform import quaternion_to_matrix, axis_angle_from_quat, compute_pose_error, apply_delta_pose, \
    rotation_6d_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d, quat_mul, quat_from_angle_axis, quaternion_to_matrix, pose_to_matrix
from curobo.util.rotation_transformer import RotationTransformer
# Local Folder
from .cost_base import CostBase, CostConfig
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType


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
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda:0')

        # self.rotation_transformer = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")
        self.gripper_penalty = 0.5

        self.value_func = None
        self.tensor_args = TensorDeviceType()

    def set_value_fn(self, value_fn):
        self.value_func = value_fn
        # self.value_func.eval()

    def set_policy_fn(self, policy_fn):
        self.policy_func = policy_fn

    def forward(self, state_batch, action_batch, ee_pos_batch, ee_quat_batch, observation: Observation, goal: Goal):
        B, T, _ = ee_pos_batch.shape
    
        if ee_pos_batch.shape[1] == 1:
            return torch.zeros(B, T, 1, device=ee_pos_batch.device)


        pos_error, axis_angle_error = compute_pose_error(
            observation.left_ee_pos[:1].repeat(B*T, 1),
            matrix_to_quaternion(rotation_6d_to_matrix(observation.left_ee_rot[:1])).repeat(B*T, 1),
            # self.rotation_transformer.inverse(observation.left_ee_rot).repeat(B*T, 1),
            ee_pos_batch.reshape(-1, 3),
            ee_quat_batch.reshape(-1, 4),
        )
        pos_error = pos_error.reshape(B, T, -1)
        
        
        axis_angle_error = axis_angle_error.reshape(B, T, -1)

        # left_gripper_qpos = state_batch.position[:, :, -1:]
        # left_gripper_qpos = torch.clamp(left_gripper_qpos, min=0.0, max=0.04)
        # left_gripper_qpos = torch.cat([left_gripper_qpos, -left_gripper_qpos], dim=-1)

        # left_gripper_velocity = state_batch.velocity[:, :, -1:]
        
        left_gripper_qpos = observation.left_gripper_qpos.reshape(-1, observation.stack_states, 2)
        left_gripper_qpos = left_gripper_qpos.repeat(B*T, 1, 1).reshape(B, T, -1)

        
        # ee_rot_batch = self.rotation_transformer.forward(ee_quat_batch)
        ee_rot_batch = matrix_to_rotation_6d(quaternion_to_matrix(ee_quat_batch.reshape(-1, 4))).reshape(B, T, -1)
        object_pos = observation.object_pos.reshape(1, -1).unsqueeze(1).repeat(B, T, 1)
        object_rot = observation.object_rot.reshape(1, -1).unsqueeze(1).repeat(B, T, 1)


        # if observation.grasped.item():
        object_quat = matrix_to_quaternion(rotation_6d_to_matrix(observation.object_rot[:1].unsqueeze(1).repeat(B, T, 1))).reshape(B, T, -1)
        grasped_object_pos, grasped_object_quat = self.apply_delta_pose(observation.object_pos[:1].unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1), 
                                                    object_quat.reshape(B*T, -1), 
                                                    torch.cat([pos_error, axis_angle_error], dim=-1).reshape(B*T, -1))
        grasped_object_pos = grasped_object_pos.reshape(B, T, -1)
        grasped_object_quat = grasped_object_quat.reshape(B, T, -1)
        grasped_object_rot = matrix_to_rotation_6d(quaternion_to_matrix(grasped_object_quat.reshape(-1, 4))).reshape(B, T, -1)

        # grasped = torch.logical_and(left_gripper_velocity < 0, observation.grasped.expand(left_gripper_velocity.shape))

        if observation.stack_states > 1:
            # If the observation is stacked, we need to repeat the grasped object position and rotation for each time step
            # stack past states
            grasped_object_pos = torch.cat([object_pos[:, 0].reshape(B, observation.stack_states, 3)[:, :observation.stack_states-1:],
                                            grasped_object_pos], dim=1)
            grasped_object_rot = torch.cat([object_rot[:, 0].reshape(B, observation.stack_states, 6)[:, :observation.stack_states-1],
                                            grasped_object_rot], dim=1)
            grasped_object_pos = grasped_object_pos.unfold(1, observation.stack_states, 1).reshape(B, T, -1)
            grasped_object_rot = grasped_object_rot.unfold(1, observation.stack_states, 1).reshape(B, T, -1)
            
        
        object_pos = torch.where(observation.grasped.unsqueeze(1).repeat(1, T, 1).bool(), grasped_object_pos, object_pos)
        object_rot = torch.where(observation.grasped.unsqueeze(1).repeat(1, T, 1).bool(), grasped_object_rot, object_rot)
        
        object_quat = matrix_to_quaternion(rotation_6d_to_matrix(object_rot.reshape(B, T, observation.stack_states, -1))).reshape(-1, 4)
        
        ee_pose = Pose(
            position=ee_pos_batch.unsqueeze(2).repeat(1, 1, observation.stack_states, 1).reshape(B*T*observation.stack_states, -1),
            quaternion=ee_quat_batch.unsqueeze(2).repeat(1, 1, observation.stack_states, 1).reshape(B*T*observation.stack_states, -1)
        )
        object_pose = Pose(
            position=object_pos.reshape(B*T*observation.stack_states, -1),
            quaternion=object_quat.reshape(B*T*observation.stack_states, -1)
        )
        world_pose_in_gripper = ee_pose.inverse()

        object_to_left_ee_pose = world_pose_in_gripper.multiply(object_pose)
        
        object_to_left_ee_pos = object_to_left_ee_pose.position
        object_to_left_ee_quat = object_to_left_ee_pose.quaternion
        object_to_left_ee_rot = object_to_left_ee_pose.get_6d_rep()
        object_to_left_ee_pos = object_to_left_ee_pos.reshape(B, T, observation.stack_states, -1)
        object_to_left_ee_rot = object_to_left_ee_rot.reshape(B, T, observation.stack_states, -1)
        
        left_ee_pos = ee_pos_batch
        left_ee_rot = ee_rot_batch
        if observation.stack_states > 1:
            left_ee_pos = torch.cat([observation.left_ee_pos.unsqueeze(0).repeat(B, 1, 1)[:, :observation.stack_states-1:, :],
                                    ee_pos_batch], dim=1)
            left_ee_rot = torch.cat([observation.left_ee_rot.unsqueeze(0).repeat(B, 1, 1)[:, :observation.stack_states-1:, :],
                                        ee_rot_batch], dim=1)
            left_ee_pos = left_ee_pos.unfold(1, observation.stack_states, 1).reshape(B, T, -1)
            left_ee_rot = left_ee_rot.unfold(1, observation.stack_states, 1).reshape(B, T, -1)
            
            
        # concat with the current state
        left_ee_pos = torch.cat([observation.left_ee_pos.unsqueeze(0).repeat(B, 1, 1), 
                                 left_ee_pos.reshape(B, T, -1)[:, :T-1]], dim=1)
        left_ee_rot = torch.cat([observation.left_ee_rot.unsqueeze(0).repeat(B, 1, 1),
                                 left_ee_rot.reshape(B, T, -1)[:, :T-1]], dim=1)
        left_gripper_qpos = torch.cat([observation.left_gripper_qpos.unsqueeze(0).repeat(B, 1, 1),
                                        left_gripper_qpos.reshape(B, T, -1)[:, :T-1]], dim=1)
        object_pos = torch.cat([observation.object_pos.unsqueeze(0).repeat(B, 1, 1),
                                object_pos.reshape(B, T, -1)[:, :T-1]], dim=1)
        object_rot = torch.cat([observation.object_rot.unsqueeze(0).repeat(B, 1, 1),
                                object_rot.reshape(B, T, -1)[:, :T-1]], dim=1)
        object_to_left_ee_pos = torch.cat([observation.object_to_left_ee_pos.unsqueeze(0).repeat(B, 1, 1),
                                            object_to_left_ee_pos.reshape(B, T, -1)[:, :T-1]], dim=1)
        object_to_left_ee_rot = torch.cat([observation.object_to_left_ee_rot.unsqueeze(0).repeat(B, 1, 1),
                                            object_to_left_ee_rot.reshape(B, T, -1)[:, :T-1]], dim=1)
        
            
        states = TensorDict(
            dict(
                left_ee_pos=left_ee_pos.reshape(B*T, -1),
                left_ee_rot=left_ee_rot.reshape(B*T, -1),
                left_gripper_qpos=left_gripper_qpos.reshape(B*T, -1),
                # left_gripper_qpos=observation.left_gripper_qpos.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1),
                object_pos=object_pos.reshape(B*T, -1) if observation.object_pos is not None else None,
                object_rot=object_rot.reshape(B*T, -1) if observation.object_rot is not None else None,
                object_to_left_ee_pos=object_to_left_ee_pos.reshape(B*T, -1),
                object_to_left_ee_rot=object_to_left_ee_rot.reshape(B*T, -1),
                # object_to_left_ee_pos=observation.object_to_left_ee_pos.unsqueeze(1).repeat(B, T, 1) if observation.object_to_left_ee_pos is not None else None,
                # object_to_left_ee_quat=observation.object_to_left_ee_quat.unsqueeze(1).repeat(B, T, 1) if observation.object_to_left_ee_quat is not None else None,
            ),
            batch_size=torch.tensor([B*T])
        )        

        batch = TensorDict(
            states=states,
            batch_size=torch.tensor([B*T])
        )

        # penalty = torch.logical_and(left_gripper_velocity > 0, observation.grasped.expand(left_gripper_velocity.shape)) * self.gripper_penalty
        # close_penalty = torch.logical_and(left_gripper_velocity < 0, 1 - observation.grasped.expand(left_gripper_velocity.shape)) * 0.01

        with torch.no_grad():
            # self.value_func.gripper_actor.eval()
            # gripper_action = self.value_func.gripper_actor(batch, std=0.0).mean
            # gripper_action = self.value_func.gripper_actor(batch)
            # action, log_prob, action_prob = self.value_func.gripper_actor.get_action(batch)
            # self.value_func.gripper_actor.train()
            value = self.value_func.predict_cost(batch, B, T, grasped=observation.grasped.item(), task=observation.task, action=state_batch.position.reshape(B*T, -1))
            
            # dist = (ee_pos_batch - object_pos.reshape(B, T, -1))**2
            # dist = dist.sum(dim=-1)
            # dist = dist.unsqueeze(-1)
            # value += dist * 0.1 * (1 - observation.grasped.unsqueeze(1).repeat(1, T, 1))
            
            # diff = (state_batch.position - observation.reference_joint_pos[:10])**2
            # diff = diff.sum(dim=-1)
            # value += 0.1 * diff.unsqueeze(0).unsqueeze(-1)
            
            # value = torch.gather(value, dim=2, index=action.unsqueeze(-1).unsqueeze(0).repeat(value.shape[0], 1, 1))
            # value = (value * action_prob).sum(dim=-1)
            # # gripepr_action = observation.gripper_action.unsqueeze(1).repeat(B, T, 1).reshape(B*T, -1).float()
            # # value = self.value_func(batch, gripepr_action)
            # value = self.value_func(batch)
            # # gripepr_action = ((observation.gripper_action + 1) / 2.).long()
            # # value = value[:, :, gripepr_action]
            # # value = value.mean(dim=0).unsqueeze(0)
            # value = value.max(dim=-1).values
            # value = value.reshape(value.shape[0], B, T, 1)
            # # print(f'value: {value.mean()}')
            
            # value *= -1
            # value += 2.
            # value = torch.clamp(value, min=0.)

            # # value = value.mean(dim=0) + value.std(dim=0) * 0.5
            # # value = value.unsqueeze(0)
            # # value += penalty.unsqueeze(0).repeat(value.shape[0], 1, 1, 1)
            # # value += close_penalty.unsqueeze(0).repeat(value.shape[0], 1, 1, 1)

        if value.shape[1] > 1:  
            cost = value[:, :, :T]
        else:
            cost = value

        cost = cost.squeeze(-1)
        # print((self.weight * cost).max())
        return self.weight * cost


    def apply_delta_pose(
        self,
        source_pos: torch.Tensor,
        source_rot: torch.Tensor,
        delta_pose: torch.Tensor,
        eps: float = 1.0e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies delta pose transformation on source pose.

        The first three elements of `delta_pose` are interpreted as cartesian position displacement.
        The remaining three elements of `delta_pose` are interpreted as orientation displacement
        in the angle-axis format.

        Args:
            source_pos: Position of source frame. Shape is (N, 3).
            source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4)..
            delta_pose: Position and orientation displacements. Shape is (N, 6).
            eps: The tolerance to consider orientation displacement as zero.

        Returns:
            A tuple containing the displaced position and orientation frames.
            Shape of the tensors are (N, 3) and (N, 4) respectively.
        """
        # number of poses given
        num_poses = source_pos.shape[0]
        device = source_pos.device

        # interpret delta_pose[:, 0:3] as target position displacements
        target_pos = source_pos + delta_pose[:, 0:3]
        # interpret delta_pose[:, 3:6] as target rotation displacements
        rot_actions = delta_pose[:, 3:6]
        angle = torch.linalg.vector_norm(rot_actions, dim=1)
        axis = rot_actions / angle.unsqueeze(-1)
        # change from axis-angle to quat convention

        identity_quat = self.identity_quat.repeat(num_poses, 1)
        rot_delta_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > eps,
            quat_from_angle_axis(angle, axis),
            identity_quat,
        )
        # TODO: Check if this is the correct order for this multiplication.
        target_rot = quat_mul(rot_delta_quat, source_rot)

        return target_pos, target_rot