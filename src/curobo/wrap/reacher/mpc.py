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
"""
This module contains :meth:`MpcSolver` that provides a high-level interface to for model
predictive control (MPC) for reaching Cartesian poses and also joint configurations while
avoiding obstacles. The solver uses Model Predictive Path Integral (MPPI) optimization as the
solver. MPC only optimizes locally so the robot can get stuck near joint limits or behind
obstacles. To generate global trajectories, use
:py:meth:`~curobo.wrap.reacher.motion_gen.MotionGen`.

A python example is available at :ref:`python_mpc_example`.



.. note::
    Gradient-based MPC is also implemented with L-BFGS but is highly experimental and not
    recommended for real robots.


.. raw:: html

    <p>
    <video autoplay="True" loop="True" muted="True" preload="auto" width="100%"><source src="../videos/mpc_clip.mp4" type="video/mp4"></video>
    </p>


"""

# Standard Library
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.opt.newton.lbfgs import LBFGSOpt, LBFGSOptConfig
from curobo.geom.sphere_fit import SphereFitType
from curobo.opt.particle.parallel_es import ParallelES, ParallelESConfig
from curobo.opt.particle.parallel_mppi import ParallelMPPI, ParallelMPPIConfig
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, Observation
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    merge_dict_a_into_b,
)
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.wrap.wrap_base import WrapResult
from curobo.wrap.wrap_mpc import WrapConfig, WrapMpc


@dataclass
class MpcSolverConfig:
    """Configuration dataclass for MPC."""

    #: MPC Solver.
    solver: WrapMpc

    #: Rollout function for auxiliary rollouts.
    rollout_fn: ArmReacher

    #: World Collision Checker.
    world_coll_checker: Optional[WorldCollision] = None

    #: Numeric precision and device to run computations.
    tensor_args: TensorDeviceType = TensorDeviceType()

    #: Capture full step in MPC as a single CUDA graph. This is not supported currently.
    use_cuda_graph_full_step: bool = False

    #: Robot configuration.
    robot_cfg: Union[str, dict, RobotConfig] = None


    @staticmethod
    def load_from_robot_config(
        robot_cfg: Union[Union[str, dict], RobotConfig],
        world_model: Union[Union[str, dict], WorldConfig],
        base_cfg: Optional[dict] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        compute_metrics: bool = True,
        use_cuda_graph: bool = True,
        particle_opt_iters: Optional[int] = None,
        self_collision_check: bool = True,
        collision_checker_type: Optional[CollisionCheckerType] = CollisionCheckerType.MESH,
        use_es: Optional[bool] = None,
        es_learning_rate: Optional[float] = 0.01,
        use_cuda_graph_metrics: bool = True,
        store_rollouts: bool = True,
        use_cuda_graph_full_step: bool = False,
        sync_cuda_time: bool = True,
        collision_cache: Optional[Dict[str, int]] = None,
        n_collision_envs: Optional[int] = None,
        collision_activation_distance: Optional[float] = None,
        world_coll_checker=None,
        step_dt: Optional[float] = None,
        use_lbfgs: bool = False,
        use_mppi: bool = True,
        particle_file: str = "particle_mpc.yml",
        override_particle_file: str = None,
        project_pose_to_goal_frame: bool = True,
    ):
        """Create an MPC solver configuration from robot and world configuration.

        Args:
            robot_cfg: Robot configuration. Can be a path to a YAML file or a dictionary or
                an instance of :class:`~curobo.types.robot.RobotConfig`.
            world_model: World configuration. Can be a path to a YAML file or a dictionary or
                an instance of :class:`~curobo.geom.types.WorldConfig`.
            base_cfg: Base configuration for the solver. This file is used to check constraints
                and convergence. If None, the default configuration from ``base_cfg.yml`` is used.
            tensor_args: Numeric precision and device to run computations.
            compute_metrics: Compute metrics on MPC step.
            use_cuda_graph: Use CUDA graph for the optimization step.
            particle_opt_iters: Number of iterations for the particle optimization.
            self_collision_check: Enable self-collision check during MPC optimization.
            collision_checker_type: Type of collision checker to use. See :ref:`world_collision`.
            use_es: Use Evolution Strategies (ES) solver for MPC. Highly experimental.
            es_learning_rate: Learning rate for ES solver.
            use_cuda_graph_metrics: Use CUDA graph for computing metrics.
            store_rollouts: Store rollouts information for debugging. This will also store the
                trajectory taken by the end-effector across the horizon.
            use_cuda_graph_full_step: Capture full step in MPC as a single CUDA graph. This is
                experimental and might not work reliably.
            sync_cuda_time: Synchronize CUDA device with host using
                :py:func:`torch.cuda.synchronize` before calculating compute time.
            collision_cache: Cache of obstacles to create to load obstacles between planning calls.
                An example: ``{"obb": 10, "mesh": 10}``, to create a cache of 10 cuboids and 10
                meshes.
            n_collision_envs: Number of collision environments to create for batched planning
                across different environments. Only used for :py:meth:`MpcSolver.setup_solve_batch_env`
                and :py:meth:`MpcSolver.setup_solve_batch_env_goalset`.
            collision_activation_distance: Distance in meters to activate collision cost. A good
                value to start with is 0.01 meters. Increase the distance if the robot needs to
                stay further away from obstacles.
            world_coll_checker: Instance of world collision checker to use for MPC. Leaving this to
                None will create a new instance of world collision checker using the provided
                :attr:`world_model`.
            step_dt: Time step to use between each step in the trajectory. If None, the default
                time step from the configuration~(`particle_mpc.yml` or `gradient_mpc.yml`)
                is used. This dt should match the control frequency at which you are sending
                commands to the robot. This dt should also be greater than than the compute
                time for a single step.
            use_lbfgs: Use L-BFGS solver for MPC. Highly experimental.
            use_mppi: Use MPPI solver for MPC.
            particle_file: Particle based MPC config file.
            override_particle_file: Optional config file for overriding the parameters in the
                particle based MPC config file.
            project_pose_to_goal_frame: Project pose to goal frame when calculating distance
                between reached and goal pose. Use this to constrain motion to specific axes
                either in the global frame or the goal frame.

        Returns:
            MpcSolverConfig: Configuration for the MPC solver.
        """

        if use_cuda_graph_full_step:
            log_error("use_cuda_graph_full_step currently is not supported")

        config_data = load_yaml(join_path(get_task_configs_path(), particle_file))
        if override_particle_file is not None:
            merge_dict_a_into_b(load_yaml(override_particle_file), config_data)
        config_data["mppi"]["n_problems"] = 1
        if step_dt is not None:
            config_data["model"]["dt_traj_params"]["base_dt"] = step_dt
        if particle_opt_iters is not None:
            config_data["mppi"]["n_iters"] = particle_opt_iters

        if base_cfg is None:
            base_cfg = load_yaml(join_path(get_task_configs_path(), "base_cfg.yml"))
        if collision_cache is not None:
            base_cfg["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_cfg["world_collision_checker_cfg"]["n_envs"] = n_collision_envs

        base_cfg["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        base_cfg["convergence"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        if collision_activation_distance is not None:
            config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance

        if not self_collision_check:
            base_cfg["constraint"]["self_collision_cfg"]["weight"] = 0.0
            config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
        if collision_checker_type is not None:
            base_cfg["world_collision_checker_cfg"]["checker_type"] = collision_checker_type
        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        if isinstance(world_model, str):
            world_model = load_yaml(join_path(get_world_configs_path(), world_model))

        if world_coll_checker is None and world_model is not None:
            world_model = WorldCollisionConfig.load_from_dict(
                base_cfg["world_collision_checker_cfg"], world_model, tensor_args
            )
            world_coll_checker = create_collision_checker(world_model)
        grad_config_data = None
        if use_lbfgs:
            grad_config_data = load_yaml(join_path(get_task_configs_path(), "gradient_mpc.yml"))
            if step_dt is not None:
                grad_config_data["model"]["dt_traj_params"]["base_dt"] = step_dt
                grad_config_data["model"]["dt_traj_params"]["max_dt"] = step_dt

            config_data["model"] = grad_config_data["model"]
            if use_cuda_graph is not None:
                grad_config_data["lbfgs"]["use_cuda_graph"] = use_cuda_graph
            grad_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame

        cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_cfg["constraint"],
            base_cfg["convergence"],
            base_cfg["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        safety_cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_cfg["constraint"],
            base_cfg["convergence"],
            base_cfg["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        aux_cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_cfg["constraint"],
            base_cfg["convergence"],
            base_cfg["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        arm_rollout_mppi = ArmReacher(cfg)
        arm_rollout_safety = ArmReacher(safety_cfg)
        arm_rollout_aux = ArmReacher(aux_cfg)
        config_data["mppi"]["store_rollouts"] = store_rollouts
        if use_cuda_graph is not None:
            config_data["mppi"]["use_cuda_graph"] = use_cuda_graph
        if use_cuda_graph_full_step:
            config_data["mppi"]["sync_cuda_time"] = False
        config_dict = ParallelMPPIConfig.create_data_dict(
            config_data["mppi"], arm_rollout_mppi, tensor_args
        )
        solvers = []
        parallel_mppi = None
        if use_es is not None and use_es:
            log_warn("ES solver for MPC is highly experimental, not safe to run on real robots")

            mppi_cfg = ParallelESConfig(**config_dict)
            if es_learning_rate is not None:
                mppi_cfg.learning_rate = es_learning_rate
            parallel_mppi = ParallelES(mppi_cfg)
        elif use_mppi:
            mppi_cfg = ParallelMPPIConfig(**config_dict)
            parallel_mppi = ParallelMPPI(mppi_cfg)
        if parallel_mppi is not None:
            solvers.append(parallel_mppi)
        if use_lbfgs:
            log_warn("LBFGS solver for MPC is highly experimental, not safe to run on real robots")
            grad_cfg = ArmReacherConfig.from_dict(
                robot_cfg,
                grad_config_data["model"],
                grad_config_data["cost"],
                base_cfg["constraint"],
                base_cfg["convergence"],
                base_cfg["world_collision_checker_cfg"],
                world_model,
                world_coll_checker=world_coll_checker,
                tensor_args=tensor_args,
            )

            arm_rollout_grad = ArmReacher(grad_cfg)
            lbfgs_cfg_dict = LBFGSOptConfig.create_data_dict(
                grad_config_data["lbfgs"], arm_rollout_grad, tensor_args
            )
            lbfgs = LBFGSOpt(LBFGSOptConfig(**lbfgs_cfg_dict))
            solvers.append(lbfgs)

        mpc_cfg = WrapConfig(
            safety_rollout=arm_rollout_safety,
            optimizers=solvers,
            compute_metrics=compute_metrics,
            use_cuda_graph_metrics=use_cuda_graph_metrics,
            sync_cuda_time=sync_cuda_time,
        )
        solver = WrapMpc(mpc_cfg)
        return MpcSolverConfig(
            solver,
            tensor_args=tensor_args,
            use_cuda_graph_full_step=use_cuda_graph_full_step,
            world_coll_checker=world_coll_checker,
            rollout_fn=arm_rollout_aux,
            robot_cfg=robot_cfg,
        )


class MpcSolver(MpcSolverConfig):
    """High-level interface for Model Predictive Control (MPC).

    MPC can reach Cartesian poses and joint configurations while avoiding obstacles. The solver
    uses Model Predictive Path Integral (MPPI) optimization as the solver. MPC only optimizes
    locally so the robot can get stuck near joint limits or behind obstacles. To generate global
    trajectories, use :py:meth:`~curobo.wrap.reacher.motion_gen.MotionGen`.

    See :ref:`python_mpc_example` for an example. This MPC solver implementation can be used in the
    following steps:

    1. Create a :py:class:`~curobo.rollout.rollout_base.Goal` object with the target pose or joint
       configuration.
    2. Create a goal buffer for the problem type using :meth:`setup_solve_single`,
       :meth:`setup_solve_goalset`, :meth:`setup_solve_batch`, :meth:`setup_solve_batch_goalset`,
       :meth:`setup_solve_batch_env`, or :meth:`setup_solve_batch_env_goalset`. Pass the goal
       object from the previous step to this function. This function will update the internal
       solve state of MPC and also the goal for MPC. An augmented goal buffer is returned.
    3. Call :meth:`step` with the current joint state to get the next action.
    4. To change the goal, create a :py:class:`~curobo.types.math.Pose` object with new pose or
       :py:class:`~curobo.types.state.JointState` object with new joint configuration. Then
       copy the target into the augmented goal buffer using
       ``goal_buffer.goal_pose.copy_(new_pose)`` or ``goal_buffer.goal_state.copy_(new_state)``.
    5. Call :meth:`update_goal` with the augmented goal buffer to update the goal for MPC.
    6. Call :meth:`step` with the current joint state to get the next action.

    To dynamically change the type of goal reached between pose and joint configuration targets,
    create the goal object in step 1 with both targets and then use :meth:`enable_cspace_cost` and
    :meth:`enable_pose_cost` to enable or disable reaching joint configuration cost and pose cost.
    """

    def __init__(self, config: MpcSolverConfig) -> None:
        """Initializes the MPC solver.

        Args:
            config: Configuration parameters for MPC.
        """
        super().__init__(**vars(config))
        self.tensor_args = self.solver.rollout_fn.tensor_args
        self._goal_buffer = Goal()
        self.batch_size = -1
        self._goal_buffer = None
        self._solve_state = None
        self._col = None
        self._step_goal_buffer = None
        self._cu_state_in = None
        self._cu_seed = None
        self._cu_step_init = None
        self._cu_step_graph = None
        self._cu_result = None

    def setup_solve_single(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        """Creates a goal buffer to solve for a robot to reach target pose or joint configuration.

        Args:
            goal: goal object containing target pose or joint configuration.
            num_seeds: Number of seeds to use in the solver.

        Returns:
            Goal: Instance of augmented goal buffer.
        """
        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE, num_mpc_seeds=num_seeds, batch_size=1, n_envs=1, n_goalset=1
        )
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)

        self.update_goal(goal_buffer)

        return goal_buffer

    def setup_solve_goalset(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        """Creates a goal buffer to solve for a robot to reach a pose in a set of poses.

        Args:
            goal: goal object containing target goalset or joint configuration.
            num_seeds: Number of seeds to use in the solver.

        Returns:
            Goal: Instance of augmented goal buffer.
        """
        solve_state = ReacherSolveState(
            ReacherSolveType.GOALSET,
            num_mpc_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)
        self.update_goal(goal_buffer)
        return goal_buffer

    def setup_solve_batch(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        """Creates a goal buffer to solve for a batch of robots to reach targets.

        Args:
            goal: goal object containing target poses or joint configurations.
            num_seeds: Number of seeds to use in the solver.

        Returns:
            Goal: Instance of augmented goal buffer.
        """
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=1,
        )
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)
        self.update_goal(goal_buffer)

        return goal_buffer

    def setup_solve_batch_goalset(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        """Creates a goal buffer to solve for a batch of robots to reach a set of poses.

        Args:
            goal: goal object containing target goalset or joint configurations.
            num_seeds: Number of seeds to use in the solver.

        Returns:
            Goal: Instance of augmented goal buffer.
        """
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_GOALSET,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)
        self.update_goal(goal_buffer)

        return goal_buffer

    def setup_solve_batch_env(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        """Creates a goal buffer to solve for a batch of robots in different collision worlds.

        Args:
            goal: goal object containing target poses or joint configurations.
            num_seeds: Number of seeds to use in the solver.

        Returns:
            Goal: Instance of augmented goal buffer.
        """
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=1,
        )
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)
        self.update_goal(goal_buffer)

        return goal_buffer

    def setup_solve_batch_env_goalset(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        """Creates a goal buffer to solve for a batch of robots in different collision worlds.

        Args:
            goal: goal object containing target goalset or joint configurations.
            num_seeds: Number of seeds to use in the solver.

        Returns:
            Goal: Instance of augmented goal buffer.
        """
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV_GOALSET,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)
        self.update_goal(goal_buffer)

        return goal_buffer

    def step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
        max_attempts: int = 1,
    ):
        """Solve for the next action given the current state.

        Args:
            current_state: Current joint state of the robot.
            shift_steps: Number of steps to shift the trajectory.
            seed_traj: Initial trajectory to seed the optimization. If None, the solver
                uses the solution from the previous step.
            max_attempts: Maximum number of attempts to solve the problem.

        Returns:
            WrapResult: Result of the optimization.
        """
        converged = True

        for _ in range(max_attempts):
            result = self._step_once(current_state.clone(), shift_steps, seed_traj)
            if (
                torch.count_nonzero(torch.isnan(result.action.position)) == 0
                and torch.count_nonzero(~result.metrics.feasible) == 0
            ):
                converged = True
                break
            self.reset()
        if not converged:
            result.action.copy_(current_state)
            log_warn("MPC didn't converge")

        return result

    def update_object_pose(self, pose: Pose):
        """Update the object pose for Value-guided MPC (used for state-based approach)
        Args:
            pose: Pose object containing object position and orientation.
        """

        self.solver.update_object_pose(pose)
        self.rollout_fn.update_object_pose(pose)

    def update_observation(self, observation: Observation):
        self.solver.update_observation(observation)
        self.rollout_fn.update_observation(observation)

    def set_value_fn(self, value_fn):
        self.rollout_fn.value_cost.set_value_fn(value_fn)
        self.solver.set_value_fn(value_fn)

    def update_goal(self, goal: Goal):
        """Update the goal for MPC.

        Args:
            goal: goal object containing target pose or joint configuration. This goal instance
                should be created using one of the setup_solve functions.
        """
        self.solver.update_params(goal)

    def reset(self):
        """Reset the solver."""
        # reset warm start
        self.solver.reset()

    def reset_cuda_graph(self):
        """Reset captured CUDA graph. This does not work."""
        self.solver.reset_cuda_graph()

    def enable_cspace_cost(self, enable=True):
        """Enable or disable reaching joint configuration cost in the solver.

        Args:
        enable: Enable or disable reaching joint configuration cost. When False, cspace cost
                is disabled.
        """
        self.solver.safety_rollout.enable_cspace_cost(enable)
        self.rollout_fn.enable_cspace_cost(enable)
        for opt in self.solver.optimizers:
            opt.rollout_fn.enable_cspace_cost(enable)

    def enable_pose_cost(self, enable=True):
        """Enable or disable reaching pose cost in the solver.

        Args:
            enable: Enable or disable reaching pose cost. When False, pose cost is disabled.
        """
        self.solver.safety_rollout.enable_pose_cost(enable)
        self.rollout_fn.enable_pose_cost(enable)
        for opt in self.solver.optimizers:
            opt.rollout_fn.enable_pose_cost(enable)

    def get_active_js(
        self,
        in_js: JointState,
    ):
        """Get controlled joints indexed in MPC order from the input joint state.

        Args:
            in_js: Input joint state.

        Returns:
            JointState: Joint state with controlled joints.
        """

        opt_jnames = self.rollout_fn.joint_names
        opt_js = in_js.get_ordered_joint_state(opt_jnames)
        return opt_js

    def update_world(self, world: WorldConfig):
        """Update the collision world for the solver.

        This allows for updating the world representation as long as the new world representation
        does not have a larger number of obstacles than the :attr:`MpcSolver.collision_cache` as
        created during initialization of :class:`MpcSolverConfig`.

        Args:
            world: New collision world configuration. See :ref:`world_collision` for more details.
        """
        self.world_coll_checker.load_collision_model(world)

    def get_visual_rollouts(self):
        """Get rollouts for debugging."""
        return self.solver.optimizers[0].get_rollouts()

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
        start_state: Optional[JointState] = None,
        goal_pose: Optional[Pose] = None,
        check_validity: bool = True,
    ) -> bool:
        """Update the pose cost metric.

        Only supports for the main end-effector. Does not support for multiple links that are
        specified with `link_poses` in planning methods.

        Args:
            metric: Type and parameters for pose constraint to add.
            start_state: Start joint state for the constraint.
            goal_pose: Goal pose for the constraint.

        Returns:
            bool: True if the constraint can be added, False otherwise.
        """
        if check_validity:
            # check if constraint is valid:
            if metric.hold_partial_pose and metric.offset_tstep_fraction < 0.0:
                if start_state is None:
                    log_error("Need start state to hold partial pose")
                if goal_pose is None:
                    log_error("Need goal pose to hold partial pose")
                start_pose = self.compute_kinematics(start_state).ee_pose.clone()
                if self.project_pose_to_goal_frame:
                    # project start pose to goal frame:
                    projected_pose = goal_pose.compute_local_pose(start_pose)
                    if torch.count_nonzero(metric.hold_vec_weight[:3] > 0.0) > 0:
                        # angular distance should be zero:
                        distance = projected_pose.angular_distance(
                            Pose.from_list([0, 0, 0, 1, 0, 0, 0], tensor_args=self.tensor_args)
                        )
                        if torch.max(distance) > 0.05:
                            log_warn(
                                "Partial orientation between start and goal is not equal"
                                + str(distance)
                            )
                            return False

                    # check linear distance:
                    if (
                        torch.count_nonzero(
                            torch.abs(
                                projected_pose.position[..., metric.hold_vec_weight[3:] > 0.0]
                            )
                            > 0.005
                        )
                        > 0
                    ):
                        log_warn("Partial position between start and goal is not equal.")
                        return False
                else:
                    # project start pose to goal frame:
                    projected_position = goal_pose.position - start_pose.position
                    # check linear distance:
                    if (
                        torch.count_nonzero(
                            torch.abs(projected_position[..., metric.hold_vec_weight[3:] > 0.0])
                            > 0.005
                        )
                        > 0
                    ):
                        log_warn("Partial position between start and goal is not equal.")
                        return False

        rollout_list = []
        for opt in self.solver.optimizers:
            rollout_list.append(opt.rollout_fn)
        rollout_list += [self.solver.safety_rollout, self.rollout_fn]

        [
            rollout.update_pose_cost_metric(metric)
            for rollout in rollout_list
            if isinstance(rollout, ArmReacher)
        ]
        return True

    def compute_kinematics(self, state: JointState) -> KinematicModelState:
        """Compute kinematics for a given joint state.

        Args:
            state: Joint state of the robot. Only :attr:`JointState.position` is used.

        Returns:
            KinematicModelState: Kinematic state of the robot.
        """
        out = self.rollout_fn.compute_kinematics(state)
        return out

    @property
    def joint_names(self):
        """Get the ordered joint names of the robot."""
        return self.rollout_fn.joint_names

    @property
    def collision_cache(self) -> Dict[str, int]:
        """Returns the collision cache created by the world collision checker."""
        return self.world_coll_checker.cache

    @property
    def kinematics(self) -> CudaRobotModel:
        """Get kinematics instance of the robot."""
        return self.rollout_fn.dynamics_model.robot_model

    @property
    def world_collision(self) -> WorldCollision:
        """Get the world collision checker."""
        return self.world_coll_checker

    @property
    def project_pose_to_goal_frame(self) -> bool:
        """Check if the pose cost metric is projected to goal frame."""
        return self.rollout_fn.goal_cost.project_distance

    def _step_once(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ) -> WrapResult:
        """Solve for the next action given the current state.

        Args:
            current_state: Current joint state of the robot.
            shift_steps: Number of steps to shift the trajectory.
            seed_traj: Initial trajectory to seed the optimization. If None, the solver
                uses the solution from the previous step.

        Returns:
            WrapResult: Result of the optimization.
        """
        # Create cuda graph for whole solve step including computation of metrics
        # Including updation of goal buffers

        if self._solve_state is None:
            log_error("Need to first setup solve state before calling solve()")

        if self.use_cuda_graph_full_step:
            st_time = time.time()
            if not self._cu_step_init:
                self._initialize_cuda_graph_step(current_state, shift_steps, seed_traj)
            self._cu_state_in.copy_(current_state)
            if seed_traj is not None:
                self._cu_seed.copy_(seed_traj)
            self._cu_step_graph.replay()
            result = self._cu_result.clone()
            torch.cuda.synchronize(device=self.tensor_args.device)
            result.solve_time = time.time() - st_time
        else:
            self._step_goal_buffer.current_state.copy_(current_state)
            result = self._solve_from_solve_state(
                self._solve_state,
                self._step_goal_buffer,
                shift_steps,
                seed_traj,
            )

        return result

    def _update_solve_state_and_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
    ) -> Goal:
        """Update solve state and goal for MPC.

        Args:
            solve_state: New solve state.
            goal: New goal buffer.

        Returns:
            Goal: Updated goal buffer.
        """
        self._solve_state, self._goal_buffer, update_reference = solve_state.update_goal(
            goal,
            self._solve_state,
            self._goal_buffer,
            self.tensor_args,
        )
        if update_reference:
            self.solver.update_nproblems(self._solve_state.get_batch_size())
            self.reset()
            self.reset_cuda_graph()
            self._col = torch.arange(
                0, goal.batch, device=self.tensor_args.device, dtype=torch.long
            )
            self._step_goal_buffer = Goal(
                current_state=self._goal_buffer.current_state.clone(),
                batch_current_state_idx=self._goal_buffer.batch_current_state_idx.clone(),
            )
        return self._goal_buffer

    def _update_batch_size(self, batch_size: int):
        """Update the batch size of the solver.

        Args:
            batch_size: Number of problems to solve in parallel.
        """
        if self.batch_size != batch_size:
            self.batch_size = batch_size

    def _solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ) -> WrapResult:
        """Solve for the next action given the current state.

        Args:
            solve_state: solve state object containing information about the current MPC problem.
            goal: goal object containing target pose or joint configuration.
            shift_steps: Number of steps to shift the trajectory before optimization.
            seed_traj: Initial trajectory to seed the optimization. If None, the solver
                uses the solution from the previous step.

        Returns:
            WrapResult: Result of the optimization.
        """
        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                log_error("Batch Env is less that goal batch")

        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)

        if seed_traj is not None:
            self.solver.update_init_seed(seed_traj)

        result = self.solver.solve(goal_buffer, seed_traj, shift_steps)
        result.js_action = self.rollout_fn.get_full_dof_from_solution(result.action)
        return result

    def _mpc_step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ):
        """One step function that is used to create a CUDA graph.

        Args:
            current_state: Current joint state of the robot.
            shift_steps: Number of steps to shift the trajectory.
            seed_traj: Initial trajectory to seed the optimization. If None, the solver
                uses the solution from the previous step.

        Returns:
            WrapResult: Result of the optimization.
        """
        self._step_goal_buffer.current_state.copy_(current_state)
        result = self._solve_from_solve_state(
            self._solve_state,
            self._step_goal_buffer,
            shift_steps,
            seed_traj,
        )

        return result

    def _initialize_cuda_graph_step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ):
        """Create a CUDA graph for the full step of MPC.

        Args:
            current_state: Current joint state of the robot.
            shift_steps: Number of steps to shift the trajectory.
            seed_traj: Initial trajectory to seed the optimization. If None, the solver
                uses the solution from the previous step.
        """
        log_info("MpcSolver: Creating Cuda Graph")
        self._cu_state_in = current_state.clone()
        if seed_traj is not None:
            self._cu_seed = seed_traj.clone()
        s = torch.cuda.Stream(device=self.tensor_args.device)
        s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._cu_result = self._mpc_step(
                    self._cu_state_in, shift_steps=shift_steps, seed_traj=self._cu_seed
                )
        torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
        self.reset()
        self._cu_step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cu_step_graph, stream=s):
            self._cu_result = self._step(
                self._cu_state_in, shift_steps=shift_steps, seed_traj=self._cu_seed
            )
        self._cu_step_init = True


    def attach_objects_to_robot(
        self,
        joint_state: JointState,
        object_names: List[str],
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
        remove_obstacles_from_world_config: bool = False,
    ) -> bool:
        """Attach an object or objects from world to a robot's link.

        This method assumes that the objects exist in the world configuration. If attaching
        objects that are not in world, use :meth:`MotionGen.attach_external_objects_to_robot`.

        Args:
            joint_state: Joint state of the robot.
            object_names: Names of objects in the world to attach to the robot.
            surface_sphere_radius: Radius (in meters) to use for points sampled on surface of the
                object. A smaller radius will allow for generating motions very close to obstacles.
            link_name: Name of the link (frame) to attach the objects to. The assumption is that
                this link does not have any geometry and all spheres of this link represent
                attached objects.
            sphere_fit_type: Sphere fit algorithm to use. See :ref:`attach_object_note` for more
                details. The default method :attr:`SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE`
                voxelizes the volume of the objects and adds spheres representing the voxels, then
                samples points on the surface of the object, adds :attr:`surface_sphere_radius` to
                these points. This should be used for most cases.
            voxelize_method: Method to use for voxelization, passed to
                :py:func:`trimesh.voxel.creation.voxelize`.
            world_objects_pose_offset: Offset to apply to the object poses before attaching to the
                robot. This is useful when attaching an object that's in contact with the world.
                The offset is applied in the world frame before attaching to the robot.
            remove_obstacles_from_world_config: Remove the obstacles from the world cache after
                attaching to the robot to reduce memory usage. Note that when an object is attached
                to the robot, it's disabled in the world collision checker. This flag when enabled,
                also removes the object from world cache. For most cases, this should be set to
                False.
        """

        log_info("MG: Attach objects to robot")
        kin_state = self.compute_kinematics(joint_state)
        ee_pose = kin_state.ee_pose  # w_T_ee
        if world_objects_pose_offset is not None:
            # add offset from ee:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
            # new ee_pose:
            # w_T_ee = offset_T_w * w_T_ee
            # ee_T_w
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            log_warn(
                "MG: No spheres found, max_spheres: "
                + str(max_spheres)
                + " n_objects: "
                + str(len(object_names))
            )
            return False
        for i, x in enumerate(object_names):
            obs = self.rollout_fn.world_coll_checker.world_model.get_obstacle(x)
            if obs is None:
                log_error(
                    "Object not found in world. Object name: "
                    + x
                    + " Name of objects in world: "
                    + " ".join([i.name for i in self.rollout_fn.world_coll_checker.world_model.objects])
                )
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]

            self.world_coll_checker.enable_obstacle(enable=False, name=x)
            if remove_obstacles_from_world_config:
                self.world_model.remove_obstacle(x)
        log_info("MG: Computed spheres for attach objects to robot")

        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

        return True
    

    def attach_spheres_to_robot(
        self,
        sphere_radius: Optional[float] = None,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        """Attach spheres to robot's link.

        Args:
            sphere_radius: Radius of the spheres. Set to None if :attr:`sphere_tensor` is provided.
            sphere_tensor: Sphere x, y, z, r tensor.
            link_name: Name of the link to attach the spheres to. Note that this link should
                already have pre-allocated spheres.
        """
        self.kinematics.kinematics_config.attach_object(
            sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
        )
        
    def update_locked_joints(
        self, lock_joints, robot_config_dict
    ):
        """Update locked joints in the robot configuration.

        Use this function to update the joint values of currently locked joints between
        planning calls. This function can also be used to change which joints are locked, however
        this is only supported when the number of locked joints is the same as the original
        robot configuration as the kinematics tensors are pre-allocated.

        Args:
            lock_joints: Dictionary of joint names and values to lock.
            robot_config_dict: Robot configuration dictionary or path to robot configuration file.
        """
        if isinstance(robot_config_dict, str):
            robot_config_dict = load_yaml(join_path(get_robot_configs_path(), robot_config_dict))[
                "robot_cfg"
            ]
        if "robot_cfg" in robot_config_dict:
            robot_config_dict = robot_config_dict["robot_cfg"]
        robot_config_dict["kinematics"]["lock_joints"] = lock_joints
        robot_cfg = RobotConfig.from_dict(robot_config_dict, self.tensor_args)
        self.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
