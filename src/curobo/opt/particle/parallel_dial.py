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
# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.opt.particle.particle_opt_base import ParticleOptBase, ParticleOptConfig, SampleMode
from curobo.opt.particle.particle_opt_utils import (
    SquashType,
    cost_to_go,
    gaussian_entropy,
    matrix_cholesky,
    scale_ctrl,
)
from curobo.rollout.rollout_base import RolloutBase, Trajectory
from curobo.types.base import TensorDeviceType
from curobo.types.robot import State
from curobo.util.logger import log_info
from curobo.util.sample_lib import HaltonSampleLib, SampleConfig, SampleLib
from curobo.util.tensor_util import copy_tensor
from curobo.util.torch_utils import get_torch_jit_decorator


class BaseActionType(Enum):
    REPEAT = 0
    NULL = 1
    RANDOM = 2


class CovType(Enum):
    SIGMA_I = 0
    DIAG_A = 1
    FULL_A = 2
    FULL_HA = 3


@dataclass
class ParallelDIALConfig(ParticleOptConfig):
    init_mean: float
    init_cov: float
    base_action: BaseActionType
    step_size_mean: float
    step_size_cov: float
    null_act_frac: float
    squash_fn: SquashType
    cov_type: CovType
    sample_params: SampleConfig
    update_cov: bool
    random_mean: bool
    beta: float
    alpha: float
    gamma: float
    kappa: float
    sample_per_problem: bool
    value_lambda: float = 60.
    value_coef: float = 200.
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5

    def __post_init__(self):
        self.init_cov = self.tensor_args.to_device(self.init_cov).unsqueeze(0)
        self.init_mean = self.tensor_args.to_device(self.init_mean).clone()
        return super().__post_init__()

    @staticmethod
    @profiler.record_function("parallel_mppi_config/create_data_dict")
    def create_data_dict(
        data_dict: Dict,
        rollout_fn: RolloutBase,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        child_dict: Optional[Dict] = None,
    ):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict = ParticleOptConfig.create_data_dict(
            data_dict, rollout_fn, tensor_args, child_dict
        )
        child_dict["base_action"] = BaseActionType[child_dict["base_action"]]
        child_dict["squash_fn"] = SquashType[child_dict["squash_fn"]]
        child_dict["cov_type"] = CovType[child_dict["cov_type"]]
        child_dict["sample_params"]["d_action"] = rollout_fn.d_action
        child_dict["sample_params"]["horizon"] = rollout_fn.action_horizon
        child_dict["sample_params"]["tensor_args"] = tensor_args
        child_dict["sample_params"] = SampleConfig(**child_dict["sample_params"])

        # init_mean:
        if "init_mean" not in child_dict:
            child_dict["init_mean"] = rollout_fn.get_init_action_seq()
        return child_dict


class ParallelDIAL(ParticleOptBase, ParallelDIALConfig):
    @profiler.record_function("parallel_mppi/init")
    def __init__(self, config: Optional[ParallelDIALConfig] = None):
        if config is not None:
            ParallelDIALConfig.__init__(self, **vars(config))
        ParticleOptBase.__init__(self)

        self.sample_lib = SampleLib(self.sample_params)
        self._sample_set = None
        self._sample_iter = None
        # initialize covariance types:

        self.Z_seq = torch.zeros(
            1,
            self.action_horizon,
            self.d_action,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        self.delta = None
        self.mean_action = None
        self.act_seq = None
        self.cov_action = None
        self.best_traj = None
        self.scale_tril = None
        self.visual_traj = None
        if self.debug_info is not None and "visual_traj" in self.debug_info.keys():
            self.visual_traj = self.debug_info["visual_traj"]
        self.top_values = None
        self.top_idx = None
        self.top_trajs = None

        self.mean_lib = HaltonSampleLib(
            SampleConfig(
                self.action_horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **{"fixed_samples": False, "seed": 2567, "filter_coeffs": None}
            )
        )

        self.reset_distribution()
        self.update_samples()
        self._use_cuda_graph = False
        self._init_cuda_graph = False
        self.info = dict(rollout_time=0.0, entropy=[])
        self._batch_size = -1
        self._store_debug = False

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0

        B = torch.log(torch.tensor(sigma1 / sigma0)).to(self.tensor_args.device) / self.n_iters
        self.sigmas = A * torch.exp(B * torch.arange(self.n_iters).to(self.tensor_args.device))
        self.sigma_control = (
            self.horizon_diffuse_factor ** torch.flip(torch.arange(self.horizon+1), dims=[0])
        )
        self.traj_diffuse_factors = (
            self.sigma_control * self.traj_diffuse_factor ** (torch.arange(self.n_iters))[:, None]
        ).to(self.tensor_args.device)

    def get_rollouts(self):
        return self.top_trajs

    def reset_distribution(self):
        """
        Reset control distribution
        """

        self.reset_mean()

    def _compute_total_cost(self, costs, value_costs=None):
        """
        Calculate weights using exponential utility
        """

        # cost_seq = self.gamma_seq * costs
        # cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / self.gamma_seq[..., 0]
        # print(self.gamma_seq.shape, costs.shape)
        cost_seq = jit_compute_total_cost(self.gamma_seq, costs)
        if value_costs is not None:
            #TODO: define a variable for lambda_ (currently, set 20 as the default)
            value_cost_seq = jit_compute_value_cost(self.gamma_seq, value_costs, self.value_lambda, self.value_coef)
            value_cost_seq = value_cost_seq.unsqueeze(0)
            cost_seq = cost_seq + value_cost_seq
        return cost_seq

    def _exp_util(self, total_costs):
        w = jit_calculate_exp_util(self.beta, total_costs)
        # w = torch.softmax((-1.0 / self.beta) * total_costs, dim=-1)
        return w

    def _exp_util_from_costs(self, costs, value_costs=None):
        if value_costs is not None:
            w = jit_calculate_exp_util_from_value_costs(costs, value_costs, self.gamma_seq, self.beta, self.value_lambda, self.value_coef)
        else:
            w = jit_calculate_exp_util_from_costs(costs, self.gamma_seq, self.beta)
        return w

    def _compute_mean(self, w, actions):
        # get the new means from here
        new_mean = torch.sum(w * actions, dim=-3)
        new_mean = jit_blend_mean(self.mean_action, new_mean, self.step_size_mean)
        return new_mean


    @torch.no_grad()
    def _update_distribution(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions
        value_costs = trajectories.value_costs

        # Let's reshape to n_problems now:

        # first find the means before doing exponential utility:
        with profiler.record_function("mppi/get_best"):

            # Update best action
            if self.sample_mode == SampleMode.BEST:
                w = self._exp_util_from_costs(costs, value_costs)
                best_idx = torch.argmax(w, dim=-1)
                self.best_traj.copy_(actions[self.problem_col, best_idx])
        with profiler.record_function("mppi/store_rollouts"):

            if self.store_rollouts and self.visual_traj is not None:
                total_costs = self._compute_total_cost(costs, value_costs)
                vis_seq = getattr(trajectories.state, self.visual_traj)
                top_values, top_idx = torch.topk(total_costs, 20, dim=1)
                self.top_values = top_values
                self.top_idx = top_idx
                top_trajs = torch.index_select(vis_seq, 0, top_idx[0])
                for i in range(1, top_idx.shape[0]):
                    trajs = torch.index_select(
                        vis_seq, 0, top_idx[i] + (self.particles_per_problem * i)
                    )
                    top_trajs = torch.cat((top_trajs, trajs), dim=0)
                if self.top_trajs is None or top_trajs.shape != self.top_trajs:
                    self.top_trajs = top_trajs
                else:
                    self.top_trajs.copy_(top_trajs)

        w = self._exp_util_from_costs(costs, value_costs)
        w = w.unsqueeze(-1).unsqueeze(-1)
        new_mean = self._compute_mean(w, actions)

        self.mean_action.copy_(new_mean)

    @torch.no_grad()
    def sample_actions(self, init_act):
        delta = torch.index_select(self._sample_set, 0, self._sample_iter).squeeze(0)
        # self._sample_iter[:] += 1
        self._sample_iter_n += 1

        if not self.sample_params.fixed_samples:
            self._sample_iter[:] += 1

            if self._sample_iter_n >= self.n_iters:
                self._sample_iter[:] = 0
                log_info(
                    "Resetting sample iterations in particle opt base to 0, this is okay during graph capture"
                )

        if self._sample_iter_n >= self.n_iters:
            self._sample_iter_n = 0

        scaled_delta = delta * self.traj_diffuse_factors[self._sample_iter_n-1][:self.horizon].view(1, 1, -1, 1)
        # scaled_delta = delta 
        #TODO multiply diffusion factor


        act_seq = self.mean_action.unsqueeze(-3) + scaled_delta
        cat_list = [act_seq]

        if self.neg_per_problem > 0:
            neg_action = -1.0 * self.mean_action
            neg_act_seqs = neg_action.unsqueeze(-3).expand(-1, self.neg_per_problem, -1, -1)
            cat_list.append(neg_act_seqs)
        if self.null_per_problem > 0:
            cat_list.append(
                self.null_act_seqs[: self.null_per_problem]
                .unsqueeze(0)
                .expand(self.n_problems, -1, -1, -1)
            )

        act_seq = torch.cat(
            (cat_list),
            dim=-3,
        )
        act_seq = act_seq.reshape(self.total_num_particles, self.action_horizon, self.d_action)
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        # import pdb; pdb.set_trace()
        # if not copy_tensor(act_seq, self.act_seq):
        #    self.act_seq = act_seq
        return act_seq  # self.act_seq

    def update_seed(self, init_act):
        self.update_init_mean(init_act)

    def update_init_mean(self, init_mean):
        # update mean:
        # init_mean = init_mean.clone()
        if init_mean.shape[0] != self.n_problems:
            init_mean = init_mean.expand(self.n_problems, -1, -1)
        if not copy_tensor(init_mean, self.mean_action):
            self.mean_action = init_mean.clone()
        if not copy_tensor(init_mean, self.best_traj):
            self.best_traj = init_mean.clone()

    def reset_mean(self):
        with profiler.record_function("mppi/reset_mean"):
            if self.random_mean:
                mean = self.mean_lib.get_samples([self.n_problems])
                self.update_init_mean(mean)
            else:
                self.update_init_mean(self.init_mean)

    def _get_action_seq(self, mode: SampleMode):
        if mode == SampleMode.MEAN:
            act_seq = self.mean_action  # .clone()  # [self.mean_idx]#.clone()
        elif mode == SampleMode.SAMPLE:
            delta = self.generate_noise(
                shape=torch.Size((1, self.action_horizon)),
                base_seed=self.seed + 123 * self.num_steps,
            )
            act_seq = self.mean_action + torch.matmul(delta, self.full_scale_tril)
        elif mode == SampleMode.BEST:
            act_seq = self.best_traj  # [self.mean_idx]
        else:
            raise ValueError("Unidentified sampling mode in get_next_action")

        # act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        return act_seq

    def generate_noise(self, shape, base_seed=None):
        """
        Generate correlated noisy samples using autoregressive process
        """
        delta = self.sample_lib.get_samples(sample_shape=shape, seed=base_seed)
        return delta

    def _calc_val(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions
        delta = actions - self.mean_action.unsqueeze(0)

        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.beta * control_costs

        val = -self.beta * torch.logsumexp((-1.0 / self.beta) * total_costs)
        return val

    def reset(self):
        self.reset_distribution()

        self._sample_iter[:] = 0
        self._sample_iter_n = 0
        self.update_samples()  # this helps in restarting optimization
        super().reset()

    @property
    def squashed_mean(self):
        return scale_ctrl(
            self.mean_action, self.action_lows, self.action_highs, squash_fn=self.squash_fn
        )


    def reset_seed(self):
        self.sample_lib = SampleLib(self.sample_params)
        self.mean_lib = HaltonSampleLib(
            SampleConfig(
                self.action_horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **{"fixed_samples": False, "seed": 2567, "filter_coeffs": None}
            )
        )
        # resample if not fixed samples:
        self.update_samples()
        super().reset_seed()

    def update_samples(self):
        with profiler.record_function("mppi/update_samples"):
            if self.sample_params.fixed_samples:
                n_iters = 1
            else:
                n_iters = self.n_iters
            if self.sample_per_problem:
                s_set = (
                    self.sample_lib.get_samples(
                        sample_shape=[
                            self.sampled_particles_per_problem * self.n_problems * n_iters
                        ],
                        base_seed=self.seed,
                    )
                    .view(
                        n_iters,
                        self.n_problems,
                        self.sampled_particles_per_problem,
                        self.action_horizon,
                        self.d_action,
                    )
                    .clone()
                )
            else:
                s_set = self.sample_lib.get_samples(
                    sample_shape=[n_iters * (self.sampled_particles_per_problem)],
                    base_seed=self.seed,
                )
                s_set = s_set.view(
                    n_iters,
                    1,
                    self.sampled_particles_per_problem,
                    self.action_horizon,
                    self.d_action,
                )
                s_set = s_set.repeat(1, self.n_problems, 1, 1, 1).clone()
            s_set[:, :, -1, :, :] = 0.0
            if not copy_tensor(s_set, self._sample_set):
                log_info("ParallelMPPI: Updating sample set")
                self._sample_set = s_set
            if self._sample_iter is None:
                log_info("ParallelMPPI: Resetting sample iterations")  # , sample_iter.shape)

                self._sample_iter = torch.zeros(
                    (1), dtype=torch.long, device=self.tensor_args.device
                )
            else:
                self._sample_iter[:] = 0

            # if not copy_tensor(sample_iter, self._sample_iter):
            #    log_info("ParallelMPPI: Resetting sample iterations")  # , sample_iter.shape)
            #    self._sample_iter = sample_iter
            self._sample_iter_n = 0

    @torch.no_grad()
    def generate_rollouts(self, init_act=None):
        """
        Samples a batch of actions, rolls out trajectories for each particle
        and returns the resulting observations, costs,
        actions

        Parameters
        ----------
        state : dict or np.ndarray
            Initial state to set the simulation problem to
        """

        return super().generate_rollouts(init_act)

@get_torch_jit_decorator()
def jit_calculate_exp_util_from_value_costs(costs, value_costs, gamma_seq, beta: float, lambda_: float, value_coef: float):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    value_cost_seq = gamma_seq * value_costs
    value_cost_seq = value_coef * torch.logsumexp((1/lambda_) * torch.sum(value_cost_seq, dim=-1, keepdim=False), dim=0)
    # value_cost_seq = value_coef * torch.sum(value_cost_seq.squeeze(0), dim=-1, keepdim=False)
    cost_seq = cost_seq + value_cost_seq
    w = torch.softmax((-1.0 / beta) * cost_seq, dim=-1)
    return w

@get_torch_jit_decorator()
def jit_calculate_exp_util(beta: float, total_costs):
    w = torch.softmax((-1.0 / beta) * total_costs, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_calculate_exp_util_from_costs(costs, gamma_seq, beta: float):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    w = torch.softmax((-1.0 / beta) * cost_seq, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_compute_total_cost(gamma_seq, costs):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    return cost_seq

@get_torch_jit_decorator()
def jit_compute_value_cost(gamma_seq, costs, lambda_: float, value_coef: float):
    cost_seq = gamma_seq * costs
    cost_seq = value_coef * torch.logsumexp((1/lambda_) * torch.sum(cost_seq, dim=-1, keepdim=False), dim=0)
    # cost_seq = value_coef * torch.sum(cost_seq.squeeze(0), dim=-1, keepdim=False) / gamma_seq[..., 0]
    # cost_seq = torch.clamp(cost_seq, min=0.)
    return cost_seq



@get_torch_jit_decorator()
def jit_blend_mean(mean_action, new_mean, step_size_mean: float):
    mean_update = (1.0 - step_size_mean) * mean_action + step_size_mean * new_mean
    return mean_update

