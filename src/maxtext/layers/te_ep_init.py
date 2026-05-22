# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TransformerEngine NCCL EP bootstrap helpers for MaxText MoE."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
from jax.sharding import PartitionSpec

from maxtext.utils import max_logging


_TE_EP_AXIS = "expert"
_TE_EP_STATE: TeEpState | None = None


@dataclass(frozen=True)
class TeEpState:
  """Process-local TE EP bootstrap state."""

  mesh_resource: Any
  ep_axis: str
  outer_axis: str | None
  outer_size: int
  ep_size: int
  expected_world_size: int
  num_experts: int
  num_local_experts: int
  max_tokens_per_rank: int
  recv_capacity_per_rank: int
  dispatch_alignment: int
  hidden_dim: int
  max_num_sms: int
  em_unfused_num_sms: int
  input_spec_2d: PartitionSpec
  input_spec_3d: PartitionSpec
  ep_spec_2d: PartitionSpec
  ep_spec_3d: PartitionSpec
  config_key: tuple[Any, ...]


def _mesh_axis_size(mesh: jax.sharding.Mesh, axis: str) -> int:
  if axis not in mesh.shape:
    raise ValueError(f"TE EP requires mesh axis '{axis}' to be present. Mesh axes: {tuple(mesh.shape.keys())}.")
  return int(mesh.shape[axis])


def _active_mesh_axes(mesh: jax.sharding.Mesh) -> dict[str, int]:
  return {axis: int(size) for axis, size in mesh.shape.items() if int(size) > 1}


def select_te_ep_outer_axis(mesh: jax.sharding.Mesh) -> str | None:
  """Select the single non-EP outer mesh axis for TE EP v1."""
  if "fsdp" in mesh.shape:
    return "fsdp"
  if "data" in mesh.shape:
    return "data"
  return None


def _validate_v1_mesh(mesh: jax.sharding.Mesh, outer_axis: str | None) -> None:
  active_axes = _active_mesh_axes(mesh)
  allowed_axes = {_TE_EP_AXIS}
  if outer_axis is not None:
    allowed_axes.add(outer_axis)
  unsupported_axes = {axis: size for axis, size in active_axes.items() if axis not in allowed_axes}
  if unsupported_axes:
    raise ValueError(
        "use_te_ep=True v1 supports only the expert axis and one outer data/FSDP axis. "
        f"Unsupported active mesh axes: {unsupported_axes}."
    )


def _mesh_axis_if_present(mesh: jax.sharding.Mesh, axis: str) -> str | None:
  return axis if axis in mesh.shape else None


def _make_mesh_resource(mesh: jax.sharding.Mesh, outer_axis: str | None, ep_axis: str) -> Any:
  from transformer_engine.jax.sharding import MeshResource  # pylint: disable=import-outside-toplevel

  kwargs = {
      "dp_resource": "data" if outer_axis == "data" else None,
      "tp_resource": _mesh_axis_if_present(mesh, "tensor"),
      "fsdp_resource": _mesh_axis_if_present(mesh, "fsdp"),
      "pp_resource": None,
      "cp_resource": _mesh_axis_if_present(mesh, "context"),
      "ep_resource": ep_axis,
  }
  return MeshResource(**kwargs)


def calculate_te_ep_capacity(
    *,
    max_tokens_per_rank: int,
    ep_size: int,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    recv_capacity_factor: float,
    dispatch_alignment: int,
) -> int:
  """Conservatively size TE EP's static receive buffer."""
  tokens_per_ep_group = max_tokens_per_rank * ep_size
  active_experts = min(num_experts, tokens_per_ep_group * top_k)
  overconcentration = max(1, math.ceil(num_experts / max(1, active_experts)))
  worst_case = max(tokens_per_ep_group * top_k, 16) * overconcentration
  target = max(1, math.ceil(worst_case * recv_capacity_factor))

  if dispatch_alignment > 0:
    slots_per_expert = math.ceil(target / num_local_experts / dispatch_alignment)
    slots_per_expert = max(1, slots_per_expert) * dispatch_alignment
    return num_local_experts * slots_per_expert
  return target


def _max_tokens_per_rank(config: Any, leading_axis_size: int) -> int:
  global_tokens = int(config.micro_batch_size_to_train_on * config.max_target_length)
  return max(1, math.ceil(global_tokens / max(1, leading_axis_size)))


def _hidden_dim(config: Any) -> int:
  return int(config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim)


def build_te_ep_state(config: Any, mesh: jax.sharding.Mesh) -> TeEpState:
  """Build the TE EP state without mutating the process singleton."""
  ep_size = _mesh_axis_size(mesh, _TE_EP_AXIS)
  outer_axis = select_te_ep_outer_axis(mesh)
  outer_size = _mesh_axis_size(mesh, outer_axis) if outer_axis is not None else 1
  _validate_v1_mesh(mesh, outer_axis)

  if int(config.num_experts) % ep_size != 0:
    raise ValueError(f"num_experts ({config.num_experts}) must be divisible by TE EP size ({ep_size}).")

  num_local_experts = int(config.num_experts) // ep_size
  min_dispatch_alignment = int(config.moe_permutation_group_align_size)
  expected_world_size = outer_size * ep_size
  max_tokens_per_rank = _max_tokens_per_rank(config, expected_world_size)
  recv_capacity_per_rank = calculate_te_ep_capacity(
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
      num_experts=int(config.num_experts),
      top_k=int(config.num_experts_per_tok),
      num_local_experts=num_local_experts,
      recv_capacity_factor=float(config.te_ep_recv_capacity_factor),
      dispatch_alignment=min_dispatch_alignment,
  )
  # TE's phuong/ep-unfused MoE example passes the full per-expert slot count as
  # dispatch alignment, so match that layout contract for the MaxText GMM path.
  dispatch_alignment = max(1, recv_capacity_per_rank // num_local_experts)

  leading_spec = (outer_axis, _TE_EP_AXIS) if outer_axis is not None else _TE_EP_AXIS
  config_key = (
      _TE_EP_AXIS,
      outer_axis,
      outer_size,
      ep_size,
      expected_world_size,
      int(config.num_experts),
      num_local_experts,
      max_tokens_per_rank,
      recv_capacity_per_rank,
      dispatch_alignment,
      _hidden_dim(config),
      int(config.te_ep_max_num_sms),
      int(config.te_ep_em_unfused_num_sms),
  )

  return TeEpState(
      mesh_resource=_make_mesh_resource(mesh, outer_axis, _TE_EP_AXIS),
      ep_axis=_TE_EP_AXIS,
      outer_axis=outer_axis,
      outer_size=outer_size,
      ep_size=ep_size,
      expected_world_size=expected_world_size,
      num_experts=int(config.num_experts),
      num_local_experts=num_local_experts,
      max_tokens_per_rank=max_tokens_per_rank,
      recv_capacity_per_rank=recv_capacity_per_rank,
      dispatch_alignment=dispatch_alignment,
      hidden_dim=_hidden_dim(config),
      max_num_sms=int(config.te_ep_max_num_sms),
      em_unfused_num_sms=int(config.te_ep_em_unfused_num_sms),
      input_spec_2d=PartitionSpec(leading_spec, None),
      input_spec_3d=PartitionSpec(leading_spec, None, None),
      ep_spec_2d=PartitionSpec(leading_spec, None),
      ep_spec_3d=PartitionSpec(leading_spec, None, None),
      config_key=config_key,
  )


def init_te_ep_for_maxtext(config: Any, mesh: jax.sharding.Mesh) -> TeEpState:
  """Initialize TE EP once per process before model/training-step tracing."""
  global _TE_EP_STATE

  candidate = build_te_ep_state(config, mesh)
  if _TE_EP_STATE is not None:
    if _TE_EP_STATE.config_key != candidate.config_key:
      raise ValueError(
          "TE EP was already initialized with a different shape/resource contract. "
          f"Existing={_TE_EP_STATE.config_key}, requested={candidate.config_key}."
      )
    return _TE_EP_STATE

  world_size = jax.process_count()
  rank = jax.process_index()
  if world_size != candidate.expected_world_size:
    raise ValueError(
        "TE EP v1 expects one JAX process per active fsdp/expert mesh slot. "
        f"process_count={world_size}, expected={candidate.expected_world_size}, "
        f"outer_axis={candidate.outer_axis}, ep_size={candidate.ep_size}."
    )

  from transformer_engine.jax.ep import ep_bootstrap  # pylint: disable=import-outside-toplevel
  from transformer_engine.jax.sharding import global_shard_guard  # pylint: disable=import-outside-toplevel

  with mesh, jax.set_mesh(mesh), global_shard_guard(candidate.mesh_resource):
    ep_bootstrap(
        world_size=world_size,
        rank=rank,
        ep_size=candidate.ep_size,
        num_experts=candidate.num_experts,
        max_tokens_per_rank=candidate.max_tokens_per_rank,
        recv_capacity_per_rank=candidate.recv_capacity_per_rank,
        hidden_dim=candidate.hidden_dim,
        max_num_sms=candidate.max_num_sms,
        em_unfused_num_sms=candidate.em_unfused_num_sms,
    )

  _TE_EP_STATE = candidate
  max_logging.log(
      "TE EP initialized: "
      f"outer_axis={candidate.outer_axis}, ep_axis={candidate.ep_axis}, ep_size={candidate.ep_size}, "
      f"max_tokens_per_rank={candidate.max_tokens_per_rank}, "
      f"recv_capacity_per_rank={candidate.recv_capacity_per_rank}, "
      f"dispatch_alignment={candidate.dispatch_alignment}, "
      f"em_unfused_num_sms={candidate.em_unfused_num_sms}"
  )
  return _TE_EP_STATE


def get_te_ep_state() -> TeEpState:
  if _TE_EP_STATE is None:
    raise ValueError("TE EP has not been initialized. Call init_te_ep_for_maxtext(config, mesh) before tracing MoE.")
  return _TE_EP_STATE


def reset_te_ep_state_for_test() -> None:
  global _TE_EP_STATE
  _TE_EP_STATE = None
