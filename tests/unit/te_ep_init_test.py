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

"""Unit tests for TE EP MaxText bootstrap helpers."""

from maxtext.layers import te_ep_init


def test_calculate_te_ep_capacity_aligned():
  capacity = te_ep_init.calculate_te_ep_capacity(
      max_tokens_per_rank=16,
      ep_size=4,
      num_experts=8,
      top_k=2,
      num_local_experts=2,
      recv_capacity_factor=1.0,
      dispatch_alignment=128,
  )

  assert capacity == 256
  assert capacity % (2 * 128) == 0


def test_calculate_te_ep_capacity_factor_before_alignment():
  capacity = te_ep_init.calculate_te_ep_capacity(
      max_tokens_per_rank=16,
      ep_size=4,
      num_experts=8,
      top_k=2,
      num_local_experts=2,
      recv_capacity_factor=3.0,
      dispatch_alignment=128,
  )

  assert capacity == 512
  assert capacity % (2 * 128) == 0


def test_calculate_te_ep_capacity_unaligned():
  capacity = te_ep_init.calculate_te_ep_capacity(
      max_tokens_per_rank=4,
      ep_size=4,
      num_experts=8,
      top_k=2,
      num_local_experts=2,
      recv_capacity_factor=1.5,
      dispatch_alignment=0,
  )

  assert capacity == 48
