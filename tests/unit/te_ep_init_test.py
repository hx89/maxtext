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

"""Unit tests for the TE EP MaxText bootstrap helpers (pure-Python parts).

The TE NCCL EP and global_shard_guard imports inside
:func:`init_te_ep_for_maxtext` are lazy, so this test file does not require
``transformer_engine`` to be installed.
"""

import unittest

from maxtext.layers import te_ep_init


class CalculateTeEpCapacityTest(unittest.TestCase):
  """Worst-case capacity math; no GPU / TE needed."""

  def test_aligned(self):
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=16,
        ep_size=4,
        num_experts=8,
        top_k=2,
        num_local_experts=2,
        recv_capacity_factor=1.0,
        dispatch_alignment=128,
    )
    # T_per_ep_group=64, worst=128, target=128. align_units=ceil(128/2/128)=1.
    # +1 headroom -> 2. POW2(2)=2. slots=2*128=256. recv=2*256=512.
    self.assertEqual(capacity, 512)
    # slots_per_expert must be a power of two (TE EP assertion in ncclEpInitHandle).
    slots = capacity // 2
    self.assertEqual(slots & (slots - 1), 0, f"slots_per_expert={slots} not a power of two")

  def test_factor_before_alignment(self):
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=16,
        ep_size=4,
        num_experts=8,
        top_k=2,
        num_local_experts=2,
        recv_capacity_factor=3.0,
        dispatch_alignment=128,
    )
    # worst=128, target=384. align_units=ceil(384/2/128)=2. +1=3. POW2(3)=4.
    # slots=4*128=512. recv=2*512=1024.
    self.assertEqual(capacity, 1024)
    slots = capacity // 2
    self.assertEqual(slots & (slots - 1), 0)

  def test_pow2_round_up_from_non_pow2(self):
    """An align_units count that's not a POW2 must be rounded up to next POW2.

    DSV3 671B with rcf=1.5 reproduced this: align_units=3 → must round to 4.
    """
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=8192,
        ep_size=4,
        num_experts=256,
        top_k=8,
        num_local_experts=64,
        recv_capacity_factor=1.5,
        dispatch_alignment=128,
    )
    # worst=8192*4*8=262144, target=ceil(262144*1.5)=393216.
    # align_units = ceil(393216/64/128) = ceil(48) = 48. +1 = 49. POW2(49)=64.
    # slots=64*128=8192. recv=64*8192=524288.
    self.assertEqual(capacity, 524288)
    slots = capacity // 64
    self.assertEqual(slots & (slots - 1), 0)

  def test_unaligned(self):
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=4,
        ep_size=4,
        num_experts=8,
        top_k=2,
        num_local_experts=2,
        recv_capacity_factor=1.5,
        dispatch_alignment=0,
    )
    # T_per_ep_group=16, worst=max(16*2, 16)*1=32, target=ceil(32*1.5)=48, no alignment.
    self.assertEqual(capacity, 48)

  def test_overconc_active_when_experts_exceed_pool(self):
    # T_per_ep_group * top_k = 1*1 = 1; num_experts=8 → overconc=8.
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=1,
        ep_size=1,
        num_experts=8,
        top_k=1,
        num_local_experts=8,
        recv_capacity_factor=1.0,
        dispatch_alignment=0,
    )
    # active=min(8, 1)=1, overconc=ceil(8/1)=8, worst=max(1, 16)*8=128.
    self.assertEqual(capacity, 128)


class ModuleSurfaceTest(unittest.TestCase):
  """Surface-level checks that don't require TE/JAX to be importable as GPU."""

  def test_te_ep_axis_constant(self):
    self.assertEqual(te_ep_init._TE_EP_AXIS, "expert")

  def test_get_state_before_init_raises(self):
    te_ep_init.reset_te_ep_state_for_test()
    with self.assertRaisesRegex(ValueError, "not been initialized"):
      te_ep_init.get_te_ep_state()


if __name__ == "__main__":
  unittest.main()
