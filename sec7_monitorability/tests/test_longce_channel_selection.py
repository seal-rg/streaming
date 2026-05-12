"""Tests for the LongCE channel-selection helper.

The cycle-mode invariants the "LongCE as regularizer" interpretation
relies on:

  1. Every channel is visited exactly once per cycle (length ceil(C / num_ch)
     calls).
  2. Reshuffle happens when the pool exhausts — no channel gets stuck
     at the back of every cycle.
  3. Skip steps (not calling the helper) do NOT consume cycle slots —
     cycle integrity is preserved across warmup / probabilistic skipping.
  4. Calls where num_ch doesn't evenly divide C still fulfill the
     requested count AND leave no channel unvisited over multiple cycles.
  5. num_ch == C degenerates to "always return every channel", which is
     the maximum-coverage degenerate case.
  6. Random mode is unchanged from prior behavior (sampled without
     replacement, sorted).

These are pure-logic tests — they do not exercise the full trainer /
forward loop. They verify that the selection primitive, which is the
whole basis of the regularizer's guarantees, behaves as specified.
"""

import os
import random
import sys
import unittest

# Match the sys.path convention used by other tests in this package so
# `train.custom_trainer_stream` resolves without needing a `train.__init__`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from train.custom_trainer_stream import StreamTrainer


class _FakeTrainer:
    """Minimal stand-in exercising just the channel-selection helper.

    Skips the real __init__ (which needs a full trl SFTTrainer config)
    and sets only the fields the selection helper reads.
    """

    def __init__(self, num_channels=10, mode="cycle"):
        self.num_channels = num_channels
        self.longce_selection_mode = mode
        self._longce_pool = []

    # Attach the real method — it only uses self.num_channels,
    # self.longce_selection_mode, and self._longce_pool.
    _select_longce_channels = StreamTrainer._select_longce_channels


class TestCycleMode(unittest.TestCase):
    def setUp(self):
        random.seed(0xDEADBEEF)
        self.t = _FakeTrainer(num_channels=10, mode="cycle")

    def test_covers_every_channel_in_one_cycle(self):
        """A full cycle's worth of calls visits every channel exactly once."""
        seen = []
        calls_per_cycle = 10 // 2  # num_channels=10, num_ch=2 → 5 calls
        for _ in range(calls_per_cycle):
            seen.extend(self.t._select_longce_channels(2))
        self.assertEqual(sorted(seen), list(range(10)))

    def test_reshuffles_after_exhaustion(self):
        """Two consecutive cycles each visit every channel exactly once."""
        first_cycle, second_cycle = [], []
        for _ in range(5):
            first_cycle.extend(self.t._select_longce_channels(2))
        for _ in range(5):
            second_cycle.extend(self.t._select_longce_channels(2))
        self.assertEqual(sorted(first_cycle), list(range(10)))
        self.assertEqual(sorted(second_cycle), list(range(10)))

    def test_cycle_order_is_shuffled(self):
        """Successive cycles produce different orderings (reshuffle works).

        With RNG seeded and C=10, the chance of two independent shuffles
        producing the same order is 1 / 10! — vanishingly small.
        """
        first = []
        for _ in range(5):
            first.extend(self.t._select_longce_channels(2))
        second = []
        for _ in range(5):
            second.extend(self.t._select_longce_channels(2))
        self.assertNotEqual(first, second)

    def test_num_ch_not_dividing_C(self):
        """C=10, num_ch=3 — partial tail is consumed plus fresh shuffle."""
        t = _FakeTrainer(num_channels=10, mode="cycle")
        picks = [t._select_longce_channels(3) for _ in range(10)]
        # Flatten; over 10 calls of 3-each = 30 selections, covers 3
        # complete cycles (3 * 10 = 30). Every channel should appear
        # exactly 3 times.
        from collections import Counter

        counts = Counter()
        for p in picks:
            counts.update(p)
        self.assertEqual(dict(counts), {i: 3 for i in range(10)})

    def test_num_ch_equals_C(self):
        """num_ch == C — every call returns every channel."""
        t = _FakeTrainer(num_channels=10, mode="cycle")
        for _ in range(5):
            picks = t._select_longce_channels(10)
            self.assertEqual(picks, list(range(10)))

    def test_num_ch_exceeds_C_is_clamped(self):
        t = _FakeTrainer(num_channels=10, mode="cycle")
        picks = t._select_longce_channels(100)
        self.assertEqual(len(picks), 10)
        self.assertEqual(picks, list(range(10)))

    def test_num_ch_zero_returns_empty(self):
        t = _FakeTrainer(num_channels=10, mode="cycle")
        self.assertEqual(t._select_longce_channels(0), [])
        # Pool stays untouched.
        self.assertEqual(t._longce_pool, [])

    def test_skip_does_not_consume_cycle(self):
        """The invariant: cycles don't advance except when called.

        This is by construction — the trainer only calls
        _select_longce_channels from inside the do_longce=True branch.
        Here we verify by NOT calling between some "skip" simulations:
        state is preserved.
        """
        t = _FakeTrainer(num_channels=10, mode="cycle")
        # Simulate 3 "active" calls.
        a = t._select_longce_channels(2)
        b = t._select_longce_channels(2)
        c = t._select_longce_channels(2)
        # Between c and d: many "skip" events (nothing happens here).
        d = t._select_longce_channels(2)
        e = t._select_longce_channels(2)
        # After 5 calls × 2 each = one full cycle. Every channel visited.
        self.assertEqual(sorted(a + b + c + d + e), list(range(10)))

    def test_output_is_sorted(self):
        """Selection returns sorted lists — downstream code relies on this
        (selected_set / ce_self dict are built in channel-id order)."""
        for _ in range(20):
            picks = self.t._select_longce_channels(3)
            self.assertEqual(picks, sorted(picks))


class TestRandomMode(unittest.TestCase):
    def setUp(self):
        random.seed(0xCAFEBABE)

    def test_random_matches_random_sample(self):
        """Random mode should produce the same sequence as the original
        random.sample(...)-based code under the same RNG state."""
        t = _FakeTrainer(num_channels=10, mode="random")
        random.seed(1234)
        r_actual = t._select_longce_channels(3)
        random.seed(1234)
        r_reference = sorted(random.sample(range(10), 3))
        self.assertEqual(r_actual, r_reference)

    def test_random_does_not_touch_pool(self):
        t = _FakeTrainer(num_channels=10, mode="random")
        t._select_longce_channels(3)
        self.assertEqual(t._longce_pool, [])

    def test_random_returns_sorted_unique(self):
        t = _FakeTrainer(num_channels=10, mode="random")
        for _ in range(20):
            picks = t._select_longce_channels(4)
            self.assertEqual(len(set(picks)), len(picks))
            self.assertEqual(picks, sorted(picks))


class TestPerRankIndependence(unittest.TestCase):
    """Per-rank independence: each rank has its own pool. If two ranks
    shuffle with different RNG states (the natural case under DDP),
    their cycles differ. This test simulates two ranks with different
    RNG seeds and confirms they cover the same channel set but with
    independent orderings.
    """

    def test_per_rank_pools_independent(self):
        # Rank 0
        random.seed(11)
        r0 = _FakeTrainer(num_channels=10, mode="cycle")
        r0_cycle = []
        for _ in range(5):
            r0_cycle.extend(r0._select_longce_channels(2))
        # Rank 1
        random.seed(22)
        r1 = _FakeTrainer(num_channels=10, mode="cycle")
        r1_cycle = []
        for _ in range(5):
            r1_cycle.extend(r1._select_longce_channels(2))
        # Same set (every channel visited once each)
        self.assertEqual(sorted(r0_cycle), list(range(10)))
        self.assertEqual(sorted(r1_cycle), list(range(10)))
        # Different orderings — cycles are truly independent.
        self.assertNotEqual(r0_cycle, r1_cycle)


if __name__ == "__main__":
    unittest.main()
