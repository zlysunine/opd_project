# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check paper formulas with hand-computed examples; no verl runtime needed."""

import importlib.util
import math
import unittest
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[2] / "verl/utils/opd_metrics.py"
SPEC = importlib.util.spec_from_file_location("opd_metrics", MODULE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
compute = module.compute_opd_dynamic_metrics


class DynamicMetricChecks(unittest.TestCase):
    def inputs(self):
        return {
            "student_log_probs": torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]]).log(),
            "teacher_log_probs": torch.tensor([[[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]]).log(),
            "overlap_mask": torch.tensor([[[1, 1, 0], [1, 0, 0], [1, 1, 1]]]),
            "response_mask": torch.tensor([[1, 1, 0]]),
            "student_entropy": torch.tensor([[1.0, 3.0, float("nan")]]),
            "teacher_entropy": torch.tensor([[3.0, 1.0, float("nan")]]),
        }

    def test_overlap_advantage_renormalizes_within_each_intersection(self):
        result = compute(**self.inputs())
        # Position 1: I={0,1}, p_I=(2/3,1/3), q_I=(2/7,5/7).
        # Position 2: singleton I -> KL=0 despite different raw probabilities.
        kl = (2 / 3) * math.log((2 / 3) / (2 / 7)) + (1 / 3) * math.log((1 / 3) / (5 / 7))
        self.assertAlmostEqual(result["opd/overlap_token_advantage"], (-kl / 2 + 0) / 2, places=6)
        self.assertAlmostEqual(result["opd/overlap_ratio"], (2 / 3 + 1 / 3) / 2)

    def test_entropy_gap_is_pointwise_absolute_not_difference_of_means(self):
        result = compute(**self.inputs())
        self.assertEqual(result["opd/student_entropy"], 2)
        self.assertEqual(result["opd/teacher_entropy"], 2)
        self.assertEqual(result["opd/entropy_gap"], 2)

    def test_padding_and_nonshared_nonfinite_values_are_excluded(self):
        data = self.inputs()
        data["student_log_probs"][0, 2] = float("nan")
        data["teacher_log_probs"][0, 2] = float("nan")
        data["teacher_log_probs"][0, 0, 2] = -torch.inf
        result = compute(**data)
        self.assertEqual(result["opd/valid_response_tokens"], 2)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_advantage_is_invariant_to_raw_intersection_probability_mass(self):
        data = self.inputs()
        reference = compute(**data)["opd/overlap_token_advantage"]
        data["student_log_probs"] -= 10
        data["teacher_log_probs"] -= 20
        actual = compute(**data)["opd/overlap_token_advantage"]
        self.assertAlmostEqual(reference, actual, places=6)

    def test_empty_intersections_are_reported_without_fake_zero_advantage(self):
        data = self.inputs()
        data["overlap_mask"].zero_()
        result = compute(**data)
        self.assertEqual(result["opd/overlap_ratio"], 0)
        self.assertEqual(result["opd/overlap_nonempty_fraction"], 0)
        self.assertNotIn("opd/overlap_token_advantage", result)

    def test_empty_responses_have_no_defined_means(self):
        data = self.inputs()
        data["response_mask"].zero_()
        self.assertEqual(compute(**data), {"opd/valid_response_tokens": 0.0})

    def test_response_positions_have_equal_weight_across_different_lengths(self):
        data = self.inputs()
        for key, value in data.items():
            data[key] = value.repeat(2, *([1] * (value.ndim - 1)))
        data["response_mask"][1] = torch.tensor([1, 0, 0])
        data["student_entropy"][1, 0] = 5
        data["teacher_entropy"][1, 0] = 5
        result = compute(**data)
        self.assertEqual(result["opd/student_entropy"], 3)
        self.assertAlmostEqual(result["opd/entropy_gap"], 4 / 3, places=6)

    def test_same_intersection_distributions_have_zero_advantage(self):
        data = self.inputs()
        data["teacher_log_probs"] = data["student_log_probs"].clone()
        self.assertAlmostEqual(compute(**data)["opd/overlap_token_advantage"], 0)

    def test_full_vocabulary_entropy_is_not_replaced_by_top_k_entropy(self):
        p = torch.tensor([0.4, 0.3, 0.2, 0.1])
        q = torch.tensor([0.3, 0.4, 0.1, 0.2])
        hp = -(p * p.log()).sum()
        hq = -(q * q.log()).sum()
        result = compute(p[:2].log().view(1, 1, 2), q[:2].log().view(1, 1, 2),
                         torch.ones(1, 1, 2), torch.ones(1, 1), hp.view(1, 1), hq.view(1, 1))
        normalized_top_k = p[:2] / p[:2].sum()
        top_k_entropy = -(normalized_top_k * normalized_top_k.log()).sum().item()
        self.assertAlmostEqual(result["opd/student_entropy"], hp.item())
        self.assertGreater(result["opd/student_entropy"] - top_k_entropy, 0.5)

    def test_measurement_does_not_attach_gradients_or_change_inputs(self):
        data = self.inputs()
        data["student_log_probs"].requires_grad_()
        original = data["student_log_probs"].detach().clone()
        result = compute(**data)
        self.assertTrue(all(isinstance(value, float) for value in result.values()))
        self.assertIsNone(data["student_log_probs"].grad)
        torch.testing.assert_close(data["student_log_probs"].detach(), original)

    def test_invalid_shapes_or_valid_nan_fail_instead_of_logging_bogus_values(self):
        data = self.inputs()
        data["teacher_entropy"] = torch.ones(1, 1)
        with self.assertRaises(ValueError):
            compute(**data)
        data = self.inputs()
        data["teacher_log_probs"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            compute(**data)


if __name__ == "__main__":
    unittest.main()
