"""Verl rule reward adapter, shared with standalone evaluation."""
from scripts.research.data import verify


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return float(verify(solution_str, ground_truth, data_source))
