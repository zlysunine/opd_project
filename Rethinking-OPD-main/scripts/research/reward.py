"""Verl rule reward adapter, shared with standalone evaluation."""
from scripts.research.data import verify


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return float(verify(solution_str, ground_truth, data_source, extra_info=extra_info))


def compute_score_details(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Same verifier as evaluation, with numeric extraction diagnostics for GRPO."""
    from analysis.evaluate import answer_status
    result=answer_status(solution_str,ground_truth,data_source,row=extra_info,
                         truncated=bool((extra_info or {}).get('_research_truncated',False)))
    return dict(score=float(result['correct']),extraction_failed=float(result['extraction_failed']),
                format_compliant=float(result.get('format_compliant',False)))
