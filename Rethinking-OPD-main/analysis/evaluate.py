"""Live behavior is kept separate from teacher-forced geometry and token confidence."""
import re

import numpy as np

from scripts.research.common import ROOT, digest, file_hash, seed_all
from scripts.research.data import verify


def answer_status(text, ground_truth, benchmark, truncated=False):
    # Same extraction and equivalence implementation as the training rule reward.
    if benchmark == 'controlled_modular':
        extracted = re.findall(r'\\boxed\{\s*([-+]?\d+)\s*\}', text)
        extracted = extracted[-1] if extracted else ''
    else:
        import sys
        directory = str(ROOT/'scripts/val/eval')
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from utils import extract_answer
        extracted = extract_answer(text)
    failed = not bool(extracted)
    return dict(extracted_answer=extracted, extraction_failed=failed,
                correct=bool(not truncated and not failed and verify(text, ground_truth, benchmark)),
                verifier_revision=file_hash(ROOT/'scripts/val/eval/utils.py'))


def rollout(model, row, config, pool, step):
    seed = int(digest([config['seed'], row['problem_id'], pool])[:8], 16)
    seed_all(seed)
    fixed = pool == 'fixed_base_rollout'
    greedy = not fixed and config.get('live_greedy', True)
    ids, text, truncated = model.generate(row, max_tokens=config['max_probe_response_tokens'],
                                         temperature=1., top_p=1., greedy=greedy)
    status = answer_status(text, row['ground_truth'], row['benchmark'], truncated)
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return {**row, **status, 'pool':pool, 'source_optimizer_step':step, 'prompt_ids':model.prompt(row),
            'response_ids':ids, 'response_text':text, 'response_length':len(ids),
            'eos':not truncated, 'truncated':truncated, 'seed':seed,
            'decoding':dict(greedy=greedy, temperature=None if greedy else 1., top_p=1.),
            'repeated_4gram_rate':1-len(set(grams))/len(grams) if grams else None}


def behavior(rows):
    if not rows:
        return {}
    topics = sorted({r['topic_label'] for r in rows})
    return dict(accuracy_micro=float(np.mean([r['correct'] for r in rows])),
                accuracy_topic_macro=float(np.mean([np.mean([r['correct'] for r in rows if r['topic_label']==t])
                                                   for t in topics])),
                response_length=float(np.mean([r['response_length'] for r in rows])),
                eos_rate=float(np.mean([r['eos'] for r in rows])),
                truncation_rate=float(np.mean([r['truncated'] for r in rows])),
                extraction_failure_rate=float(np.mean([r['extraction_failed'] for r in rows])))
