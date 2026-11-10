"""Use the vendored LLaMA-Factory trainer with actual label-token accounting."""
import json
from pathlib import Path
import sys
import time

import torch
from transformers import TrainerCallback

from .common import load_verl_utility


class Accounting(TrainerCallback):
    def __init__(self, out):
        self.out = Path(out)
        self.budget = load_verl_utility('research_budget').ResearchBudget(self.out/'metrics.jsonl')
        self.questions = self.tokens = 0
        self.limits = json.loads((self.out/'resolved_config.json').read_text())
        self.handle = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        def before(module, inputs, named):
            if module.training and torch.is_grad_enabled():
                labels = named.get('labels')
                if labels is not None:
                    # Causal shift drops the first label.
                    self.tokens += int((labels[...,1:] != -100).sum())
                    self.questions += len(labels)
        self.budget = load_verl_utility('research_budget').ResearchBudget(self.out/'metrics.jsonl', state.global_step)
        self.handle = model.register_forward_pre_hook(before, with_kwargs=True)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    def on_step_end(self,args,state,control,**kwargs):
        counts = torch.tensor([self.questions,self.tokens],device=args.device,dtype=torch.long)
        if torch.distributed.is_initialized(): torch.distributed.all_reduce(counts)
        metrics = self.budget.update(int(counts[0]),0,int(counts[1]),0)
        self.questions = self.tokens = 0
        if torch.cuda.is_available(): metrics['memory/peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if state.is_world_process_zero: self.budget.log(state.global_step,metrics)
        stop = self.budget.exceeded({k:self.limits.get('budget_'+k) for k in ('optimized_tokens','wall_seconds')})
        stop_flag = torch.tensor(int(stop),device=args.device)
        if torch.distributed.is_initialized(): torch.distributed.all_reduce(stop_flag,op=torch.distributed.ReduceOp.MAX)
        if stop_flag.item():
            control.should_training_stop = control.should_save = True
        return control

    def on_train_end(self,args,state,control,**kwargs):
        if self.handle: self.handle.remove()


if __name__ == '__main__':
    from llamafactory.train.tuner import run_exp
    run_exp(json.loads(Path(sys.argv[1]).read_text()), callbacks=[Accounting(sys.argv[2])])
