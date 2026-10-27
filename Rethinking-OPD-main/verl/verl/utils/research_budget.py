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

"""Append-only research accounting; restore only through loaded checkpoint step."""
import json
import time
from pathlib import Path


class ResearchBudget:
    def __init__(self, path, loaded_step=0):
        self.path = Path(path)
        self.start = time.monotonic()
        self.totals = dict.fromkeys(('questions', 'rollout_tokens', 'optimized_tokens', 'teacher_forward_tokens',
                                    'updates', 'wall_seconds'), 0.0)
        if self.path.exists():
            rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
            rows = [r for r in rows if r['step'] <= loaded_step]
            if rows:
                self.totals.update(rows[-1]['budget'])
        self.previous_wall = self.totals['wall_seconds']

    def update(self, questions, rollout_tokens, optimized_tokens, teacher_tokens, updates=1):
        for key, value in dict(questions=questions, rollout_tokens=rollout_tokens, optimized_tokens=optimized_tokens,
                               teacher_forward_tokens=teacher_tokens, updates=updates).items():
            self.totals[key] += value
        self.totals['wall_seconds'] = self.previous_wall + time.monotonic()-self.start
        return {'budget/'+key: value for key, value in self.totals.items()}

    def exclude_pause(self, seconds):
        """Keep synchronous checkpoint analysis out of the training wall budget."""
        if seconds < 0:
            raise ValueError('Pause duration must be nonnegative')
        self.start += seconds

    def exceeded(self, limits):
        return any(limit is not None and self.totals[key] >= limit for key,limit in limits.items())

    def log(self, step, metrics):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Configured metrics are scalar numbers. Undefined numerical values become null.
        import math
        clean = {k: None if isinstance(v,float) and not math.isfinite(v) else v for k,v in metrics.items()}
        with self.path.open('a') as f:
            f.write(json.dumps({'step':step, 'budget':self.totals, 'metrics':clean},
                               default=lambda value: value.item())+'\n')
