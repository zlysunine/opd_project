"""Isolated, recoverable continuous-batch generation; HF retains geometry duties."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.research.common import digest, write_json


def normalize_response(ids, eos_token_id, finish_reason):
    ids = list(ids)
    if eos_token_id in ids:
        ids = ids[:ids.index(eos_token_id)+1]
    elif finish_reason == 'stop':
        # vLLM versions can omit the EOS from returned token IDs.
        ids.append(eos_token_id)
    elif finish_reason != 'length':
        raise ValueError(f'Unexpected vLLM finish reason: {finish_reason}')
    return ids, not ids or ids[-1] != eos_token_id


def generate(model, rows, seeds, max_tokens, directory, on_result, memory_utilization=.80):
    """Run vLLM in an owned subprocess, freeing its KV cache before HF analysis."""
    import torch
    if str(model.device) != 'cuda':
        raise ValueError('vLLM monitoring requires the CUDA model profile')
    path = Path(directory)/'vllm_raw'
    path.mkdir(exist_ok=True)
    request = dict(model=model.path, prompts=[model.prompt(r) for r in rows], seeds=seeds,
                   max_tokens=max_tokens, eos_token_id=model.tokenizer.eos_token_id,
                   memory_utilization=memory_utilization)
    request_path = path/'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('vLLM generation request changed')
    write_json(request_path, request)
    gc.collect()
    torch.cuda.empty_cache()
    env = os.environ.copy()
    # expandable_segments is for HF analysis, not vLLM's CUDA IPC allocations.
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    command = [sys.executable, '-u', '-m', 'analysis.vllm_generation', '--request', str(request_path)]
    with (path/'engine.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        delivered = set()
        try:
            while True:
                for i in range(len(rows)):
                    output = path/f'{i:06d}.json'
                    if i in delivered or not output.exists():
                        continue
                    result = json.loads(output.read_text())
                    if result['request_hash'] != digest(request):
                        raise ValueError('vLLM response belongs to another request')
                    ids, truncated = normalize_response(result['ids'], request['eos_token_id'], result['finish_reason'])
                    on_result(i, (ids, model.tokenizer.decode(ids, skip_special_tokens=True), truncated))
                    delivered.add(i)
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'vLLM exited {code}; inspect {path}/engine.log')
                    if len(delivered) == len(rows):
                        break
                    # One final scan after the worker publishes the last result.
                    if not all((path/f'{i:06d}.json').exists() for i in range(len(rows))):
                        raise ValueError('vLLM exited without all responses')
                time.sleep(.5)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()


def worker(request_path):
    from vllm import LLMEngine, EngineArgs, SamplingParams
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    root = request_path.parent
    pending = [i for i in range(len(request['prompts'])) if not (root/f'{i:06d}.json').exists()]
    if not pending:
        return
    args = EngineArgs(model=request['model'], tokenizer=request['model'], dtype='bfloat16',
                      tensor_parallel_size=1, gpu_memory_utilization=request['memory_utilization'],
                      max_model_len=max(map(len, request['prompts']))+request['max_tokens'],
                      max_num_seqs=64, max_num_batched_tokens=8192,
                      enable_prefix_caching=True, generation_config='vllm', seed=42,
                      compilation_config={'level':0, 'cudagraph_mode':'FULL_DECODE_ONLY',
                                          'cudagraph_capture_sizes':[1,2,4,8,16,32,64]})
    engine = LLMEngine.from_engine_args(args)
    for i in pending:
        params = SamplingParams(temperature=0., top_p=1., top_k=-1, max_tokens=request['max_tokens'],
                                seed=request['seeds'][i], stop_token_ids=[request['eos_token_id']])
        engine.add_request(str(i), {'prompt_token_ids':request['prompts'][i]}, params)
    started = time.monotonic()
    while engine.has_unfinished_requests():
        for result in engine.step():
            if not result.finished:
                continue
            output = result.outputs[0]
            index = int(result.request_id)
            write_json(root/f'{index:06d}.json', dict(request_hash=digest(request), ids=list(output.token_ids),
                       finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                       elapsed_seconds=time.monotonic()-started))
            print(f'VLLM_COMPLETED index={index} tokens={len(output.token_ids)} '
                  f'elapsed={time.monotonic()-started:.1f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    worker(parser.parse_args().request)
