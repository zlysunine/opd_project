"""Offline model adapter used for reference generation, evaluation and extraction.

Production optimization stays in verl/LLaMA-Factory. Tiny GPT-2 is only a local
wiring test; it is never presented as Qwen experimental evidence.
"""
import json
from pathlib import Path

import torch
from .common import opsd, digest, checkpoint_identity


class ByteTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 0, 1, 2
    chat_template = 'byte-smoke:user+assistant:v1'
    def get_vocab(self):
        return {f'byte_{i}': i+3 for i in range(256)} | {'PAD':0, 'EOS':1, 'BOS':2}
    def encode(self, text, **kwargs):
        return [c+3 for c in text.encode('utf8')]
    def decode(self, ids, **kwargs):
        return bytes([i-3 for i in ids if 3<=i<259]).decode('utf8', errors='replace')
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = ''.join(m['role']+': '+m['content']+'\n' for m in messages)
        text += 'assistant: ' if add_generation_prompt else ''
        return [self.bos_token_id]+self.encode(text) if tokenize else text


def trajectory_ids(row, tokenizer):
    ids=row.get('response_ids')
    if not isinstance(ids,list) or not ids:
        raise ValueError('Expected nonempty response_ids list')
    length=row.get('response_length',len(ids))
    if not 0<length<=len(ids):raise ValueError('Invalid response_length')
    ids=ids[:length]
    if tokenizer.eos_token_id in ids:
        ids=ids[:ids.index(tokenizer.eos_token_id)+1]
    return ids


def create_tiny(path, seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=259, n_positions=2048, n_embd=24, n_layer=2, n_head=2,
                                    bos_token_id=2, eos_token_id=1, pad_token_id=0,
                                    resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
    model.save_pretrained(path)
    (Path(path)/'byte_tokenizer.json').write_text(json.dumps({'type':'byte_smoke_v1'}))


class Model:
    def __init__(self, path, revision, device='cpu', dtype='float32', template_path=None, thinking=False,
                 prompt_mode='chat'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.path, self.revision, self.device = str(path), revision, device
        if path is None:
            raise ValueError('Provide an existing local model path (--model)')
        if not revision or revision in ('main', 'master'):
            raise ValueError('Explicit model revision/source identifier is required')
        if not Path(path).is_dir():
            raise FileNotFoundError(f'Local model missing: {path}; download explicitly, no fallback')
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                     torch_dtype=getattr(torch, dtype), attn_implementation='eager').to(device).eval()
        self.tokenizer = ByteTokenizer() if (Path(path)/'byte_tokenizer.json').exists() else AutoTokenizer.from_pretrained(path, local_files_only=True)
        if template_path:
            self.tokenizer.chat_template = Path(template_path).read_text()
        if prompt_mode not in ('chat','plain_completion'):
            raise ValueError(prompt_mode)
        self.prompt_mode = prompt_mode
        if not self.tokenizer.chat_template and prompt_mode=='chat':
            raise ValueError('Checkpoint has no chat template; provide --template from a verified source')
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Explicit tokenizer pad token required; do not silently alter tokenizer')
        self.thinking = thinking
        self.identity = {'path': str(Path(path).resolve()), 'revision': revision,
                         'content_hash': checkpoint_identity(path), 'model_config_hash': digest(self.model.config.to_dict()),
                         'tokenizer_hash': digest(self.tokenizer.get_vocab()),
                         'template_hash': digest(self.tokenizer.chat_template if prompt_mode=='chat'
                                                 else opsd.plain_messages('{question}')[0]['content']),
                         'enable_thinking': thinking if prompt_mode=='chat' else None, 'prompt_mode':prompt_mode,
                         'dtype': dtype, 'privilege_template_hash': digest(opsd.PRIVILEGE_TEMPLATE if prompt_mode=='chat'
                                                         else 'Reference information:\n{reference}\n\n')}

    def prompt(self, row, privileged=False):
        if self.prompt_mode=='plain_completion':
            messages=opsd.plain_messages(row['question'])
            if privileged: messages=opsd.privileged_messages(messages,row['privilege_text'],self.prompt_mode)
            return self.tokenizer.encode(messages[0]['content'],add_special_tokens=False)
        messages = opsd.privileged_messages(row['messages'], row['privilege_text']) if privileged else row['messages']
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=self.thinking)

    def pack(self, prompt, response):
        ids = torch.tensor([response], device=self.device, dtype=torch.long)
        return opsd.pack_responses([prompt], ids, torch.ones_like(ids), self.tokenizer.pad_token_id,
                                   self.model.config.max_position_embeddings-len(response))

    def logits(self, prompt, response):
        batch = self.pack(prompt, response)
        output = self.model(**{k:v for k,v in batch.items() if k!='responses'}, use_cache=False)
        return output.logits[0, len(prompt)-1:len(prompt)-1+len(response)]

    @torch.no_grad()
    def generate(self, row, max_tokens=128, temperature=1., top_p=1., privileged=False, greedy=False):
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError('Sampling requires positive temperature and 0<top_p<=1')
        prompt = self.prompt(row, privileged)
        if len(prompt)+max_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Prompt + generation exceeds model context; shorten explicitly')
        inputs = torch.tensor([prompt], device=self.device)
        sampling={} if greedy else dict(temperature=temperature,top_p=top_p,top_k=0)
        ids = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_tokens,
                                  **sampling, do_sample=not greedy, num_beams=1,
                                  pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)[0, len(prompt):].tolist()
        if self.tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(self.tokenizer.eos_token_id)+1]
        return ids, self.tokenizer.decode(ids, skip_special_tokens=True), not ids or ids[-1] != self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate_batch(self, rows, seeds, max_tokens=128, temperature=1., top_p=1., greedy=False):
        """Left-padded inference with a separate sampling RNG for every question.

        A logits processor samples each row independently; HF then selects the
        sampled token by argmax. This avoids sharing a batch-wide sampling RNG.
        Batched floating-point computation may differ from the old serial pool.
        """
        from transformers import LogitsProcessor, LogitsProcessorList

        if not rows or len(rows) != len(seeds) or temperature <= 0 or top_p != 1.:
            raise ValueError('Batched monitoring requires nonempty rows, matching seeds, temperature>0 and top_p=1')
        prompts = [self.prompt(row) for row in rows]
        width = max(map(len, prompts))
        if width + max_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Batched prompt + generation exceeds model context')
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, width-len(prompt):] = torch.tensor(prompt, device=ids.device)
            mask[i, width-len(prompt):] = 1

        class IndependentSampling(LogitsProcessor):
            def __init__(processor):
                processor.generators = [torch.Generator(device=ids.device).manual_seed(seed) for seed in seeds]

            def __call__(processor, input_ids, scores):
                probabilities = (scores.float() / temperature).softmax(-1)
                choices = torch.stack([torch.multinomial(p, 1, generator=g)
                                       for p, g in zip(probabilities, processor.generators)])
                return torch.full_like(scores, -float('inf')).scatter_(1, choices, 0.)

        processors = LogitsProcessorList([] if greedy else [IndependentSampling()])
        generated = self.model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens, do_sample=False,
            num_beams=1, logits_processor=processors, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[:, width:].tolist()
        results = []
        for response in generated:
            if self.tokenizer.eos_token_id in response:
                response = response[:response.index(self.tokenizer.eos_token_id)+1]
            results.append((response, self.tokenizer.decode(response, skip_special_tokens=True),
                            not response or response[-1] != self.tokenizer.eos_token_id))
        return results

    def layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        raise ValueError('Unsupported model block layout; add an explicit architecture adapter')

    def hook_module(self, layer, point):
        block = self.layers()[layer]
        if point == 'block_output':
            return block, {'includes_residual': True, 'after_layer_norm': False}
        if point == 'attention_output':
            return getattr(block, 'self_attn', None) or block.attn, {'includes_residual': False, 'after_layer_norm': False}
        if point == 'post_mlp':
            return block.mlp, {'includes_residual': False, 'after_layer_norm': False}
        raise ValueError(point)
