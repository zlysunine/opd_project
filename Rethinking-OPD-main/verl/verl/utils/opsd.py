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

"""Text-only OPSD alignment and the existing OPD surrogate's pure tensor terms.

No dependency on Ray/DataProto: the CPU smoke and FSDP worker share this code.
"""
import copy

import torch
from torch.utils.checkpoint import checkpoint
from scripts.research.answer_protocol import build_student_prompt

PRIVILEGE_TEMPLATE = "\n\nReference information (possibly empty):\n{reference}\nSolve the original question using any useful information above."


def plain_messages(question):
    """Base continuation protocol, shared by training and independent probes."""
    return [{"role": "user", "content": build_student_prompt(question)}]


def privileged_messages(messages, reference, prompt_mode="chat"):
    messages = copy.deepcopy(list(messages))
    for message in reversed(messages):
        if message["role"] == "user":
            if not isinstance(message["content"], str):
                raise ValueError("OPSD currently supports text-only user messages")
            if prompt_mode == "plain_completion":
                message["content"] = f"Reference information:\n{reference}\n\n" + message["content"]
            elif prompt_mode == "chat":
                message["content"] += PRIVILEGE_TEMPLATE.format(reference=reference)
            else:
                raise ValueError(f"Unknown prompt mode: {prompt_mode}")
            return messages
    raise ValueError("OPSD requires a user message")


def pack_responses(prompts, responses, response_mask, pad_id, max_prompt_length):
    """Left-pad prompts, append unchanged response IDs; EOS validity is supplied.

    Logits at P-1+t predict response[t]. Position IDs count only unmasked tokens.
    Reject rather than silently truncate privileged prompts.
    """
    if len(prompts) != len(responses) or response_mask.shape != responses.shape:
        raise ValueError("Batch or response mask mismatch")
    if not prompts or any(not p or len(p) > max_prompt_length for p in prompts):
        raise ValueError("Empty or overlong prompt; increase max_teacher_prompt_length explicitly")
    width = max(map(len, prompts))
    ids = responses.new_full((len(prompts), width + responses.shape[1]), pad_id)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, width-len(prompt):width] = torch.as_tensor(prompt, device=ids.device)
        mask[i, width-len(prompt):width] = 1
    ids[:, width:] = responses
    mask[:, width:] = response_mask
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0), "responses": responses}


def privileged_batch(extra_info, tokenizer, responses, response_mask, config):
    """Build teacher inputs from metadata without changing student tokens or prompts."""
    prompts = []
    kwargs = config.get("chat_template_kwargs", {"enable_thinking": False})
    for info in extra_info:
        if "messages" not in info or "privilege_text" not in info:
            raise ValueError("OPSD requires extra_info.messages and privilege_text")
        messages = privileged_messages(info["messages"], info["privilege_text"], config.get("prompt_mode", "chat"))
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        ))
    return pack_responses(prompts, responses, response_mask, tokenizer.pad_token_id,
                          config["max_teacher_prompt_length"])


def forward_kl_sum(student_logits, teacher_logits, response_mask, student_temperature=1.0,
                   teacher_temperature=1.0, chunk_size=16, checkpoint_chunks=True):
    """Sum full-vocabulary KL(stopgrad(teacher) || student) over valid positions.

    Input logits predict the SAME response positions, with shape [B, T, V].
    Prompt/padding positions never contribute. Teacher targets are always detached.
    Checkpointing recomputes chunk softmaxes in backward instead of retaining all
    FP32 probability matrices. Divide by the global valid-token count at the caller.
    """
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Forward KL requires matching [batch, response_length, vocab] logits")
    if response_mask.shape != student_logits.shape[:2]:
        raise ValueError("Forward KL response mask shape mismatch")
    if student_temperature <= 0 or teacher_temperature <= 0 or chunk_size < 1:
        raise ValueError("Forward KL temperatures and chunk_size must be positive")
    s = student_logits.reshape(-1, student_logits.shape[-1])
    q = teacher_logits.detach().reshape_as(s)
    valid = response_mask.bool().reshape(-1)
    # Empty connected sum: excludes nonfinite padding and produces zero gradients.
    total = s.reshape(-1)[:0].sum()
    dtype = torch.float64 if s.dtype == torch.float64 else torch.float32

    def loss_chunk(student, teacher):
        log_p = (student.to(dtype) / student_temperature).log_softmax(-1)
        with torch.no_grad():
            log_q = (teacher.to(dtype) / teacher_temperature).log_softmax(-1)
            prob_q = log_q.exp()
        return (prob_q * (log_q - log_p)).sum()

    for start in range(0, len(s), chunk_size):
        mask = valid[start:start + chunk_size]
        if not mask.any():
            continue
        student = s[start:start + chunk_size][mask]
        teacher = q[start:start + chunk_size][mask]
        if checkpoint_chunks and student.requires_grad:
            term = checkpoint(loss_chunk, student, teacher, use_reentrant=False)
        else:
            term = loss_chunk(student, teacher)
        total = total + term
    return total


def response_mask_until_eos(responses, eos_id, lengths=None):
    """Include the first EOS, exclude later padding; truncated non-EOS tokens count."""
    eos = responses.eq(eos_id)
    mask = (eos.cumsum(-1) - eos.to(torch.long)).eq(0)
    if lengths is not None:
        mask &= torch.arange(responses.shape[1], device=responses.device)[None] < lengths[:, None]
    return mask.to(torch.long)


def assert_tokenizer_compatible(student, teacher, student_vocab_size=None, teacher_vocab_size=None):
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Tokenizer token-to-ID mapping mismatch; OPD logits cannot be aligned")
    for name in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if getattr(student, name) != getattr(teacher, name):
            raise ValueError(f"Tokenizer {name} mismatch")
    if student_vocab_size is not None and student_vocab_size != teacher_vocab_size:
        raise ValueError("Model output vocabulary size mismatch")


def student_topk_reward(student_logp, teacher_on_student_logp):
    """Repository only_stu/student_p: normalized p weights, RAW log q-log p.

    This is not the KL between two top-k-renormalized distributions.
    All reward/weight terms stop gradient, as in compute_distillation_reward.
    """
    with torch.no_grad():
        return (teacher_on_student_logp - student_logp) * student_logp.softmax(-1)


def clipped_surrogate_terms(log_prob, old_log_prob, advantages, low=0.2, high=0.2, dual_clip=3.0):
    """Exact existing vanilla PPO terms, shared by original OPD and CPU smoke."""
    delta = (log_prob - old_log_prob).clamp(-20, 20)
    ratio = delta.exp()
    loss1 = -advantages * ratio
    loss2 = -advantages * ratio.clamp(1-low, 1+high)
    clipped = torch.maximum(loss1, loss2)
    losses = torch.where(advantages < 0, torch.minimum(-advantages * dual_clip, clipped), clipped)
    return losses, delta, loss1, loss2
