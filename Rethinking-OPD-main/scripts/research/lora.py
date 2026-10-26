"""Turn the vendored FSDP merger's base+adapter output into an evaluable HF model."""
import json
from pathlib import Path

from .common import write_json


def finalize_lora_merge(target, config):
    """Restore the REAL training alpha (vendored exporter defaults it to zero), then merge."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    target=Path(target)
    adapter=target/'lora_adapter'
    path=adapter/'adapter_config.json'
    if not path.is_file() or not (adapter/'adapter_model.safetensors').is_file():
        raise FileNotFoundError('LoRA checkpoint export is missing its adapter; refusing base-only evaluation')
    settings=json.loads(path.read_text())
    if settings['r']!=config['lora_rank']:
        raise ValueError('Exported LoRA rank differs from the training run')
    # The archived adapter must reference the ORIGINAL frozen base, not the output
    # that is about to contain merged weights (which would apply the adapter twice).
    base_source=settings.get('base_model_name_or_path','')
    if config.get('registry') and config.get('student'):
        registry=json.loads(Path(config['registry']).read_text())
        base_source=registry[config['student']]['path']
    settings.update(lora_alpha=config['lora_alpha'],base_model_name_or_path=base_source)
    write_json(path,settings)
    base=AutoModelForCausalLM.from_pretrained(target,local_files_only=True,
                                             torch_dtype=torch.float32,attn_implementation='eager')
    adapted=PeftModel.from_pretrained(base,adapter,local_files_only=True,is_trainable=False)
    merged=adapted.merge_and_unload(safe_merge=True)
    merged.save_pretrained(target,safe_serialization=True)
    write_json(target/'lora_merge.json',{'merged':True,'rank':settings['r'],
               'alpha':settings['lora_alpha'],'source_adapter':str(adapter)})
