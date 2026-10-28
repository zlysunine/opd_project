"""Recover accepted rows using saved decisions; do not repeat reference verification."""
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scripts.research.common import digest, file_hash, write_json, write_jsonl
from scripts.research.data import canonical, prepare_deepmath
from scripts.research.answer_protocol import answer_metadata
from scripts.research.preparation_progress import PreparationProgress

ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
WD=ROOT/'research_runs/base_opsd_20261028_acceptance_2step'
AUDIT=WD/'shared/preparation_audit'
config=json.loads((WD/'config.json').read_text())
progress=PreparationProgress(AUDIT/'recovery_progress.json')
started=time.time()
try:
    checks={r['problem_id']:r for r in map(json.loads,(AUDIT/'eligibility.jsonl').open())}
    accepted_ids={pid for pid,r in checks.items() if r['exclusion_reason'] is None}
    native=config['dataset']['input']
    expected_hash='e4981f9b3f3b2840b71539a1970858f1448ea798f26b1e976031cbb1fb7780b5'
    native_hash=file_hash(native)
    assert native_hash==expected_hash,'Native file differs from originally checked dataset'
    reader=pq.ParquetFile(native)
    progress.phase('recover_native_rows',total=reader.metadata.num_rows,accepted=0)
    selected=[];index=0
    for batch in reader.iter_batches(batch_size=256):
        for raw in batch.to_pylist():
            row=canonical(raw,index)
            original=row['problem_id']
            row={**row,'source_problem_id':original,'problem_id':f'{original}__row{index}'}
            if row['problem_id'] in accepted_ids:selected.append(row)
            index+=1
        progress.update(index,reader.metadata.num_rows,accepted=len(selected))
    assert {r['problem_id'] for r in selected}==accepted_ids
    tokenizer=AutoTokenizer.from_pretrained(config['model']['path'],local_files_only=True)
    progress.phase('recover_accepted_metadata',total=len(selected),accepted=len(selected))
    rows=prepare_deepmath(selected,AUDIT/'accepted_pool.adapter.jsonl',config['dataset']['revision'],
        config['dataset']['difficulty_min'],config['dataset']['difficulty_max'],
        solution_field=config['dataset']['reference_field'],tokenizer=tokenizer,
        progress=lambda done,total,n:progress.update(done,total,accepted=n))
    accepted=[]
    for row in rows:
        check=checks[row['problem_id']]
        metadata=answer_metadata(row)
        assert metadata['answer_encoding_status']=='ok' and check['reference_verified']
        accepted.append({**row,**metadata,'reference_correctness':check['reference_verified']})
    target=AUDIT/'accepted_pool.recovered.jsonl'
    write_jsonl(target,accepted)
    write_json(AUDIT/'accepted_pool.recovery_manifest.json',dict(
        status='completed',accepted_before_deduplication=len(accepted),
        decision_count=len(checks),decision_sha256=file_hash(AUDIT/'eligibility.jsonl'),
        native_sha256=native_hash,native_revision=config['dataset']['revision'],
        accepted_ids_sha256=digest(sorted(accepted_ids)),recovered_sha256=file_hash(target),
        reused='original per-question eligibility and reference-verification decisions',
        reconstructed='original row payload and derived reference-token-length metadata only',
        source_snapshot=str(ROOT/'research_runs/server_handoff_20261028/before_progress_feature/snapshot.json'),
        no_new_quality_receipts=True,wall_seconds=time.time()-started))
    progress.finish()
    print('Recovered original accepted pool:',len(accepted),'rows at',target,flush=True)
except BaseException as error:
    progress.finish(error)
    raise
