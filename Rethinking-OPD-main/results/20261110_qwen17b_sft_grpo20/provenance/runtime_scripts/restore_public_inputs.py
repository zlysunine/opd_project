"""Fetch pinned public input assets, verify hashes, and combine native Parquet shards."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib,json,os,pathlib,shutil,subprocess,time,urllib.request

ROOT=pathlib.Path('/root/autodl-tmp/opsd_inputs')
MODEL_REV='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
DATA_REV='5cf055d1fe3d7a2eb19719ac020211469736ae44'
MODEL_DIR=ROOT/'models/Qwen3-1.7B-Base'
DATA_DIR=ROOT/'datasets/DeepMath-103K'/DATA_REV
LOGS=ROOT/'download_logs'
LOGS.mkdir(parents=True,exist_ok=True)

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def metadata(kind,repo,revision):
    url=f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'opsd-training-inputs'}),timeout=60) as r:info=json.load(r)
    if info['sha']!=revision:raise RuntimeError('Official revision mismatch')
    return info

def digest(path,algorithm='sha256',prefix=b''):
    h=hashlib.new(algorithm);h.update(prefix)
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def parallel_model(url,partial,total,workers=8):
    import math,requests
    session=requests.Session();session.trust_env=False
    parts=partial.with_name(partial.name.removesuffix('.incomplete')+'.parts')
    parts.mkdir(exist_ok=True)
    layout=parts/'layout.json'
    if not layout.exists():
        prefix=partial.stat().st_size if partial.exists() else 0
        if partial.exists():partial.replace(parts/'prefix')
        width=math.ceil((total-prefix)/workers)
        ranges=[dict(index=i,start=start,end=min(total-1,start+width-1)) for i,start in enumerate(range(prefix,total,width))] if width else []
        save(layout,dict(prefix=prefix,total=total,ranges=ranges))
    plan=json.loads(layout.read_text())
    if plan['total']!=total:raise RuntimeError('Partial model identity changed')
    def piece(item):
        path=parts/f"part_{item['index']:02d}"
        target_size=item['end']-item['start']+1
        for attempt in range(8):
            have=path.stat().st_size if path.exists() else 0
            if have==target_size:return
            if have>target_size:raise RuntimeError('Oversized model part')
            cursor=item['start']+have
            try:
                with session.get(url+f'&range_start={cursor}',headers={'Range':f"bytes={cursor}-{item['end']}"},stream=True,timeout=(25,90)) as response:
                    response.raise_for_status()
                    expected=f"bytes {cursor}-{item['end']}/{total}"
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise RuntimeError('Server did not honor the requested model byte range')
                    with path.open('ab') as stream:
                        for block in response.iter_content(1024*1024):
                            if block:stream.write(block)
                if path.stat().st_size==target_size:
                    print('ASSET_PART_COMPLETE',partial.name,item['index'],target_size,flush=True)
                    return
            except Exception as error:
                print('ASSET_PART_RETRY',partial.name,item['index'],attempt,type(error).__name__,str(error)[:140],flush=True)
                time.sleep(3)
        raise RuntimeError(f"Model part {item['index']} failed after retries")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in as_completed([pool.submit(piece,r) for r in plan['ranges']]):result.result()
    with partial.open('wb') as out:
        for path in ([parts/'prefix'] if plan['prefix'] else [])+[parts/f"part_{r['index']:02d}" for r in plan['ranges']]:
            with path.open('rb') as source:shutil.copyfileobj(source,out,8*1024*1024)

model=metadata('models','Qwen/Qwen3-1.7B-Base',MODEL_REV)
data=metadata('datasets','zwhe99/DeepMath-103K',DATA_REV)
save(ROOT/'official_model_metadata.json',model);save(ROOT/'official_dataset_metadata.json',data)
tasks=[]
for kind,repo,revision,info,directory in [('models','Qwen/Qwen3-1.7B-Base',MODEL_REV,model,MODEL_DIR),('datasets','zwhe99/DeepMath-103K',DATA_REV,data,DATA_DIR)]:
    for item in info['siblings']:
        name=item['rfilename']
        if kind=='datasets' and not (name.startswith('data/') and name.endswith('.parquet') or name=='README.md'):continue
        if kind=='models' and name=='.gitattributes':continue
        tasks.append((kind,repo,revision,item,directory))

def fetch(task):
    kind,repo,revision,item,directory=task
    name=item['rfilename'];target=directory/name;target.parent.mkdir(parents=True,exist_ok=True)
    expected=item.get('lfs',{}).get('sha256') if item.get('lfs') else None
    expected_size=item['size']
    if not target.exists():
        partial=target.with_name(target.name+'.incomplete')
        prefix='datasets/' if kind=='datasets' else ''
        url=f'https://hf-mirror.com/{prefix}{repo}/resolve/{revision}/{name}?download=true'
        logfile=LOGS/(kind+'_'+name.replace('/','_')+'.log')
        print('DOWNLOAD_START',repo,name,expected_size,flush=True)
        parallel_model(url,partial,expected_size,workers=4 if name=='model.safetensors' else 2)
        if partial.stat().st_size!=expected_size:raise RuntimeError(f'Size mismatch: {partial}')
        actual=digest(partial)
        if expected and actual!=expected:raise RuntimeError(f'SHA256 mismatch: {partial}')
        partial.replace(target)
        if target.with_name(target.name+'.parts').exists():
            shutil.rmtree(target.with_name(target.name+'.parts'))
    if target.stat().st_size!=expected_size:raise RuntimeError(f'Existing asset size mismatch: {target}')
    actual=digest(target)
    if expected and actual!=expected:raise RuntimeError(f'Existing asset SHA256 mismatch: {target}')
    print('ASSET_VERIFIED',repo,name,expected_size,actual,flush=True)
    return dict(repo=repo,revision=revision,relative_path=name,path=str(target),bytes=expected_size,sha256=actual)

records=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for f in as_completed([pool.submit(fetch,t) for t in tasks]):
        records.append(f.result());save(ROOT/'verified_assets.json',records)

import pyarrow.parquet as pq
native=DATA_DIR/'native_train.parquet'
shards=sorted((DATA_DIR/'data').glob('train-*.parquet'))
required={'question','final_answer','difficulty','topic','r1_solution_1'}
if len(shards)!=10:raise RuntimeError('Expected ten native DeepMath shards')
if not native.exists():
    tmp=native.with_name(native.name+'.incomplete')
    rows=0
    with pq.ParquetWriter(tmp,pq.ParquetFile(shards[0]).schema_arrow,compression='zstd') as writer:
        for shard in shards:
            source=pq.ParquetFile(shard)
            if not required<=set(source.schema_arrow.names):raise RuntimeError('Missing native fields')
            for batch in source.iter_batches(batch_size=1024):
                writer.write_batch(batch);rows+=batch.num_rows
            print('COMBINED_SHARD',shard.name,'rows_so_far',rows,flush=True)
    tmp.replace(native)
else:rows=pq.ParquetFile(native).metadata.num_rows
identity=dict(model_id='Qwen/Qwen3-1.7B-Base',model_revision=MODEL_REV,model_dir=str(MODEL_DIR),dataset_id='zwhe99/DeepMath-103K',dataset_revision=DATA_REV,dataset_file=str(native),dataset_rows=rows,dataset_sha256=digest(native),original_shards=[r for r in records if r['repo']=='zwhe99/DeepMath-103K' and r['relative_path'].endswith('.parquet')],conversion='All columns preserved; native shards concatenated in filename order with Arrow, Zstandard Parquet compression; no row filtering')
save(ROOT/'inputs.json',identity)
print('INPUTS_READY',json.dumps(identity),flush=True)
