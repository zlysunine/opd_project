from pathlib import Path
import os,subprocess,signal,time,re
from packaging.utils import parse_wheel_filename,canonicalize_name
from packaging.specifiers import SpecifierSet
setup=Path('/root/autodl-tmp/sft_grpo_setup')
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if cmd.startswith(b'/root/autodl-tmp/envs/sft_grpo/bin/python\0-m\0pip\0install\0'):
  os.kill(int(p.name),signal.SIGTERM)
  print('Stopped previous dependency resolver',p.name,flush=True)
time.sleep(1)
pins={canonicalize_name(s.split('==')[0]):SpecifierSet('=='+s.split('==')[1]) for s in (setup/'constraints-pinned-history.txt').read_text().splitlines()}
local=[];local_names=set()
for p in (setup/'wheels').glob('*.whl'):
 name,version,_,_=parse_wheel_filename(p.name)
 if name in pins and version not in pins[name]:continue
 local_names.add(name)
 local.append(str(p)+('[default]' if name=='ray' else ''))
required=['numpy==1.26.4','transformers==4.56.1','peft==0.17.1','vllm==0.11.0','ray[default]==2.59.0','pyarrow==21.0.0','tensordict==0.10.0','torchdata==0.11.0','hydra-core','pandas>=2,<3','scipy','scikit-learn','accelerate','datasets','codetiming','pylatexenc','sympy','wandb','tensorboard','latex2sympy2_extended','math_verify','plotly','ninja','packaging','setuptools','wheel']
required=[r for r in required if canonicalize_name(re.split(r'[\[<>=]',r)[0]) not in local_names]
print('EXPLICIT_LOCAL_WHEELS',len(local),'OTHER_REQUIREMENTS',required,flush=True)
env=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
command=['/root/autodl-tmp/envs/sft_grpo/bin/python','-m','pip','install','--no-cache-dir','-c',str(setup/'constraints-pinned-history.txt'),*required,*local]
subprocess.run(command,env=env,check=True)
