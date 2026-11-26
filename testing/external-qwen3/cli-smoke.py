#!/usr/bin/env python3
# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU CLI qualification using the committed tiny compose and real corpus input.

Run after building waldo: python3 testing/external-qwen3/cli-smoke.py --binary
/path/to/waldo --work /new/workdir --application-cache /persistent/cache
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--work', required=True)
p.add_argument('--application-cache', required=True)
p.add_argument('--python', default='/usr/bin/python3')
p.add_argument('--compose', default='examples/trainers/qwen3-hf/compose.tiny.yaml',
               help='external compose, relative to the repository or absolute')
p.add_argument('--trainer-environment', default='',
               help='rewrite the trainer environment (for example cuda) in a copy of the compose')
p.add_argument('--data', action='append', default=[], metavar='KEY=VALUE',
               help='override a stage data key (for example steps=10) in a copy of the compose')
p.add_argument('--processes-per-node', type=int, default=1)
p.add_argument('--source-repository', default='')
p.add_argument('--source-commit', default='')
p.add_argument('--tokenizer-environment', default='')
p.add_argument('--records', type=int, default=20, help='fixture corpus record count')
p.add_argument('--stop', default='checkpoint', choices=['checkpoint', 'none'],
               help='none skips the interrupted/resumed pair and trains one run (large compose)')
a = p.parse_args()
work = Path(a.work).resolve()
work.mkdir(parents=True, exist_ok=False)
binary = str(Path(a.binary).resolve())
repo = Path(__file__).resolve().parents[2]
env = dict(os.environ, WALDO_CONFIG=str(work/'config.json'),
           WALDO_APPLICATION_CACHE=str(Path(a.application_cache).resolve()),
           WALDO_APPLICATION_PYTHON=a.python)

def training_args(args):
    args = list(args)
    if 'train' in args and a.processes_per_node > 1:
        args += ['--launch-profile', 'distributed', '--processes-per-node', str(a.processes_per_node)]
    return args

def run(*args):
    args = training_args(args)
    result = subprocess.run([binary, *args], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with (work/'commands.log').open('a') as log:
        log.write('waldo '+ ' '.join(args)+'\n'+result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout

raw = work/'input'/'raw'
raw.mkdir(parents=True)
rows = ''.join(json.dumps({'text': (f'Book {i}. The quick brown fox reads a book about science and history. '*30)})+'\n' for i in range(a.records)).encode()
(raw/'records.jsonl').write_bytes(rows)
sha = hashlib.sha256(rows).hexdigest()
tree = hashlib.sha256(f'{sha}\t{len(rows)}\trecords.jsonl\n'.encode()).hexdigest()
manifest = {'kind':'waldo-source-directory','schema':1,'retrieved_at':'2026-11-03T00:00:00Z',
 'corpus':{'id':'books','title':'CLI smoke books','description':'Disposable external trainer CLI fixture.'},
 'sources':[{'id':'books','path':'','license':'CC0-1.0',
 'source':{'name':'books','version':'fixture-1','url':'https://example.invalid/books','category':'public-dataset','license_evidence':{'declaration':'CC0-1.0'}},
 'input':{'format':'jsonl','type':'record-map','fields':{'text':['text']}},'artifacts':[]}],
 'fetcher':{'name':'cli-smoke'},'raw':{'path':'raw','file_count':1,'byte_count':len(rows),'tree_sha256':tree}}
(work/'input'/'manifest.json').write_text(json.dumps(manifest))
run('index','init',str(work/'index'))
for key,value in {'lookaside':(work/'lookaside').as_uri(),'lookaside.cache':str(work/'cache'),
 'lookaside.scratch':str(work/'scratch'),'ingest.staging':str(work/'staging'),
 'model.root':str(work/'models'),'index':str(work/'index')}.items():
    run('config','set',key,value)
run('index','ingest',str(work/'input'),str(work/'index'/'books'))
contributions = list((work/'staging').glob('*/contribution'))
assert len(contributions)==1
shutil.copytree(contributions[0],work/'index',dirs_exist_ok=True)
# Isolate corpus provenance from commits in a parent WALDO checkout.
subprocess.run(['git','init',str(work/'index')],check=True,capture_output=True)
subprocess.run(['git','-C',str(work/'index'),'add','.'],check=True)
subprocess.run(['git','-C',str(work/'index'),'-c','user.name=CLI smoke',
                '-c','user.email=smoke@example.invalid','commit','-s','-m',
                'Pin disposable corpus fixture'],check=True,capture_output=True)
subprocess.run(['git','init','--bare',str(work/'index-origin.git')],check=True,capture_output=True)
subprocess.run(['git','-C',str(work/'index'),'remote','add','origin',
                str(work/'index-origin.git')],check=True)
subprocess.run(['git','-C',str(work/'index'),'push','-u','origin','HEAD'],
               check=True,capture_output=True)
provider = {'kind':'waldo-disclosure-provider','schema':1,'provider':{'name':'CLI smoke','address':'Test fixture','contact':'test@example.invalid'},'code_of_practice_status':'not-assessed','copyright_policy_url':'https://example.invalid/copyright'}
(work/'provider.json').write_text(json.dumps(provider))
run('config','set','disclosure.provider',str(work/'provider.json'))
compose = str((repo/a.compose).resolve())
if a.trainer_environment:
    text = Path(compose).read_text()
    assert text.count('  environment: cpu\n') == 1 or text.count('  environment: cuda\n') == 1
    text = text.replace('  environment: cpu\n', f'  environment: {a.trainer_environment}\n', 1).replace('  environment: cuda\n', f'  environment: {a.trainer_environment}\n', 1)
    compose = str(work/'compose.yaml')
    Path(compose).write_text(text)
if a.source_repository or a.source_commit or a.tokenizer_environment:
    import re
    text = Path(compose).read_text()
    if a.source_repository:
        text = re.sub(r'(?m)^(    repository: ).*$', lambda m: m[1] + a.source_repository, text)
    if a.source_commit:
        text = re.sub(r'(?m)^(    commit: ).*$', lambda m: m[1] + a.source_commit, text)
    if a.tokenizer_environment:
        text = text.replace('  environment: tokenizer\n', '  environment: '+a.tokenizer_environment+'\n')
    compose = str(work/'compose.yaml')
    Path(compose).write_text(text)
if a.data:
    import re
    text = Path(compose).read_text()
    for item in a.data:
        key, value = item.split('=', 1)
        text, n = re.subn(rf'(?m)^(      {re.escape(key)}: ).*$', rf'\g<1>{value}', text)
        assert n == 1, key
    compose = str(work/'compose.yaml')
    Path(compose).write_text(text)
if a.stop == 'none':
    # Single budget-capped run: record wall time and peak GPU memory.
    import threading, time
    peak = [0]; done = threading.Event()
    def sample():
        while not done.is_set():
            # GB10 has unified memory and nvidia-smi reports N/A, so sample host memory in use.
            info = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
            peak[0] = max(peak[0], (info['MemTotal'] - info['MemAvailable']) // 1024)
            done.wait(1)
    threading.Thread(target=sample, daemon=True).start()
    info0 = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    base = [(info0['MemTotal'] - info0['MemAvailable']) // 1024]
    started = time.monotonic()
    run('--json','model','train','single',compose)
    elapsed = time.monotonic() - started
    done.set()
    record = json.loads((work/'models/single/runs/pretrain/RUN.json').read_text())
    assert record['state'] == 'complete'
    print(json.dumps({'wall_seconds': round(elapsed, 1), 'peak_host_memory_in_use_mib_sampled': peak[0], 'baseline_host_memory_in_use_mib': base[0], 'complete': record['complete']}), flush=True)
    print('PASS: single budget-capped run', flush=True)
    raise SystemExit(0)
print('Training uninterrupted control',flush=True)
run('--json','model','train','control',compose)
print('Interrupting run after first durable checkpoint',flush=True)
process = subprocess.Popen([binary,*training_args(['--json','model','train','resumed',compose])],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
interrupted = False
with (work/'interrupted.log').open('w') as log:
    for line in process.stderr:
        log.write(line)
        try: event=json.loads(line)
        except json.JSONDecodeError: continue
        if not interrupted and event.get('training',{}).get('kind')=='checkpoint':
            process.send_signal(signal.SIGINT)
            interrupted=True
    log.write(process.stdout.read())
code=process.wait()
assert interrupted and code!=0, (interrupted,code)
record_path=work/'models/resumed/runs/pretrain/RUN.json'
record=json.loads(record_path.read_text())
assert record['state']=='interrupted' and record['checkpoints']
original_bom=(work/'models/resumed/runs/pretrain/RUN-BOM.json').read_bytes()
print('Resuming same CLI command',flush=True)
run('--json','model','train','resumed',compose)
record=json.loads(record_path.read_text())
assert record['state']=='complete' and len(record['attempts'])==2
assert original_bom==(work/'models/resumed/runs/pretrain/RUN-BOM.json').read_bytes()
for artifact in record['complete']['artifacts']:
    if artifact['role']=='weights':
        relative=Path('runs/pretrain/artifacts')/artifact['path']
        assert (work/'models/resumed'/relative).read_bytes()==(work/'models/control'/relative).read_bytes()
# Completed re-invocation must not add another attempt.
run('--json','model','train','resumed',compose)
assert len(json.loads(record_path.read_text())['attempts'])==2
run('--json','model','continue','resumed')
assert len(json.loads(record_path.read_text())['attempts'])==2
print('Exporting verified HF package',flush=True)
run('--json','model','export','resumed',str(work/'export'),'--format','huggingface','--allow-incomplete')
export=work/'export'
assert (export/'model.safetensors').is_file() and (export/'config.json').is_file()
bom=json.loads((export/'BOM.json').read_text())
assert bom['schema']==2 and bom['identity']==json.loads((work/'models/resumed/PLAN.json').read_text())['identity']
assert not list(export.rglob('*.pt')) and not list(export.rglob('MANIFEST.json'))
for r in bom['runs']:
    for artifact in [r['bom'],*r['artifacts']]:
        data=(export/artifact['path']).read_bytes()
        assert len(data)==artifact['bytes'] and hashlib.sha256(data).hexdigest()==artifact['sha256']
print('PASS: CLI train, checkpoint interruption, same-run resume, byte-identical weights, completed no-op, and export',flush=True)
# Exercise the configured signer subprocess contract without requiring external
# credentials. These mock bundles are test evidence, NOT valid signatures.
mockbin=work/'mockbin'
mockbin.mkdir()
cosign=mockbin/'cosign'
cosign.write_text('''#!/usr/bin/env python3
import hashlib,json,pathlib,sys
args=sys.argv[1:]
bundle=pathlib.Path(args[args.index('--bundle')+1])
document=pathlib.Path(args[-1])
bundle.write_text(json.dumps({'mock':True,'sha256':hashlib.sha256(document.read_bytes()).hexdigest()}))
''')
cosign.chmod(0o755)
env['PATH']=str(mockbin)+os.pathsep+env['PATH']
(work/'mock-key').write_text('test fixture; not a real signing key')
run('config','set','signing.method','sigstore-key')
run('config','set','signing.key',str(work/'mock-key'))
result=json.loads(run('--json','model','export','resumed',str(work/'signed-export'),'--allow-incomplete'))
assert result['signed'] is True
for name in ('BOM','EU-BOM'):
    bundle=json.loads((work/'signed-export'/f'{name}.sigstore.json').read_text())
    assert bundle['mock'] and bundle['sha256']==hashlib.sha256((work/'signed-export'/f'{name}.json').read_bytes()).hexdigest()
run('--json','model','summary','resumed')
run('--json','model','list')
assert json.loads(run('model','bom','resumed'))['model_id']==record['model_id']
print('PASS: configured signing dispatch (mock cosign), summary, list, and BOM',flush=True)
