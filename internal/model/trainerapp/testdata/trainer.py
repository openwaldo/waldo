# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import json
import math
import sys
import struct

MODE = 'ok'

def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, protocol=1, **fields)), flush=True)

command = sys.argv[1]
request = json.loads(sys.stdin.readline())
if command == 'probe':
    result = dict(kind='probe', protocol=1,
                  capabilities=dict(objectives=['causal-language-modeling'], launch_profiles=['single-process'], resume=True),
                  frameworks={'fake': '1'}, devices=[dict(name='cpu', kind='cpu', memory_bytes=1024)], driver='unavailable', cuda='unavailable',
                  schemas=dict(architecture={'type': 'object'}, parameters={'type': 'object', 'properties': {
                      'lr': {'type': 'number'}, 'micro': {'type': 'integer', 'x-waldo-role': 'execution'}}}))
    if MODE == 'version': result['protocol'] = 9
    if MODE == 'schemas': del result['schemas']
    if MODE == 'role': result['schemas']['parameters']['properties']['micro']['x-waldo-role'] = 'host'
    if MODE == 'capability': result['capabilities']['resume'] = False
    if MODE == 'objective': result['capabilities']['objectives'] = []
    if MODE == 'missing': del result['devices']
    if MODE == 'probe-error':
        emit('error', error='no accelerator', error_class='configuration')
        sys.exit(0)
    print(json.dumps(result), flush=True)
    if MODE == 'extra': print('{}', flush=True)
elif command == 'resolve':
    world = request['world']['world_size']
    global_batch = request['data']['global_batch_sequences']
    parameters = dict(lr=0.1, micro=global_batch // world)
    if MODE == 'world-recipe': parameters['lr'] *= world
    if MODE == 'invalid-resolved': parameters['micro'] = 'bad'
    emit('resolved' if MODE == 'resolve-kind' else 'resolve', parameters=parameters, batch=dict(micro_batch_sequences=global_batch // world,
         accumulation_steps=2 if MODE == 'batch' else 1, world_size=world))
else:
    assert request['kind'] == 'begin'
    begin = request['begin']
    counts = {}
    if begin['resume'] is not None:
        assert begin['resume']['checkpoint_manifest']
        counts['books'] = begin['resume']['tokens']
    for line in sys.stdin:
        frame = json.loads(line)
        if frame['kind'] == 'evaluation_record':
            assert 'tokens' in frame['record'] and 'loss_mask' in frame['record']
        if frame['kind'] == 'record':
            assert 'text' not in frame['record']
            for corpus, count in frame['record']['consumption'].items():
                counts[corpus] = counts.get(corpus, 0) + count
        if frame['kind'] == 'end': break
    if MODE == 'crash': sys.exit(4)
    if MODE == 'error':
        emit('error', error='fake failure', error_class='numerical-integrity')
        sys.exit(0)
    if MODE == 'unknown':
        emit('surprise')
        sys.exit(0)
    if MODE == 'train-version':
        print('{"kind":"event","protocol":2}')
        sys.exit(0)
    if MODE == 'hang':
        import time
        time.sleep(60)
    tokens = sum(counts.values())
    log = dict(kind='log', step=0, tokens=0, message='x')
    if MODE == 'non-json': print('not json', flush=True)
    if MODE == 'unknown-field': emit('event', event=log, surprise=1)
    if MODE == 'unknown-event-field': emit('event', event=dict(log, surprise=1))
    if MODE == 'trailing': print(json.dumps(dict(kind='event', protocol=1, event=log)) + ' {}', flush=True)
    if MODE == 'schema-1': print(json.dumps(dict(kind='event', schema=1, event=log)), flush=True)
    if MODE == 'error-no-class': emit('error', error='fake failure', error_class='')
    if MODE == 'event-with-error': emit('event', event=log, error='x')
    if MODE == 'unknown-event-kind': emit('event', event=dict(log, kind='bogus'))
    if MODE == 'empty-log': emit('event', event=dict(log, message=''))
    if MODE == 'token-overflow': emit('event', event=dict(kind='progress', step=1, tokens=tokens + 1, loss=1.))
    if MODE == 'nan-loss': emit('event', event=dict(kind='progress', step=1, tokens=tokens, loss=float('nan')))
    if MODE == 'oversize': emit('event', event=dict(log, message='x' * (16 << 20)))
    header = json.dumps({'weight': dict(dtype='F32', shape=[1], data_offsets=[0, 4])}).encode()
    payload = struct.pack('<Q', len(header)) + header + struct.pack('<f', 1.0)
    os.makedirs(begin['paths']['checkpoints'], exist_ok=True)
    with open(os.path.join(begin['paths']['checkpoints'], 'model.safetensors'), 'wb') as f: f.write(payload)
    artifact = dict(path='model.safetensors', sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), role='weights', format='safetensors')
    checkpoint = dict(step=1, tokens=tokens, artifacts=[artifact])
    if MODE == 'checkpoint-step': checkpoint['step'] = 0
    evaluation = dict(step=1, tokens=tokens, nll_sum=2., target_tokens=2, text_bytes=2, loss=1., bits_per_byte=1 / math.log(2))
    emit('event', event=dict(kind='log', step=0, tokens=0, message='hello'))
    emit('event', event=dict(kind='progress', step=1, tokens=tokens, loss=1.))
    emit('event', event=dict(kind='checkpoint', step=1, tokens=tokens, checkpoint=checkpoint))
    if MODE == 'nonmonotonic': emit('event', event=dict(kind='progress', step=0, tokens=0, loss=1.))
    if MODE == 'bad-metrics': evaluation['nll_sum'] = 4.
    if MODE == 'missing-nll': del evaluation['nll_sum']
    if MODE == 'null-nll': evaluation['nll_sum'] = None
    emit('event', event=dict(kind='evaluation', step=1, tokens=tokens, evaluation=evaluation))
    complete = dict(steps=1, consumed_tokens=tokens, artifacts=[artifact], artifact_profile='hf-causal-lm',
                    consumption=[dict(corpus=c, token_targets=n) for c, n in counts.items()], evaluations=[evaluation],
                    artifact_evaluation=evaluation, selected_checkpoint=dict(step=1, tokens=tokens, metric='heldout_loss', value=1.), environment={'fake': '1'})
    if MODE == 'missing-complete': del complete['environment']
    if MODE == 'consumption': complete['consumption'][0]['corpus'] = 'wrong'
    if MODE == 'escape': artifact['path'] = '../escape'
    if MODE == 'selection': complete['selected_checkpoint']['step'] = 0
    if MODE == 'profile': complete['artifact_profile'] = 'other'
    if MODE == 'no-complete': sys.exit(0)
    emit('complete', complete=complete, **({'error_class': 'x'} if MODE == 'complete-with-error' else {}))
    if MODE == 'duplicate-complete': emit('complete', complete=complete)
    if MODE == 'exit-after-complete':
        print('trainer-stderr-marker', file=sys.stderr, flush=True)
        sys.exit(3)
    if MODE == 'after-complete': emit('event', event=dict(kind='log', message='late'))
