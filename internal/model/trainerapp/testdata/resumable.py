# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
# Inserted as the train branch by the Go fixture; standard-library only.
import pickle
import random
import struct

begin = request['begin']
rng = random.Random(begin['data']['seed'])
state = dict(step=0, tokens=0, counts={}, weight=0.0, momentum=0.0)
if begin['resume'] is not None:
    with open(next(path for name, path in begin['resume']['paths'].items() if name.endswith('/runtime.pkl')), 'rb') as f:
        state = pickle.load(f)
    rng.setstate(state.pop('rng'))
    assert state['step'] == begin['resume']['step']
    if MODE == 'conformance-bad-resume': state['momentum'] = 0.0
batch = []

def save(directory, path, payload, role, fmt):
    os.makedirs(os.path.dirname(os.path.join(directory, path)), exist_ok=True)
    with open(os.path.join(directory, path), 'wb') as f: f.write(payload)
    return dict(path=path, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), role=role, format=fmt)

for line in sys.stdin:
    frame = json.loads(line)
    if frame['kind'] == 'end': break
    if frame['kind'] != 'record': continue
    record = frame['record']
    assert record['ordinal'] == state['step'] * begin['data']['global_batch_sequences'] + len(batch)
    batch.append(record)
    if len(batch) != begin['data']['global_batch_sequences']: continue
    micro = begin['batch']['micro_batch_sequences']
    sums = []
    for offset in range(0, len(batch), micro):
        records = batch[offset:offset + micro]
        targets = [token for r in records for token, mask in zip(r['tokens'][1:], r['loss_mask']) if mask]
        sums.append((sum(targets), len(targets)))
    loss = sum(n for n, _ in sums) / sum(n for _, n in sums)
    if MODE == 'conformance-bad-normalization': loss = sum(s / n for s, n in sums) / len(sums)
    state['momentum'] = state['momentum'] * 0.5 + loss + rng.random()
    state['weight'] -= state['momentum'] / (state['step'] + 1)
    for r in batch:
        for corpus, count in r['consumption'].items():
            state['counts'][corpus] = state['counts'].get(corpus, 0) + count
            state['tokens'] += count
    state['step'] += 1
    emit('event', event=dict(kind='progress', step=state['step'], tokens=state['tokens'], loss=loss))
    state['rng'] = rng.getstate()
    artifact = save(begin['paths']['checkpoints'], f"step-{state['step']}/runtime.pkl", pickle.dumps(state), 'resume-state', 'pickle')
    emit('event', event=dict(kind='checkpoint', step=state['step'], tokens=state['tokens'], checkpoint=dict(step=state['step'], tokens=state['tokens'], artifacts=[artifact])))
    batch = []
# One real scalar safetensors tensor; fixed header means file equality proves
# terminal tensor equality without depending on a tensor framework.
header = json.dumps({'weight': dict(dtype='F64', shape=[1], data_offsets=[0, 8])}).encode()
header += b' ' * (-len(header) % 8)
payload = struct.pack('<Q', len(header)) + header + struct.pack('<d', state['weight'])
artifact = save(begin['paths']['artifacts'], 'model.safetensors', payload, 'weights', 'safetensors')
emit('complete', complete=dict(steps=state['step'], consumed_tokens=state['tokens'], artifacts=[artifact], artifact_profile='hf-causal-lm', consumption=[dict(corpus=c, token_targets=n) for c, n in state['counts'].items()], evaluations=[], selected_checkpoint=dict(step=state['step'], tokens=state['tokens'], metric='', value=0), environment={'fake': '1'}))
