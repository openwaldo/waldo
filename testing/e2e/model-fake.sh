#!/bin/sh
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
temporary_base=${TMPDIR:-/tmp}
work=$(mktemp -d "$temporary_base/waldo-model-e2e.XXXXXX")

cleanup() {
  if [ "${WALDO_E2E_KEEP:-0}" = "1" ]; then
    echo "preserved model E2E workspace: $work"
    return
  fi
  case "$work" in
    "$temporary_base"/waldo-model-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing to remove unexpected workspace: $work" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

binary="$work/waldo"
index_root="$work/waldo-index"
lookaside="$work/lookaside"
cache="$work/cache"
scratch="$work/scratch"
staging="$work/staging"
models="$work/models"
model_export="$work/model-export"
input="$work/training"
raw_input="$input/raw"
compose="$work/model.yaml"
disclosure="$work/eu-gpai.json"
provider="$work/provider.json"
export WALDO_CONFIG="$work/config.json"

echo "testing: complete fake-model lifecycle"
(cd "$repo_root" && GOCACHE="$work/go-cache" go build -o "$binary" ./cmd/waldo)
mkdir -p "$raw_input"
cat > "$raw_input/records.jsonl" <<'EOF'
{"text":"Contact test@example.org; this assessed row must be excluded.","metadata":{"namespace":0}}
{"text":"alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta","metadata":{"namespace":0}}
{"text":"Repeated navigation footer line.\nRepeated navigation footer line.\nRepeated navigation footer line.\nRepeated navigation footer line.\nRepeated navigation footer line.\nRepeated navigation footer line.\nRepeated navigation footer line.\nRepeated navigation footer line.","metadata":{"namespace":0}}
{"text":"A preserved training record with enough ordinary prose for this fixture.","metadata":{"namespace":0}}
{"text":"A separate record reserved for deterministic evaluation.","metadata":{"namespace":0}}
{"text":"Auxiliary discussion that must be excluded by main-content classification.","metadata":{"namespace":1}}
EOF
file_bytes=$(wc -c < "$raw_input/records.jsonl" | tr -d ' ')
if command -v sha256sum >/dev/null 2>&1; then
  file_sha=$(sha256sum "$raw_input/records.jsonl" | awk '{print $1}')
  tree_sha=$(printf '%s\t%s\t%s\n' "$file_sha" "$file_bytes" records.jsonl | sha256sum | awk '{print $1}')
else
  file_sha=$(shasum -a 256 "$raw_input/records.jsonl" | awk '{print $1}')
  tree_sha=$(printf '%s\t%s\t%s\n' "$file_sha" "$file_bytes" records.jsonl | shasum -a 256 | awk '{print $1}')
fi
cat > "$input/manifest.json" <<EOF
{
  "kind":"waldo-source-directory",
  "schema":1,
  "retrieved_at":"2026-09-13T00:00:00Z",
  "corpus":{"id":"model-e2e","title":"Model-E2E-Corpus","description":"Disposable fake-model input."},
  "sources":[{
    "id":"model-e2e","path":"","license":"CC0-1.0",
    "source":{"name":"model-e2e","version":"fixture-1","url":"https://example.invalid/model-e2e","category":"public-dataset","license_evidence":{"declaration":"CC0-1.0"}},
    "input":{"format":"jsonl","type":"record-map","main_content":{"metadata.namespace":0},"fields":{"text":["text"]}},
    "artifacts":[]
  }],
  "fetcher":{"name":"model-e2e"},
  "raw":{"path":"raw","file_count":1,"byte_count":$file_bytes,"tree_sha256":"$tree_sha"}
}
EOF

"$binary" index init "$index_root"
"$binary" config set lookaside "file://$lookaside"
"$binary" config set lookaside.cache "$cache"
"$binary" config set lookaside.cache.max-size 64MiB
"$binary" config set lookaside.scratch "$scratch"
"$binary" config set ingest.staging "$staging"
"$binary" config set model.root "$models"
"$binary" config set model.backend fake
"$binary" config set index "$index_root"
cat > "$provider" <<EOF
{
  "kind": "waldo-disclosure-provider",
  "schema": 1,
  "provider": {
    "name": "OpenWALDO E2E",
    "address": "1 Test Way, Test City",
    "contact": "test@example.invalid"
  },
  "code_of_practice_status": "not-assessed",
  "copyright_policy_url": "https://example.invalid/copyright"
}
EOF
"$binary" config set disclosure.provider "$provider"

destination="$index_root/core/e2e/model-corpus"
"$binary" index ingest "$input" "$destination"

contribution=""
for candidate in "$staging"/*/contribution; do
  [ -d "$candidate" ] || continue
  [ -z "$contribution" ] || { echo "multiple contribution overlays found" >&2; exit 1; }
  contribution=$candidate
done
[ -n "$contribution" ] || { echo "contribution overlay not found" >&2; exit 1; }
cp -R "$contribution"/. "$index_root"/

"$binary" index audit "$destination"
cat > "$compose" <<EOF
kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 128
  vocabulary_size: 256
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: float32
  tokenizer:
    name: byte
    revision: sha256:model-e2e
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    filter:
      main_content: true
      exclude:
        repetitive_content: true
        boilerplate_content: true
    corpora:
      - core/e2e/model-corpus
    parameters:
      steps: 2
      batch_size: 1
      sequence_length: 64
      learning_rate: 0.001
      seed: 7
EOF

forecast_output=$("$binary" model forecast "$compose")
printf '%s\n' "$forecast_output"
printf '%s\n' "$forecast_output" | grep -Eq '^HOST:[[:space:]]+[^/]+/[^[:space:]]+$'
printf '%s\n' "$forecast_output" | grep -Eq '^BACKEND:[[:space:]]+fake@'
printf '%s\n' "$forecast_output" | grep -Eq '^READY:[[:space:]]+no$'
printf '%s\n' "$forecast_output" | grep -q '^REASON:'
if printf '%s\n' "$forecast_output" | grep -Eq 'HOST COMPARISON|GPUS.*MFR|FIT|~|unified'; then
  echo "forecast contains an unwanted display field" >&2
  exit 1
fi
comparison_output=$("$binary" model forecast "$compose" --compare-hosts)
printf '%s\n' "$comparison_output" | grep -q 'GPUS.*MFR.*ACCELERATOR.*MEMORY/GPU.*APPROX. TIME'
printf '%s\n' "$comparison_output" | grep -q 'Apple.*M4 Max 40-core GPU'
printf '%s\n' "$comparison_output" | grep -q 'NVIDIA.*H100 SXM'
[ ! -e "$models/smoke" ] || { echo "forecast created model state" >&2; exit 1; }
forecast_json=$("$binary" --json model forecast "$compose")
printf '%s\n' "$forecast_json" | grep -Eq '"catalog"[[:space:]]*:[[:space:]]*"openwaldo-training-hardware-'
printf '%s\n' "$forecast_json" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*false'
printf '%s\n' "$forecast_json" | grep -Eq '"reason"[[:space:]]*:'
if printf '%s\n' "$forecast_json" | grep -Eq '"configurations"[[:space:]]*:'; then
  echo "default JSON forecast exposed host comparisons" >&2
  exit 1
fi
comparison_json=$("$binary" --json model forecast "$compose" --compare-hosts)
printf '%s\n' "$comparison_json" | grep -Eq '"approximate_seconds"[[:space:]]*:'

build_output=$("$binary" model train smoke "$compose" --audit)
printf '%s\n' "$build_output"
printf '%s\n' "$build_output" | grep -q 'trained model smoke'

inspect_output=$("$binary" model summary smoke)
printf '%s\n' "$inspect_output"
printf '%s\n' "$inspect_output" | grep -q 'complete'
printf '%s\n' "$inspect_output" | grep -q 'simulated'

json_inspection=$("$binary" --json model summary smoke)
printf '%s\n' "$json_inspection" | grep -Eq '"state"[[:space:]]*:[[:space:]]*"complete"'
printf '%s\n' "$json_inspection" | grep -Eq '"simulated"[[:space:]]*:[[:space:]]*true'
printf '%s\n' "$json_inspection" | grep -Eq '"profile"[[:space:]]*:[[:space:]]*"causal-pretrain-shuffled"'
printf '%s\n' "$json_inspection" | grep -Eq '"packing"[[:space:]]*:[[:space:]]*"continuous-eos-v1"'
printf '%s\n' "$json_inspection" | grep -Eq '"checkpoints"[[:space:]]*:'

for required in PLAN.json MODEL.json MODEL-BOM.json; do
  [ -s "$models/smoke/$required" ] || { echo "missing model record $required" >&2; exit 1; }
done
run_count=$(find "$models/smoke/runs" -type f -name RUN.json -print | wc -l | tr -d ' ')
[ "$run_count" -eq 1 ] || { echo "found $run_count run records, want 1" >&2; exit 1; }
run_bom=$(find "$models/smoke/runs" -type f -name RUN-BOM.json -print -quit)
[ -n "$run_bom" ] || { echo "missing training run BOM" >&2; exit 1; }
grep -Eq '"attestation"[[:space:]]*:' "$run_bom" || { echo "run BOM omitted shard attestation evidence" >&2; exit 1; }
grep -Eq '"status"[[:space:]]*:[[:space:]]*"embedded"' "$run_bom" || { echo "run BOM omitted embedded shard BOM status" >&2; exit 1; }
grep -Eq '"main_content"[[:space:]]*:[[:space:]]*true' "$run_bom" || { echo "run BOM omitted the applied main-content requirement" >&2; exit 1; }
grep -Eq '"repetitive_content"[[:space:]]*:[[:space:]]*true' "$run_bom" || { echo "run BOM omitted the applied repetitive-content exclusion" >&2; exit 1; }
grep -Eq '"boilerplate_content"[[:space:]]*:[[:space:]]*true' "$run_bom" || { echo "run BOM omitted the applied boilerplate-content exclusion" >&2; exit 1; }
artifact_count=$(find "$models/smoke/runs" -type f -name fake-model.json -print | wc -l | tr -d ' ')
[ "$artifact_count" -eq 1 ] || { echo "found $artifact_count fake artifacts, want 1" >&2; exit 1; }
fake_artifact=$(find "$models/smoke/runs" -type f -name fake-model.json -print -quit)
grep -Eq '"training_records"[[:space:]]*:[[:space:]]*2' "$fake_artifact" || { echo "content exclusions did not leave exactly two training records" >&2; exit 1; }
checkpoint_count=$(find "$models/smoke/runs" -type f -name 'step-*.json' -print | wc -l | tr -d ' ')
[ "$checkpoint_count" -eq 1 ] || { echo "found $checkpoint_count fake checkpoints, want 1" >&2; exit 1; }

repeat_output=$("$binary" model train smoke "$compose" --audit)
printf '%s\n' "$repeat_output" | grep -q 'unchanged; verified all 1 stages of compose'
run_count=$(find "$models/smoke/runs" -type f -name RUN.json -print | wc -l | tr -d ' ')
[ "$run_count" -eq 1 ] || { echo "completed corpus was trained again" >&2; exit 1; }

changed_compose="$work/model-changed.yaml"
sed 's/steps: 2/steps: 3/' "$compose" > "$changed_compose"
if "$binary" model train smoke "$changed_compose" >"$work/changed.out" 2>&1; then
  echo "changed compose was incorrectly treated as complete" >&2
  exit 1
fi
grep -q 'completed run BOMs do not match compose' "$work/changed.out" || {
  echo "changed compose did not report its completion mismatch" >&2
  cat "$work/changed.out" >&2
  exit 1
}
run_count=$(find "$models/smoke/runs" -type f -name RUN.json -print | wc -l | tr -d ' ')
[ "$run_count" -eq 1 ] || { echo "changed compose added an unexpected run" >&2; exit 1; }

"$binary" model init manual --preset 10m >/dev/null
"$binary" model train manual core/e2e/model-corpus >/dev/null
"$binary" model summary manual | grep -q 'RUNS:.*1'
"$binary" model list 'm*' | grep -q 'manual'
"$binary" model bom manual | grep -q '"subject": "model"'
export_stderr="$work/model-export.stderr"
"$binary" model export manual "$model_export" --allow-incomplete >/dev/null 2>"$export_stderr"
[ -s "$model_export/BOM.json" ] || { echo "model export missing BOM" >&2; exit 1; }
[ -s "$model_export/EU-BOM.json" ] || { echo "model export missing EU BOM" >&2; exit 1; }
[ ! -e "$model_export/MODEL-BOM.json" ] || { echo "model export exposed internal MODEL-BOM name" >&2; exit 1; }
grep -q 'model export is unsigned' "$export_stderr"
if "$binary" model chat manual >/dev/null 2>&1; then
  echo "fake model unexpectedly supported chat" >&2
  exit 1
fi
"$binary" model rm manual >/dev/null
[ ! -e "$models/manual" ] || { echo "model rm left manual model" >&2; exit 1; }
if "$binary" model bom smoke "$disclosure" --format eu-gpai >/dev/null 2>&1; then
  echo "complete EU GPAI export unexpectedly passed without required facts" >&2
  exit 1
fi
"$binary" model bom smoke "$disclosure" --format eu-gpai --allow-incomplete
grep -q '"kind": "waldo-eu-gpai-training-content"' "$disclosure"
grep -q '"status": "incomplete-draft"' "$disclosure"

if find "$scratch" -type f -print 2>/dev/null | grep . >/dev/null 2>&1; then
  echo "model lifecycle left partial-download scratch files" >&2
  exit 1
fi

echo "E2E fake model passed: ingested, audited, forecasted, initialized, trained, composed, replaced, summarized, listed, exported, removed, and disclosed"
