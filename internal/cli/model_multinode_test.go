// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/corpus"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/record"
	"github.com/openwaldo/waldo/internal/shard"
	"github.com/openwaldo/waldo/internal/training"
	"github.com/parquet-go/parquet-go"
)

func TestModelTrainWorkerRejectsInvalidTopology(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	var out, errs bytes.Buffer
	if code := Run([]string{"config", "set", "model.backend", "torchtitan"}, &out, &errs); code != 0 {
		t.Fatalf("seed config: %s", errs.String())
	}
	for _, test := range []struct {
		name string
		args []string
		want string
	}{
		{"single node", []string{"model", "train-worker", "--nodes", "1", "--node-rank", "1", "--rendezvous", "h:1", "--rendezvous-id", "r"}, "greater than 1"},
		{"rank zero", []string{"model", "train-worker", "--nodes", "4", "--node-rank", "0", "--rendezvous", "h:1", "--rendezvous-id", "r"}, "node-rank must be in 1..3"},
		{"rank too high", []string{"model", "train-worker", "--nodes", "4", "--node-rank", "4", "--rendezvous", "h:1", "--rendezvous-id", "r"}, "node-rank must be in 1..3"},
		{"missing rendezvous", []string{"model", "train-worker", "--nodes", "4", "--node-rank", "1", "--rendezvous-id", "r"}, "requires --rendezvous"},
		{"malformed rendezvous", []string{"model", "train-worker", "--nodes", "4", "--node-rank", "1", "--rendezvous", "hostonly", "--rendezvous-id", "r"}, "must be host:port"},
		{"traversal rendezvous id", []string{"model", "train-worker", "--nodes", "4", "--node-rank", "1", "--rendezvous", "h:1", "--rendezvous-id", ".."}, "must start with a letter or digit"},
		{"separator rendezvous id", []string{"model", "train-worker", "--nodes", "4", "--node-rank", "1", "--rendezvous", "h:1", "--rendezvous-id", "a/b"}, "must start with a letter or digit"},
		{"zero nodes", []string{"model", "train-worker", "--nodes", "0", "--node-rank", "1", "--rendezvous", "h:1", "--rendezvous-id", "r"}, "--nodes must be an integer greater than or equal to 1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := Run(test.args, &stdout, &stderr)
			if code == 0 {
				t.Fatalf("expected failure, got success; stdout = %q", stdout.String())
			}
			if !strings.Contains(stderr.String(), test.want) {
				t.Fatalf("stderr = %q, want substring %q", stderr.String(), test.want)
			}
		})
	}
}

func TestConfigNCCLKeysRoundTrip(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	run := func(args ...string) (string, string, int) {
		var stdout, stderr bytes.Buffer
		code := Run(args, &stdout, &stderr)
		return stdout.String(), stderr.String(), code
	}
	if _, stderr, code := run("config", "set", "model.nccl.interface", "roce0"); code != 0 {
		t.Fatalf("set iface: %s", stderr)
	}
	if _, stderr, code := run("config", "set", "model.nccl.hca", "mlx5_0"); code != 0 {
		t.Fatalf("set hca: %s", stderr)
	}
	loaded, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Model.NCCLInterface != "roce0" || loaded.Model.NCCLHCA != "mlx5_0" {
		t.Fatalf("config = %+v", loaded.Model)
	}
	if out, _, code := run("config", "get", "model.nccl.interface"); code != 0 || !strings.Contains(out, "roce0") {
		t.Fatalf("get iface out = %q code = %d", out, code)
	}
	if _, stderr, code := run("config", "unset", "model.nccl.interface"); code != 0 {
		t.Fatalf("unset iface: %s", stderr)
	}
	loaded, err = config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Model.NCCLInterface != "" {
		t.Fatalf("iface not cleared: %+v", loaded.Model)
	}
}

func TestValidateDistributedBatchSize(t *testing.T) {
	if err := validateDistributedBatchSize("stage pretrain", 16, 2, 4); err != nil {
		t.Fatal(err)
	}
	for _, batch := range []int64{2, 10} {
		err := validateDistributedBatchSize("stage pretrain", batch, 1, 4)
		if err == nil || !strings.Contains(err.Error(), "stage pretrain global micro-batch size") {
			t.Fatalf("batch %d error = %v", batch, err)
		}
	}
	if err := validateDistributedBatchSize("stage pretrain", 16, 8, 4); err == nil {
		t.Fatal("accepted a physical micro-batch smaller than the world size")
	}
}

func seedMultiNodeCorpus(t *testing.T) corpus.BOM {
	t.Helper()
	root := t.TempDir()
	text := "harbor lanterns seawall at dusk"
	var parquetData bytes.Buffer
	writer := parquet.NewGenericWriter[shard.Row](&parquetData)
	if _, err := writer.Write([]shard.Row{{
		SHA256: record.TextHash(text), Kind: record.KindPretrain, Text: text,
		Source: "fixture", License: "CC0-1.0", Tokens: 5,
	}}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	content := parquetData.Bytes()
	digestArray := sha256.Sum256(content)
	digest := hex.EncodeToString(digestArray[:])
	source := filepath.Join(root, "source.parquet")
	if err := os.WriteFile(source, content, 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLIFile(t, filepath.Join(root, "index.json"), `{
  "kind": "index", "schema": 1, "path": "",
  "entries": [{"name": "books", "type": "dir"}]
}`)
	writeCLIFile(t, filepath.Join(root, "books", "index.json"), `{
  "kind": "index", "schema": 1, "path": "books",
  "entries": [{"name": "books.json", "type": "manifest"}]
}`)
	writeCLIFile(t, filepath.Join(root, "books", "books.json"), fmt.Sprintf(`{
  "kind": "manifest", "schema": 1, "name": "books", "title": "Books",
  "description": "Small books.", "license": "CC0-1.0",
  "sources": [{"name": "source", "source": "Fixture", "url": "https://example.test", "sha256": %q}],
  "converted_by": {"tool": "test", "version": "1", "profile": "text", "recipe": "test/v1", "tokenizer": "byte"},
  "shards": [{"url": %q, "sha256": %q, "sources": ["source"], "docs": 1, "tokens": 5, "bytes": %d}]
}`, strings.Repeat("a", 64), source, digest, len(content)))
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	if err := config.Save(config.Config{Index: root, Lookaside: config.Lookaside{Scratch: filepath.Join(t.TempDir(), "scratch")}, Model: config.Model{Root: filepath.Join(t.TempDir(), "models"), Backend: "fake"}}); err != nil {
		t.Fatal(err)
	}
	targets, err := resolveIndexArgumentsWithWarning(context.Background(), []string{"books"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := corpus.NewLicensePolicy(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		t.Fatal(err)
	}
	bom, err := corpus.BuildBOM(context.Background(), targets, policy, cache)
	if err != nil {
		t.Fatal(err)
	}
	return bom
}

func multiNodePlanForTest(t *testing.T, bom corpus.BOM, architecture string) model.MultiNodePlan {
	t.Helper()
	parameters, err := training.ResolveParameters(training.Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	cache, err := lookaside.DefaultCache()
	if err != nil {
		t.Fatal(err)
	}
	materialized, err := corpus.Materialize(context.Background(), bom, cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	partition, err := training.NewRecordPartition(verifiedTrainingInputs(materialized, bom), parameters)
	if err != nil {
		t.Fatal(err)
	}
	evaluation := partition.Evaluation
	return model.MultiNodePlan{
		Kind: model.MultiNodePlanKind, Schema: model.MultiNodePlanSchema,
		RunID: "run0001", Stage: "train-0001", StageOrdinal: 1, StageCount: 1, Nodes: 2, Objective: "causal-language-modeling",
		ArchitectureSHA256: strings.Repeat("b", 64), Architecture: json.RawMessage(architecture),
		Parameters: parameters, CorpusBOM: bom, EvaluationSet: &evaluation,
	}
}

func TestPlannedStageRejectsMissingDistributionEvidenceBeforeMaterialization(t *testing.T) {
	bom := seedMultiNodeCorpus(t)
	stage := model.Stage{
		Name: "pretrain", Type: "pre-training", Objective: "causal-language-modeling",
		Parameters: training.Parameters{Steps: 1, BatchSize: 1, SequenceLength: 8, LearningRate: 0.001, DistributionPolicy: corpus.DistributionPolicyDistributable},
	}
	err := reviewPlannedStageDistribution(stage, bom)
	if err == nil || !strings.Contains(err.Error(), "distributable corpus gate") || !strings.Contains(err.Error(), "upstream license evidence") {
		t.Fatalf("distribution review error = %v", err)
	}
}

func TestSecondaryTrainingRequestFromPlan(t *testing.T) {
	bom := seedMultiNodeCorpus(t)
	cache, err := lookaside.DefaultCache()
	if err != nil {
		t.Fatal(err)
	}
	byteArchitecture := `{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`

	t.Run("byte plan round trips", func(t *testing.T) {
		plan := multiNodePlanForTest(t, bom, byteArchitecture)
		request, err := secondaryTrainingRequest(Context{Execution: context.Background()}, plan, t.TempDir(), cache, t.TempDir(), io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		if request.RunID != "run0001" || request.Stage != "train-0001" || request.Records == nil {
			t.Fatalf("request = %+v", request)
		}
		if request.Tokenizer.Name != "byte" || request.Tokenizer.VocabularySize != 259 {
			t.Fatalf("tokenizer = %+v", request.Tokenizer)
		}
	})

	t.Run("evaluation split mismatch fails closed", func(t *testing.T) {
		plan := multiNodePlanForTest(t, bom, byteArchitecture)
		plan.EvaluationSet = &training.EvaluationSet{Selection: "lowest-sha256-v1", SHA256: strings.Repeat("f", 64)}
		_, err := secondaryTrainingRequest(Context{Execution: context.Background()}, plan, t.TempDir(), cache, t.TempDir(), io.Discard)
		if err == nil || !strings.Contains(err.Error(), "does not match the primary's") {
			t.Fatalf("split guard error = %v", err)
		}
	})

	t.Run("cl100k plan resolves subword tokenizer", func(t *testing.T) {
		plan := multiNodePlanForTest(t, bom, `{"family":"decoder-transformer","vocabulary_size":100259,"tokenizer":{"name":"tiktoken/cl100k_base","revision":"tiktoken-cl100k-base"}}`)
		request, err := secondaryTrainingRequest(Context{Execution: context.Background()}, plan, t.TempDir(), cache, t.TempDir(), io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		if request.Tokenizer.Name != "tiktoken/cl100k_base" || request.Tokenizer.VocabularySize != 100259 {
			t.Fatalf("tokenizer = %+v", request.Tokenizer)
		}
	})

	t.Run("resume artifacts are verified and passed to the worker", func(t *testing.T) {
		plan := multiNodePlanForTest(t, bom, byteArchitecture)
		data := []byte("checkpoint state")
		digest := sha256.Sum256(data)
		path := filepath.Join(t.TempDir(), "state.json")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		artifact := training.Artifact{Path: "artifacts/checkpoints/step-00000001/state.json", SHA256: hex.EncodeToString(digest[:]), Bytes: int64(len(data))}
		plan.Resume = &training.ResumePoint{Step: 1, Tokens: 8, Checkpoint: training.Checkpoint{Step: 1, Tokens: 8, Artifacts: []training.Artifact{artifact}}}
		plan.ResumePaths = []string{path}
		request, err := secondaryStreamRequest(plan, t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		if request.Resume == nil || request.Resume.Step != 1 || !reflect.DeepEqual(request.Resume.Paths, []string{path}) {
			t.Fatalf("secondary resume = %+v", request.Resume)
		}
	})

	t.Run("unsupported tokenizer fails closed", func(t *testing.T) {
		plan := multiNodePlanForTest(t, bom, `{"family":"decoder-transformer","vocabulary_size":9,"tokenizer":{"name":"other","revision":"one"}}`)
		_, err := secondaryTrainingRequest(Context{Execution: context.Background()}, plan, t.TempDir(), cache, t.TempDir(), io.Discard)
		if err == nil || !strings.Contains(err.Error(), "resolve primary plan tokenizer") {
			t.Fatalf("tokenizer guard error = %v", err)
		}
	})
}

func TestSecondaryInitializationRejoinsModelRoot(t *testing.T) {
	modelRoot := t.TempDir()
	weights := []byte("tiny-weights")
	digestArray := sha256.Sum256(weights)
	relative := "smoke/runs/0001-train-run0000/artifacts/model.safetensors"
	absolute := filepath.Join(modelRoot, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(absolute, weights, 0o644); err != nil {
		t.Fatal(err)
	}
	artifact := training.Artifact{Path: "artifacts/model.safetensors", SHA256: hex.EncodeToString(digestArray[:]), Bytes: int64(len(weights))}
	base := model.MultiNodePlan{
		Initialization:     &training.Initialization{SourceType: "run", SourceRunID: "run0000", Artifact: artifact},
		InitializationPath: relative,
	}

	t.Run("rejoin resolves and verifies", func(t *testing.T) {
		initialization, err := secondaryInitialization(base, modelRoot)
		if err != nil {
			t.Fatal(err)
		}
		if initialization == nil || initialization.Path != absolute || initialization.SourceRunID != "run0000" {
			t.Fatalf("initialization = %+v", initialization)
		}
	})

	t.Run("missing portable path fails closed", func(t *testing.T) {
		plan := base
		plan.InitializationPath = ""
		if _, err := secondaryInitialization(plan, modelRoot); err == nil || !strings.Contains(err.Error(), "without a portable path") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("digest mismatch fails closed", func(t *testing.T) {
		plan := base
		mismatched := artifact
		mismatched.SHA256 = strings.Repeat("f", 64)
		initialization := *base.Initialization
		initialization.Artifact = mismatched
		plan.Initialization = &initialization
		if _, err := secondaryInitialization(plan, modelRoot); err == nil || !strings.Contains(err.Error(), "verify initialization weights") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("escaping path fails closed", func(t *testing.T) {
		plan := base
		plan.InitializationPath = "../outside/model.safetensors"
		if _, err := secondaryInitialization(plan, modelRoot); err == nil || !strings.Contains(err.Error(), "escapes the model root") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("path without weights fails closed", func(t *testing.T) {
		plan := base
		plan.Initialization = nil
		if _, err := secondaryInitialization(plan, modelRoot); err == nil || !strings.Contains(err.Error(), "without initialization weights") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("no initialization passes through", func(t *testing.T) {
		plan := base
		plan.Initialization = nil
		plan.InitializationPath = ""
		initialization, err := secondaryInitialization(plan, modelRoot)
		if err != nil || initialization != nil {
			t.Fatalf("initialization = %+v, err = %v", initialization, err)
		}
	})
}

func TestRunSecondaryStagesFollowsCompose(t *testing.T) {
	bom := seedMultiNodeCorpus(t)
	cache, err := lookaside.DefaultCache()
	if err != nil {
		t.Fatal(err)
	}
	configuration, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	modelRoot, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		t.Fatal(err)
	}
	byteArchitecture := `{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`
	stagePlan := func(runID string, ordinal, count int) model.MultiNodePlan {
		plan := multiNodePlanForTest(t, bom, byteArchitecture)
		plan.RunID = runID
		plan.StageOrdinal = ordinal
		plan.StageCount = count
		return plan
	}
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, Rendezvous: "primary:29500", RendezvousID: "loop-42"}
	scratch := t.TempDir()

	t.Run("two stages then done", func(t *testing.T) {
		seedPlan(t, modelRoot, "loop-42", stagePlan("runA", 1, 2))
		var runs []string
		runner := func(_ training.Cluster, request training.Request) error {
			runs = append(runs, request.RunID)
			if request.RunID == "runA" {
				seedPlan(t, modelRoot, "loop-42", stagePlan("runB", 2, 2))
			}
			if request.ArtifactDirectory != filepath.Join(scratch, request.RunID) {
				t.Errorf("stage scratch = %q", request.ArtifactDirectory)
			}
			return nil
		}
		var stdout bytes.Buffer
		if err := runSecondaryStages(Context{Execution: context.Background()}, cluster, modelRoot, scratch, cache, time.Minute, runner, &stdout, io.Discard); err != nil {
			t.Fatal(err)
		}
		if len(runs) != 2 || runs[0] != "runA" || runs[1] != "runB" {
			t.Fatalf("runs = %v", runs)
		}
		if !strings.Contains(stdout.String(), "secondary node completed") {
			t.Fatalf("stdout = %q", stdout.String())
		}
	})

	t.Run("single stage stays one shot", func(t *testing.T) {
		seedPlan(t, modelRoot, "loop-43", stagePlan("runC", 1, 1))
		single := cluster
		single.RendezvousID = "loop-43"
		count := 0
		runner := func(training.Cluster, training.Request) error {
			count++
			return nil
		}
		var stdout bytes.Buffer
		if err := runSecondaryStages(Context{Execution: context.Background()}, single, modelRoot, t.TempDir(), cache, time.Minute, runner, &stdout, io.Discard); err != nil {
			t.Fatal(err)
		}
		if count != 1 {
			t.Fatalf("runner calls = %d", count)
		}
	})

	t.Run("stage failure exits immediately", func(t *testing.T) {
		seedPlan(t, modelRoot, "loop-44", stagePlan("runD", 1, 3))
		failing := cluster
		failing.RendezvousID = "loop-44"
		runner := func(training.Cluster, training.Request) error {
			return fmt.Errorf("rendezvous exploded")
		}
		var stdout bytes.Buffer
		err := runSecondaryStages(Context{Execution: context.Background()}, failing, modelRoot, t.TempDir(), cache, time.Minute, runner, &stdout, io.Discard)
		if err == nil || !strings.Contains(err.Error(), "rendezvous exploded") {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestComposeTrainingAcceptsTopologyFlags(t *testing.T) {
	seedMultiNodeCorpus(t)
	composePath := filepath.Join(t.TempDir(), "compose.yaml")
	composeYAML := `kind: waldo-model-compose
schema: 1
architecture:
  family: decoder-transformer
  context_tokens: 128
  vocabulary_size: 259
  hidden_size: 64
  intermediate_size: 192
  layers: 2
  attention_heads: 4
  key_value_heads: 2
  tie_embeddings: true
  parameter_dtype: bfloat16
  tokenizer:
    name: byte
    revision: builtin-byte-schema-1
stages:
  - name: pretrain
    type: pre-training
    objective: causal-language-modeling
    corpora:
      - books
    parameters:
      steps: 1
      batch_size: 1
      sequence_length: 64
      learning_rate: 0.001
      seed: 7
`
	if err := os.WriteFile(composePath, []byte(composeYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	run := func(args ...string) (string, int) {
		var stdout, stderr bytes.Buffer
		code := Run(args, &stdout, &stderr)
		return stderr.String(), code
	}

	t.Run("topology threads into backend check", func(t *testing.T) {
		stderr, code := run("model", "train", "cm1", composePath, "--nodes", "2", "--rendezvous", "primary:29500", "--rendezvous-id", "cm1")
		if code == 0 || !strings.Contains(stderr, "TorchTitan backend") {
			t.Fatalf("code = %d, stderr = %q", code, stderr)
		}
	})

	t.Run("invalid topology fails before compose work", func(t *testing.T) {
		stderr, code := run("model", "train", "cm2", composePath, "--nodes", "2", "--rendezvous", "hostonly", "--rendezvous-id", "cm2")
		if code == 0 || !strings.Contains(stderr, "must be host:port") {
			t.Fatalf("code = %d, stderr = %q", code, stderr)
		}
	})

	t.Run("stage parameters still rejected", func(t *testing.T) {
		stderr, code := run("model", "train", "cm3", composePath, "--batch-size", "4")
		if code == 0 || !strings.Contains(stderr, "--batch-size cannot be used with compose") {
			t.Fatalf("code = %d, stderr = %q", code, stderr)
		}
	})

	t.Run("single node compose still trains", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := Run([]string{"model", "train", "cm4", composePath}, &stdout, &stderr); code != 0 {
			t.Fatalf("single-node compose failed: %s", stderr.String())
		}
	})
}

func TestRunSecondaryStagesRejectsTopologyMismatch(t *testing.T) {
	bom := seedMultiNodeCorpus(t)
	cache, err := lookaside.DefaultCache()
	if err != nil {
		t.Fatal(err)
	}
	configuration, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	modelRoot, err := config.EffectiveModelRoot(configuration)
	if err != nil {
		t.Fatal(err)
	}
	plan := multiNodePlanForTest(t, bom, `{"family":"decoder-transformer","vocabulary_size":259,"tokenizer":{"name":"byte","revision":"builtin-byte-schema-1"}}`)
	plan.Nodes = 4
	seedPlan(t, modelRoot, "mismatch", plan)
	cluster := training.Cluster{Nodes: 2, NodeRank: 1, Rendezvous: "primary:29500", RendezvousID: "mismatch"}
	launched := false
	runner := func(training.Cluster, training.Request) error {
		launched = true
		return nil
	}
	err = runSecondaryStages(Context{Execution: context.Background()}, cluster, modelRoot, t.TempDir(), cache, time.Minute, runner, io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "every node must agree on the topology") {
		t.Fatalf("topology mismatch error = %v", err)
	}
	if launched {
		t.Fatal("worker must not launch when the topology disagrees")
	}
}

func TestSecondaryScratchIsNodeUnique(t *testing.T) {
	t.Setenv("WALDO_CONFIG", filepath.Join(t.TempDir(), "config.json"))
	scratchRoot := t.TempDir()
	seen := map[string]bool{}
	for rank := 1; rank <= 3; rank++ {
		path := filepath.Join(scratchRoot, "train-worker", "rid", fmt.Sprintf("node-%d", rank))
		if seen[path] {
			t.Fatalf("rank %d reuses another node's scratch path %s", rank, path)
		}
		seen[path] = true
		if err := os.MkdirAll(filepath.Join(path, "run0001"), 0o755); err != nil {
			t.Fatalf("rank %d scratch: %v", rank, err)
		}
	}
	if len(seen) != 3 {
		t.Fatalf("expected three distinct scratch paths, got %d", len(seen))
	}
}

func TestSelectedCorpusGroupPreservesCanonicalManifestPath(t *testing.T) {
	selections := []string{"core", "core/peps.yaml"}
	if got := selectedCorpusGroup("core/peps", selections); got != "core/peps.yaml" {
		t.Fatalf("selected corpus group = %q", got)
	}
}
