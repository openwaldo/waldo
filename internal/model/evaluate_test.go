// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

// byteArchitecture uses a tokenizer real backends accept, as evaluation requires.
func byteArchitecture() Architecture {
	architecture := testArchitecture()
	architecture.VocabularySize = 259
	architecture.Tokenizer = Tokenizer{Name: "byte", Revision: training.ByteTokenizerRevision}
	return architecture
}

func trainedWeightsFixture(t *testing.T, root, name string) (Builder, string) {
	t.Helper()
	data := []byte("trained model weights")
	digest := sha256.Sum256(data)
	weights := hex.EncodeToString(digest[:])
	backend := backendFunc(func(_ context.Context, request training.Request) (training.Observation, error) {
		if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
			return training.Observation{}, err
		}
		if err := os.WriteFile(filepath.Join(request.ArtifactDirectory, "model.safetensors"), data, 0o644); err != nil {
			return training.Observation{}, err
		}
		loss := 0.5
		return training.Observation{Steps: 2, ConsumedTokens: 128, FinalLoss: &loss, Evaluations: []training.Evaluation{{Step: 2, Tokens: 128, Metrics: map[string]float64{"heldout_loss": 0.75}}}, Artifacts: []training.Artifact{{Path: "artifacts/model.safetensors", SHA256: weights, Bytes: int64(len(data))}}}, nil
	})
	builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Initialize(name, byteArchitecture()); err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Train(context.Background(), name, preparedFixture(t, testStage("pretrain"))); err != nil {
		t.Fatal(err)
	}
	return builder, weights
}

func TestEvaluateScoresCurrentWeightsWithoutRecordingRun(t *testing.T) {
	root := t.TempDir()
	builder, weights := trainedWeightsFixture(t, root, "scored")
	var request training.Request
	backend := backendFunc(func(_ context.Context, received training.Request) (training.Observation, error) {
		request = received
		return training.Observation{Evaluations: []training.Evaluation{{Metrics: map[string]float64{"heldout_loss": 1.25}}}}, nil
	})
	builder.Resolver = training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})
	result, err := builder.Evaluate(context.Background(), "scored", preparedFixture(t, testStage("pretrain")))
	if err != nil {
		t.Fatal(err)
	}
	if !request.EvaluateOnly || request.Initialization == nil || request.Initialization.Artifact.SHA256 != weights {
		t.Fatalf("backend request = evaluate-only %v, initialization %+v", request.EvaluateOnly, request.Initialization)
	}
	if result.Loss != 1.25 || result.WeightsSHA256 != weights || result.Set.Records == 0 || result.Set != request.EvaluationSet || result.CorpusBOMSHA256 == "" {
		t.Fatalf("evaluation = %+v", result)
	}
	inspection, err := Inspect(root, "scored")
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.Model.Runs) != 1 {
		t.Fatalf("evaluation recorded runs: %+v", inspection.Model.Runs)
	}
	if trained := inspection.RunBOMs[0].EvaluationSet; trained == nil || *trained != result.Set {
		t.Fatalf("evaluated held-out set %+v, training selected %+v", result.Set, trained)
	}
}

func TestEvaluateRefusesWhatItCannotScore(t *testing.T) {
	root := t.TempDir()
	trained, _ := trainedWeightsFixture(t, root, "trained")
	untrained := Builder{Root: root, Resolver: training.FakeResolver()}
	if _, err := untrained.Initialize("untrained", byteArchitecture()); err != nil {
		t.Fatal(err)
	}
	noHeldOut := testStage("pretrain")
	zero := 0.0
	noHeldOut.Parameters.EvaluationFraction = &zero
	trained.Resolver = training.FakeResolver()
	for _, test := range []struct {
		builder Builder
		name    string
		stage   Stage
		want    string
	}{
		{untrained, "untrained", testStage("pretrain"), "no weights to evaluate"},
		{trained, "trained", testStage("pretrain"), "only simulates"},
		{trained, "trained", noHeldOut, "no held-out records"},
	} {
		t.Run(fmt.Sprintf("%s/%s", test.name, test.want), func(t *testing.T) {
			if _, err := test.builder.Evaluate(context.Background(), test.name, preparedFixture(t, test.stage)); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Evaluate error = %v, want %q", err, test.want)
			}
		})
	}
}
