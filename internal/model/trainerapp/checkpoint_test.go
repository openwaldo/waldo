// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/training"
)

func TestCheckpointValidation(t *testing.T) {
	for _, mode := range []string{"valid", "file", "manifest", "run", "trainer", "tokenizer", "environment", "world", "execution", "evaluation", "stage", "unverified", "symlink", "missing", "consumption", "position"} {
		t.Run(mode, func(t *testing.T) {
			ctx, req := setup(t, "ok")
			c, err := Open(ctx, req, nil)
			if err != nil {
				t.Fatal(err)
			}
			b := begin(t, c)
			var ref CheckpointReference
			_, err = c.Train(ctx, b, evals(), source, func(e Event) error {
				if e.DurableCheckpoint != nil {
					ref = *e.DurableCheckpoint
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(b.CheckpointDirectory, ref.Path)
			file := filepath.Join(filepath.Dir(path), "model.safetensors")
			switch mode {
			case "file":
				err = os.WriteFile(file, []byte("corrupt checkpoint"), 0600)
			case "manifest":
				err = os.WriteFile(path, []byte(`{}`), 0600)
			case "run":
				b.CheckpointIdentity.RunBOMSHA256 = digest("different")
			case "trainer":
				b.CheckpointIdentity.TrainerSHA256 = digest("different")
			case "tokenizer":
				b.CheckpointIdentity.TokenizerSHA256 = digest("different")
			case "environment":
				b.CheckpointIdentity.EnvironmentSHA256 = digest("different")
			case "world":
				b.World.WorldSize = 2
			case "execution":
				b.Parameters["micro"] = 1
				b.ParametersSHA256 = digest(b.Parameters)
			case "evaluation":
				b.EvaluationSet.SHA256 = "different"
			case "stage":
				b.Stage = "different"
			case "unverified":
				ref.SHA256 = ""
			case "symlink":
				outside := filepath.Join(t.TempDir(), "outside")
				raw, e := os.ReadFile(file)
				if e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(outside, raw, 0600); e != nil {
					t.Fatal(e)
				}
				if e = os.Remove(file); e != nil {
					t.Fatal(e)
				}
				err = os.Symlink(outside, file)
			case "missing":
				err = os.Remove(file)
			case "consumption", "position":
				raw, e := os.ReadFile(path)
				if e != nil {
					t.Fatal(e)
				}
				var m CheckpointManifest
				if e = json.Unmarshal(raw, &m); e != nil {
					t.Fatal(e)
				}
				if mode == "consumption" {
					m.Consumption["books"]++
				} else {
					m.Step = b.Data.Steps + 1
				}
				raw, e = json.Marshal(m)
				if e != nil {
					t.Fatal(e)
				}
				err = os.WriteFile(path, raw, 0600)
				ref.SHA256 = application.Digest(raw)
			}
			if err != nil {
				t.Fatal(err)
			}
			resume, err := LoadCheckpoint(b, ref)
			if mode == "valid" {
				if err != nil || resume.Step != 1 || resume.Tokens != 4 || resume.consumption["books"] != 4 {
					t.Fatalf("%+v %v", resume, err)
				}
				return
			}
			if err == nil {
				t.Fatal("invalid checkpoint accepted")
			}
		})
	}
}

func TestResumeReverifiedBeforeLaunch(t *testing.T) {
	ctx, req := setup(t, "ok")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := begin(t, c)
	var ref CheckpointReference
	_, err = c.Train(ctx, b, evals(), source, func(e Event) error {
		if e.DurableCheckpoint != nil {
			ref = *e.DurableCheckpoint
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	b.Resume, err = LoadCheckpoint(b, ref)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(b.Resume.Paths["model.safetensors"], []byte("corrupted"), 0600); err != nil {
		t.Fatal(err)
	}
	evidence := len(c.Evidence())
	if _, err = c.Train(ctx, b, evals(), source, nil); err == nil || len(c.Evidence()) != evidence {
		t.Fatal("launched with corrupt checkpoint", err)
	}
}

func conformanceFactory(t *testing.T, mode string) ConformanceFactory {
	t.Helper()
	return func(ctx context.Context, name string, alternate bool) (ConformanceAttempt, error) {
		base, err := os.ReadFile("testdata/trainer.py")
		if err != nil {
			return ConformanceAttempt{}, err
		}
		train, err := os.ReadFile("testdata/resumable.py")
		if err != nil {
			return ConformanceAttempt{}, err
		}
		code := strings.Replace(string(base), "MODE = 'ok'", "MODE = '"+mode+"'", 1)
		code = strings.Replace(code, "parameters = dict(lr=0.1, micro=global_batch // world)", "parameters = dict(lr=0.1, micro=request['parameters'].get('micro', global_batch // world))", 1)
		code = strings.Replace(code, "micro_batch_sequences=global_batch // world,", "micro_batch_sequences=parameters['micro'],", 1)
		code = strings.Replace(code, "accumulation_steps=2 if MODE == 'batch' else 1", "accumulation_steps=global_batch // (parameters['micro'] * world)", 1)
		code = strings.Split(code, "\nelse:\n")[0] + "\nelse:\n" + "    " + strings.ReplaceAll(strings.TrimSpace(string(train)), "\n", "\n    ") + "\n"
		req := fixture(t, code)
		c, err := Open(ctx, req, nil)
		if err != nil {
			return ConformanceAttempt{}, err
		}
		b := begin(t, c)
		b.RunID = name
		b.Data.Steps = 4
		b.Data.GlobalBatchSequences = 4
		b.Data.Seed = 42
		b.Determinism = "required"
		b.EvaluationSet = training.EvaluationSet{}
		b.Paths.Artifacts = filepath.Join(req.Scratch, "artifacts")
		b.Paths.Checkpoints = filepath.Join(req.Scratch, "checkpoints")
		micro := 4
		if alternate {
			micro = 2
		}
		resolved, err := c.Resolve(ctx, b.Architecture, map[string]any{"micro": micro}, b.Data, single())
		if err != nil {
			return ConformanceAttempt{}, err
		}
		b.Parameters = resolved.Parameters
		b.ParametersSHA256 = resolved.ParametersSHA256
		b.Batch = resolved.Batch
		return ConformanceAttempt{Client: c, Begin: b, SourceAt: func(start int64) SequenceSource {
			return func(ctx context.Context, consume func(training.PreparedSequence) error) error {
				for i := start; i < 16; i++ {
					r := training.PreparedSequence{Ordinal: i, Tokens: []int{1, 2, 3}, LossMask: []bool{true, true}, Consumption: map[string]int64{"books": 2}}
					if i%4 < 2 {
						r.Tokens = []int{1, 7, 3}
						r.LossMask = []bool{true, false}
						r.Consumption = map[string]int64{"papers": 1}
					}
					if err := consume(r); err != nil {
						return err
					}
				}
				return nil
			}
		}}, nil
	}
}
func TestConformance(t *testing.T) {
	for _, mode := range []string{"conformance", "conformance-bad-resume", "conformance-bad-normalization"} {
		t.Run(mode, func(t *testing.T) {
			report, err := CheckConformance(context.Background(), conformanceFactory(t, mode), 1e-12, 1e-12)
			if mode == "conformance" {
				if err != nil {
					t.Fatal(err)
				}
				if !report.ResumeBitIdentical || report.CheckpointStep != 1 || report.ComparedLosses != 4 || report.MaximumLossDelta != 0 {
					t.Fatalf("%+v", report)
				}
				t.Logf("CPU fake trainer: %+v", report)
			} else {
				want := "not bit-identical"
				if mode == "conformance-bad-normalization" {
					want = "decomposition loss differs"
				}
				if err == nil || !strings.Contains(err.Error(), want) {
					t.Fatalf("expected %s, got %v", want, err)
				}
			}
		})
	}
}

func TestCheckpointRejectedBeforeSink(t *testing.T) {
	for _, mode := range []string{"hash", "counts"} {
		t.Run(mode, func(t *testing.T) {
			code, err := os.ReadFile("testdata/trainer.py")
			if err != nil {
				t.Fatal(err)
			}
			script := string(code)
			if mode == "hash" {
				script = strings.Replace(script, "sha256=hashlib.sha256(payload).hexdigest()", "sha256='a' * 64", 1)
			} else {
				script = strings.Replace(script, "tokens = sum(counts.values())", "tokens = sum(counts.values()) - 1", 1)
			}
			ctx := context.Background()
			c, err := Open(ctx, fixture(t, script), nil)
			if err != nil {
				t.Fatal(err)
			}
			accepted := false
			_, err = c.Train(ctx, begin(t, c), evals(), source, func(e Event) error {
				if e.DurableCheckpoint != nil {
					accepted = true
				}
				return nil
			})
			if err == nil || accepted {
				t.Fatalf("accepted invalid checkpoint: %v", err)
			}
		})
	}
}

func TestResumedCorpusReconciliation(t *testing.T) {
	ctx, req := setup(t, "consumption")
	c, err := Open(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := begin(t, c)
	var ref CheckpointReference
	// The bad completion still leaves a verified checkpoint from its earlier event.
	_, err = c.Train(ctx, b, evals(), source, func(e Event) error {
		if e.DurableCheckpoint != nil {
			ref = *e.DurableCheckpoint
		}
		return nil
	})
	if err == nil || ref.Path == "" {
		t.Fatalf("fixture: %v", err)
	}
	b.Resume, err = LoadCheckpoint(b, ref)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Train(ctx, b, evals(), func(context.Context, func(training.PreparedSequence) error) error { return nil }, nil)
	if err == nil || !strings.Contains(err.Error(), "per-corpus consumption") {
		t.Fatalf("accepted forged historical corpus: %v", err)
	}
}

func TestCheckpointSchemaFixture(t *testing.T) {
	raw, err := os.ReadFile("testdata/checkpoint-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var m CheckpointManifest
	if err = strict(raw, &m); err != nil {
		t.Fatal(err)
	}
	if m.Kind != "waldo-trainer-checkpoint" || m.Schema != 1 || m.Step != 1 || m.Tokens != 4 || m.Consumption["books"] != 4 || m.Files[0].Role != "resume-state" {
		t.Fatalf("schema-1 fixture changed: %+v", m)
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var before, after any
	if err = json.Unmarshal(raw, &before); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(encoded, &after); err != nil {
		t.Fatal(err)
	}
	if digest(before) != digest(after) {
		t.Fatal("schema-1 fields lost or added on roundtrip")
	}
	for _, field := range []string{"world", "execution", "consumption", "identity", "files", "parameters_sha256"} {
		var obj map[string]json.RawMessage
		if err = json.Unmarshal(raw, &obj); err != nil {
			t.Fatal(err)
		}
		delete(obj, field)
		incomplete, e := json.Marshal(obj)
		if e != nil {
			t.Fatal(e)
		}
		if e = strict(incomplete, &m); e == nil {
			t.Fatalf("accepted missing %s", field)
		}
	}
}
