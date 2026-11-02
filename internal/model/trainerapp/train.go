// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package trainerapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"

	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/training"
)

type Paths struct {
	Artifacts   string `json:"artifacts"`
	Checkpoints string `json:"checkpoints"`
	Scratch     string `json:"scratch"`
}
type Resume struct {
	Step               int64             `json:"step"`
	Tokens             int64             `json:"tokens"`
	CheckpointManifest json.RawMessage   `json:"checkpoint_manifest"`
	Paths              map[string]string `json:"paths"`
	reference          CheckpointReference
	consumption        map[string]int64
}
type Begin struct {
	// Lifecycle-owned pins and durable checkpoint destination are not trainer input.
	PriorCheckpoints    []CheckpointReference           `json:"-"`
	PriorEvaluations    []Evaluation                    `json:"-"`
	CheckpointIdentity  CheckpointIdentity              `json:"-"`
	CheckpointDirectory string                          `json:"-"`
	Paths               Paths                           `json:"paths"`
	ModelID             string                          `json:"model_id"`
	RunID               string                          `json:"run_id"`
	Stage               string                          `json:"stage"`
	Objective           string                          `json:"objective"`
	Data                training.ExternalDataParameters `json:"data"`
	Architecture        map[string]any                  `json:"architecture"`
	ArchitectureSHA256  string                          `json:"architecture_sha256"`
	Parameters          map[string]any                  `json:"parameters"`
	ParametersSHA256    string                          `json:"parameters_sha256"`
	Batch               Batch                           `json:"batch"`
	Tokenizer           training.TokenizerSpec          `json:"tokenizer"`
	EvaluationSet       training.EvaluationSet          `json:"evaluation_set"`
	Initialization      *training.WorkerInitialization  `json:"initialization"`
	Resume              *Resume                         `json:"resume"`
	Determinism         string                          `json:"determinism"`
	World               applaunch.World                 `json:"world"`
}

// SequenceSource supplies already packed, WALDO-owned sequences positioned at
// the resume boundary. LossMask has one entry per target (len(Tokens)-1).
// Implementations must honor ctx and stop immediately when consume fails.
type SequenceSource func(context.Context, func(training.PreparedSequence) error) error

type EvaluationRecord struct {
	ID        string `json:"id"`
	Tokens    []int  `json:"tokens"`
	LossMask  []bool `json:"loss_mask"`
	TextBytes int64  `json:"text_bytes"`
}
type Artifact struct {
	training.Artifact
	Role   string `json:"role"`
	Format string `json:"format"`
}
type Checkpoint struct {
	Step      int64      `json:"step"`
	Tokens    int64      `json:"tokens"`
	Artifacts []Artifact `json:"artifacts"`
}
type Evaluation struct {
	Step         int64   `json:"step"`
	Tokens       int64   `json:"tokens"`
	NLLSum       float64 `json:"nll_sum"`
	TargetTokens int64   `json:"target_tokens"`
	TextBytes    int64   `json:"text_bytes"`
	Loss         float64 `json:"loss"`
	BitsPerByte  float64 `json:"bits_per_byte"`
}
type Event struct {
	// DurableCheckpoint is available only after the bundle has been verified and committed.
	DurableCheckpoint *CheckpointReference `json:"-"`
	Kind              string               `json:"kind"`
	Message           string               `json:"message,omitempty"`
	Step              int64                `json:"step"`
	Tokens            int64                `json:"tokens"`
	Loss              *float64             `json:"loss,omitempty"`
	Checkpoint        *Checkpoint          `json:"checkpoint,omitempty"`
	Evaluation        *Evaluation          `json:"evaluation,omitempty"`
}

// Telemetry adapts protocol events to existing progress/telemetry sinks. The
// original event still retains external artifact roles and NLL evidence.
func (e Event) Telemetry() training.Event {
	t := training.Event{Kind: e.Kind, Message: e.Message, Step: e.Step, Tokens: e.Tokens, Loss: e.Loss}
	if e.Checkpoint != nil {
		t.Checkpoint = &training.Checkpoint{Step: e.Checkpoint.Step, Tokens: e.Checkpoint.Tokens}
		for _, a := range e.Checkpoint.Artifacts {
			t.Checkpoint.Artifacts = append(t.Checkpoint.Artifacts, a.Artifact)
		}
	}
	if e.Evaluation != nil {
		v := e.Evaluation
		t.Evaluation = &training.Evaluation{Step: v.Step, Tokens: v.Tokens, Metrics: map[string]float64{"heldout_loss": v.Loss, "heldout_nll_sum": v.NLLSum, "heldout_bits_per_byte": v.BitsPerByte}}
	}
	return t
}

type Complete struct {
	ArtifactEvaluation *Evaluation                   `json:"artifact_evaluation,omitempty"`
	Steps              int64                         `json:"steps"`
	ConsumedTokens     int64                         `json:"consumed_tokens"`
	Artifacts          []Artifact                    `json:"artifacts"`
	ArtifactProfile    string                        `json:"artifact_profile"`
	Consumption        []training.CorpusConsumption  `json:"consumption"`
	Evaluations        []Evaluation                  `json:"evaluations"`
	SelectedCheckpoint *training.CheckpointSelection `json:"selected_checkpoint"`
	Environment        map[string]any                `json:"environment"`
}
type outputFrame struct {
	Kind       string    `json:"kind"`
	Protocol   int       `json:"protocol"`
	Event      *Event    `json:"event,omitempty"`
	Complete   *Complete `json:"complete,omitempty"`
	Error      string    `json:"error,omitempty"`
	ErrorClass string    `json:"error_class,omitempty"`
}
type inputFrame struct {
	Kind     string `json:"kind"`
	Protocol int    `json:"protocol"`
	Begin    *Begin `json:"begin,omitempty"`
	Record   any    `json:"record,omitempty"`
}

// Train drains output while streaming input and always reaps the child. Returned
// terminal artifacts have passed file, tensor and selection verification.
// Checkpoints are copied, verified and committed before the sink is called; the
// lifecycle must persist DurableCheckpoint independently. A sink error stops the run.
func (c *Client) Train(ctx context.Context, b Begin, evaluations []EvaluationRecord, records SequenceSource, sink func(Event) error) (Complete, error) {
	var result Complete
	if err := c.validateBegin(b); err != nil {
		return result, err
	}
	if records == nil {
		return result, fmt.Errorf("training requires a sequence source")
	}
	if err := validateEvaluationRecords(b, evaluations); err != nil {
		return result, err
	}
	candidates := map[int64]CheckpointReference{}
	for _, ref := range b.PriorCheckpoints {
		r, err := LoadCheckpoint(b, ref)
		if err != nil {
			return result, integrity(err)
		}
		candidates[r.Step] = ref
	}
	if b.Resume != nil {
		candidates[b.Resume.Step] = b.Resume.reference
	}
	observed := append([]Evaluation(nil), b.PriorEvaluations...)
	for _, e := range observed {
		if err := validateEvaluation(e, b); err != nil {
			return result, integrity(err)
		}
	}
	initial := map[string]int64{}
	startStep := int64(0)
	if b.Resume != nil {
		initial = b.Resume.consumption
		startStep = b.Resume.Step
	}
	ledger, err := newLedger(b.Paths.Scratch, initial)
	if err != nil {
		return result, err
	}
	defer ledger.close()
	if err := ledger.add(startStep, nil); err != nil {
		return result, err
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r := c.request
	r.Command = "train"
	r.Args = nil
	p, err := applaunch.Start(ctx, r)
	if err != nil {
		return result, err
	}
	c.evidence = append(c.evidence, p.Evidence())
	type sentResult struct {
		consumption map[string]int64
		err         error
	}
	done := make(chan sentResult, 1)
	go func() {
		counts, e := send(ctx, p, b, evaluations, records, ledger)
		p.Stdin.Close()
		if e != nil {
			cancel()
		}
		done <- sentResult{counts, e}
	}()
	defer func() { cancel(); p.Stdin.Close(); p.Stdout.Close(); p.Wait(); <-done }()
	s := scanner(p.Stdout)
	terminal := false
	step, tokens := int64(0), int64(0)
	if b.Resume != nil {
		step, tokens = b.Resume.Step, b.Resume.Tokens
	}
	for s.Scan() {
		if terminal {
			return result, integrity(fmt.Errorf("frame after complete"))
		}
		var f outputFrame
		if err := strict(s.Bytes(), &f); err != nil {
			return result, integrity(err)
		}
		if f.Protocol != Protocol {
			return result, integrity(fmt.Errorf("unsupported trainer protocol %d", f.Protocol))
		}
		switch f.Kind {
		case "error":
			return result, decodeError(s.Bytes())
		case "event":
			if f.Event == nil || f.Complete != nil || f.Error != "" || f.ErrorClass != "" {
				return result, integrity(fmt.Errorf("invalid event frame"))
			}
			e := *f.Event
			if err := validateEvent(e, b, step, tokens); err != nil {
				return result, integrity(err)
			}
			if e.Kind == "progress" || e.Kind == "checkpoint" {
				step, tokens = e.Step, e.Tokens
			}
			if e.Checkpoint != nil {
				counts, err := ledger.at(e.Step)
				if err != nil {
					return result, err
				}
				ref, err := c.commitCheckpoint(b, *e.Checkpoint, counts)
				if err != nil {
					return result, integrity(err)
				}
				e.DurableCheckpoint = &ref
				candidates[e.Step] = ref
			}
			if e.Evaluation != nil {
				observed = append(observed, *e.Evaluation)
			}
			if sink != nil {
				if err := sink(e); err != nil {
					return result, err
				}
			}
		case "complete":
			if f.Complete == nil || f.Event != nil || f.Error != "" || f.ErrorClass != "" {
				return result, integrity(fmt.Errorf("invalid complete frame"))
			}
			result = *f.Complete
			if err := validateComplete(result, b, c.request.Source.Manifest.ArtifactProfile, step, tokens); err != nil {
				return result, integrity(err)
			}
			terminal = true
		default:
			return result, fmt.Errorf("unknown trainer frame %q", f.Kind)
		}
	}
	if err := s.Err(); err != nil {
		return result, err
	}
	if !terminal {
		cancel()
		sent := <-done
		done <- sent
		return result, errors.Join(fmt.Errorf("trainer missing complete"), sent.err, p.Wait())
	}
	sent := <-done
	done <- sent
	if err := errors.Join(sent.err, p.Wait()); err != nil {
		return result, err
	}
	// Resume manifests supply historical consumption; this transport checks the
	// current attempt's sent targets against the reported cumulative delta total.
	total := int64(0)
	for _, n := range sent.consumption {
		total += n
	}
	baseline := int64(0)
	if b.Resume != nil {
		baseline = b.Resume.Tokens
	}
	if result.ConsumedTokens-baseline != total {
		return result, integrity(fmt.Errorf("consumption differs from tokens sent"))
	}
	reported := map[string]int64{}
	for _, v := range result.Consumption {
		reported[v.Corpus] = v.TokenTargets
	}
	for corpus, n := range initial {
		sent.consumption[corpus] += n
	}
	if digest(reported) != digest(sent.consumption) {
		return result, integrity(fmt.Errorf("per-corpus consumption differs from tokens sent and checkpoint"))
	}

	return result, verifyCompletionFiles(result, b, candidates, observed)
}
func (c *Client) validateBegin(b Begin) error {
	if err := b.CheckpointIdentity.validate(); err != nil {
		return err
	}
	if !filepath.IsAbs(b.CheckpointDirectory) {
		return fmt.Errorf("checkpoint directory must be absolute")
	}
	if b.ModelID == "" || b.RunID == "" || b.Stage == "" {
		return fmt.Errorf("begin requires model_id, run_id and stage")
	}
	if b.World != c.request.World {
		return fmt.Errorf("begin world differs from launch")
	}
	supported := false
	for _, o := range c.probe.Capabilities.Objectives {
		if o == b.Objective {
			supported = true
		}
	}
	if !supported || !slices.Contains(c.request.Source.Manifest.Objectives, b.Objective) {
		return fmt.Errorf("unsupported begin objective")
	}
	for _, p := range []string{b.Paths.Artifacts, b.Paths.Checkpoints, b.Paths.Scratch} {
		if !filepath.IsAbs(p) {
			return fmt.Errorf("begin paths must be absolute")
		}
	}
	if b.Paths.Scratch != c.request.Scratch {
		return fmt.Errorf("begin scratch differs from launch")
	}
	if b.Determinism != "required" && b.Determinism != "best-effort" && b.Determinism != "off" {
		return fmt.Errorf("invalid determinism")
	}
	if b.Data.Steps < 1 || b.Data.SequenceLength < 1 || b.Data.GlobalBatchSequences < 1 || b.Data.Steps > math.MaxInt64/b.Data.GlobalBatchSequences || b.Data.SequenceLength == math.MaxInt64 || b.Data.Steps*b.Data.GlobalBatchSequences > math.MaxInt64/b.Data.SequenceLength {
		return fmt.Errorf("invalid data budget")
	}
	if b.Tokenizer.VocabularySize < 1 {
		return fmt.Errorf("missing tokenizer facts")
	}
	if b.Architecture == nil || b.Parameters == nil || digest(b.Architecture) != b.ArchitectureSHA256 || digest(b.Parameters) != b.ParametersSHA256 {
		return fmt.Errorf("begin architecture/parameter hash mismatch")
	}
	a, err := normalize(b.Architecture)
	if err != nil {
		return err
	}
	if err = c.architecture.Validate(a); err != nil {
		return err
	}
	v, err := normalize(b.Parameters)
	if err != nil {
		return err
	}
	if err = c.parameters.Validate(v); err != nil {
		return err
	}
	if err := b.Batch.validate(b.Data.GlobalBatchSequences, int64(b.World.WorldSize)); err != nil {
		return err
	}
	if b.Resume != nil {
		verified, err := LoadCheckpoint(b, b.Resume.reference)
		if err != nil {
			return err
		}
		if !sameResume(b.Resume, verified) {
			return fmt.Errorf("resume differs from verified checkpoint")
		}
		_, execution, err := c.parameters.Split(b.Parameters)
		if err != nil {
			return err
		}
		var m CheckpointManifest
		if err := json.Unmarshal(verified.CheckpointManifest, &m); err != nil {
			return err
		}
		if digest(execution) != digest(m.Execution) {
			return fmt.Errorf("checkpoint execution parameters mismatch")
		}
		b.Resume.consumption = verified.consumption
	}

	return nil
}
func validateTokens(tokens []int, mask []bool, vocab int) (int64, error) {
	if len(tokens) < 2 || len(mask) != len(tokens)-1 {
		return 0, fmt.Errorf("token sequence requires tokens and one loss mask per target")
	}
	for _, t := range tokens {
		if t < 0 || t >= vocab {
			return 0, fmt.Errorf("token outside vocabulary")
		}
	}
	n := int64(0)
	for _, m := range mask {
		if m {
			n++
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("sequence has no supervised targets")
	}
	return n, nil
}
func validateEvaluationRecords(b Begin, records []EvaluationRecord) error {
	n, bytes := int64(0), int64(0)
	ids := map[string]bool{}
	for _, r := range records {
		if r.ID == "" || ids[r.ID] || r.TextBytes < 0 {
			return fmt.Errorf("invalid evaluation record identity/bytes")
		}
		ids[r.ID] = true
		count, err := validateTokens(r.Tokens, r.LossMask, b.Tokenizer.VocabularySize)
		if err != nil {
			return err
		}
		n += count
		bytes += r.TextBytes
	}
	if int64(len(records)) != b.EvaluationSet.Records || n != b.EvaluationSet.TokenTargets || bytes != b.EvaluationSet.TextBytes {
		return fmt.Errorf("evaluation records differ from pins")
	}
	return nil
}
func send(ctx context.Context, p *applaunch.Process, b Begin, evaluations []EvaluationRecord, records SequenceSource, ledger *consumptionLedger) (map[string]int64, error) {
	counts := map[string]int64{}
	if err := write(p.Stdin, inputFrame{Kind: "begin", Protocol: Protocol, Begin: &b}); err != nil {
		return counts, err
	}
	for _, r := range evaluations {
		if err := write(p.Stdin, inputFrame{Kind: "evaluation_record", Protocol: Protocol, Record: r}); err != nil {
			return counts, err
		}
	}
	ordinal := int64(0)
	if b.Resume != nil {
		ordinal = b.Resume.Step * b.Data.GlobalBatchSequences
	}
	limit := b.Data.Steps * b.Data.GlobalBatchSequences
	stepCounts := map[string]int64{}
	err := records(ctx, func(r training.PreparedSequence) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if ordinal >= limit || r.Ordinal != ordinal || int64(len(r.Tokens)) != b.Data.SequenceLength+1 {
			return fmt.Errorf("sequence ordinal/length exceeds or differs from data plan")
		}
		n, err := validateTokens(r.Tokens, r.LossMask, b.Tokenizer.VocabularySize)
		if err != nil {
			return err
		}
		total := int64(0)
		for corpus, count := range r.Consumption {
			if corpus == "" || count <= 0 || count > n-total {
				return fmt.Errorf("invalid sequence corpus consumption")
			}
			total += count
			counts[corpus] += count
			stepCounts[corpus] += count
		}
		if total != n {
			return fmt.Errorf("sequence consumption differs from supervised targets")
		}
		if (ordinal+1)%b.Data.GlobalBatchSequences == 0 {
			if err := ledger.add((ordinal+1)/b.Data.GlobalBatchSequences, stepCounts); err != nil {
				return err
			}
			stepCounts = map[string]int64{}
		}
		if err = write(p.Stdin, inputFrame{Kind: "record", Protocol: Protocol, Record: r}); err != nil {
			return err
		}
		ordinal++
		return nil
	})
	if err != nil {
		return counts, err
	}
	if ordinal != limit {
		return counts, fmt.Errorf("sequence stream ended before exact step budget")
	}
	return counts, write(p.Stdin, inputFrame{Kind: "end", Protocol: Protocol})
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func validateEvaluation(e Evaluation, b Begin) error {
	if !finite(e.NLLSum) || !finite(e.Loss) || !finite(e.BitsPerByte) {
		return &training.WorkerError{Class: training.WorkerErrorNumericalIntegrity, Message: "non-finite evaluation"}
	}
	if e.Step < 0 || e.Step > b.Data.Steps || e.Tokens < 0 || e.TargetTokens <= 0 || e.TextBytes <= 0 || e.TargetTokens != b.EvaluationSet.TokenTargets || e.TextBytes != b.EvaluationSet.TextBytes || !finite(e.NLLSum) || e.NLLSum < 0 || !finite(e.Loss) || !finite(e.BitsPerByte) {
		return fmt.Errorf("invalid evaluation counts/metrics")
	}
	loss := e.NLLSum / float64(e.TargetTokens)
	bits := e.NLLSum / (float64(e.TextBytes) * math.Ln2)
	if math.Abs(e.Loss-loss) > 1e-8*math.Max(1, loss) || math.Abs(e.BitsPerByte-bits) > 1e-8*math.Max(1, bits) {
		return fmt.Errorf("evaluation metrics inconsistent with NLL sum")
	}
	return nil
}
func validateArtifacts(artifacts []Artifact) error {
	if len(artifacts) == 0 {
		return fmt.Errorf("missing artifacts")
	}
	seen := map[string]bool{}
	for _, a := range artifacts {
		if a.Path == "" || filepath.IsAbs(a.Path) || !filepath.IsLocal(a.Path) || filepath.Clean(a.Path) != a.Path || seen[a.Path] || a.Bytes < 0 || len(a.SHA256) != 64 || a.Role == "" || a.Format == "" {
			return fmt.Errorf("invalid artifact declaration")
		}
		for _, ch := range a.SHA256 {
			if !(ch >= '0' && ch <= '9' || ch >= 'a' && ch <= 'f') {
				return fmt.Errorf("invalid artifact digest")
			}
		}
		seen[a.Path] = true
	}
	return nil
}
func validateEvent(e Event, b Begin, step, tokens int64) error {
	if e.Loss != nil && !finite(*e.Loss) {
		return &training.WorkerError{Class: training.WorkerErrorNumericalIntegrity, Message: "non-finite training loss"}
	}
	if e.Step < 0 || e.Step > b.Data.Steps || e.Tokens < 0 || e.Tokens > e.Step*b.Data.GlobalBatchSequences*b.Data.SequenceLength || e.Loss != nil && !finite(*e.Loss) {
		return fmt.Errorf("invalid event counters/loss")
	}
	switch e.Kind {
	case "progress":
		if e.Step < step || e.Tokens < tokens || e.Loss == nil || e.Checkpoint != nil || e.Evaluation != nil {
			return fmt.Errorf("invalid/nonmonotonic progress")
		}
	case "log":
		if e.Message == "" || e.Checkpoint != nil || e.Evaluation != nil {
			return fmt.Errorf("invalid log event")
		}
	case "checkpoint":
		if e.Checkpoint == nil || e.Evaluation != nil || e.Step < step || e.Tokens < tokens || e.Checkpoint.Step != e.Step || e.Checkpoint.Tokens != e.Tokens {
			return fmt.Errorf("invalid checkpoint event")
		}
		return validateArtifacts(e.Checkpoint.Artifacts)
	case "evaluation":
		if e.Evaluation == nil || e.Checkpoint != nil || e.Evaluation.Step != e.Step || e.Evaluation.Tokens != e.Tokens {
			return fmt.Errorf("invalid evaluation event")
		}
		return validateEvaluation(*e.Evaluation, b)
	default:
		return fmt.Errorf("unknown event kind %q", e.Kind)
	}
	return nil
}
func validateComplete(c Complete, b Begin, profile string, step, tokens int64) error {
	if c.Steps != b.Data.Steps || c.Steps < step || c.ConsumedTokens < tokens || c.ArtifactProfile != profile || c.Environment == nil || c.Consumption == nil || c.Evaluations == nil || c.SelectedCheckpoint == nil {
		return fmt.Errorf("complete missing fields or inconsistent counters/profile")
	}
	if err := validateArtifacts(c.Artifacts); err != nil {
		return err
	}
	total := int64(0)
	seen := map[string]bool{}
	for _, v := range c.Consumption {
		if v.Corpus == "" || seen[v.Corpus] || v.TokenTargets < 0 || v.TokenTargets > c.ConsumedTokens-total {
			return fmt.Errorf("invalid completion consumption")
		}
		seen[v.Corpus] = true
		total += v.TokenTargets
	}
	if total != c.ConsumedTokens {
		return fmt.Errorf("completion consumption total mismatch")
	}
	for _, e := range c.Evaluations {
		if err := validateEvaluation(e, b); err != nil {
			return err
		}
	}
	s := c.SelectedCheckpoint
	if s.Step < 0 || s.Step > c.Steps || s.Tokens < 0 || s.Tokens > c.ConsumedTokens || !finite(s.Value) {
		return fmt.Errorf("invalid selected checkpoint")
	}
	if b.EvaluationSet.Records > 0 {
		if len(c.Evaluations) == 0 || s.Metric != "heldout_loss" {
			return fmt.Errorf("missing held-out checkpoint selection")
		}
		best := c.Evaluations[0]
		for _, e := range c.Evaluations[1:] {
			if e.Loss < best.Loss || e.Loss == best.Loss && e.Step < best.Step {
				best = e
			}
		}
		if s.Step != best.Step || s.Tokens != best.Tokens || s.Value != best.Loss {
			return fmt.Errorf("selected checkpoint is not best held-out candidate")
		}
	} else if s.Step != c.Steps {
		return fmt.Errorf("without evaluation select terminal checkpoint")
	}
	return nil
}

// SchemaDocument encodes exact probe bytes without json.Marshal's whitespace
// compaction. The lifecycle writer atomically persists this beside PLAN.json.
// Tokenizer bytes are supplied by tokenizerapp's validated probe document.
func (c *Client) SchemaDocument(tokenizer json.RawMessage) ([]byte, error) {
	if _, err := application.CompileSchema(tokenizer, true); err != nil {
		return nil, err
	}
	s := c.Schemas()
	out := []byte("{\"kind\":\"waldo-trainer-schemas\",\"schema\":1,\"architecture\":")
	out = append(out, s.Architecture...)
	out = append(out, []byte(",\"parameters\":")...)
	out = append(out, s.Parameters...)
	out = append(out, []byte(",\"tokenizer\":")...)
	out = append(out, tokenizer...)
	out = append(out, '}', '\n')
	return out, nil
}
