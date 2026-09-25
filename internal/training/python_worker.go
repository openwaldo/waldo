// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package training

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/openwaldo/waldo/internal/corpus"
)

var workerExitDrain = 5 * time.Second

// configureGracefulCancellation gives launchers such as torchrun an
// opportunity to forward termination to their worker processes. Killing the
// launcher immediately can orphan ranks because they use their own process
// groups.
func configureGracefulCancellation(command *exec.Cmd) {
	command.Cancel = func() error {
		if command.Process == nil {
			return os.ErrProcessDone
		}
		pid := command.Process.Pid
		err := command.Process.Signal(syscall.SIGTERM)
		if errors.Is(err, os.ErrProcessDone) || errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		if err != nil {
			return err
		}
		deadline := time.Now().Add(workerExitDrain)
		for time.Now().Before(deadline) {
			if err := syscall.Kill(pid, 0); errors.Is(err, syscall.ESRCH) {
				return nil
			}
			time.Sleep(25 * time.Millisecond)
		}
		if err := syscall.Kill(-pid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
			return err
		}
		return nil
	}
	command.WaitDelay = workerExitDrain
}

func awaitProcessExit(command *exec.Cmd) error {
	state, err := command.Process.Wait()
	if err != nil {
		return err
	}
	if !state.Success() {
		return &exec.ExitError{ProcessState: state}
	}
	return nil
}

func terminateWorkerGroup(command *exec.Cmd) string {
	if command.Process == nil || command.SysProcAttr == nil || !command.SysProcAttr.Setpgid {
		return "left running"
	}
	if err := syscall.Kill(-command.Process.Pid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		return fmt.Sprintf("could not be killed: %v", err)
	}
	return "killed"
}

func stopWorkerCommand(command *exec.Cmd) string {
	if command.Cancel != nil {
		if err := command.Cancel(); err == nil || errors.Is(err, os.ErrProcessDone) || errors.Is(err, syscall.ESRCH) {
			return "terminated"
		}
	}
	return terminateWorkerGroup(command)
}

func writeStoppedByWorkerExit(err error) bool {
	return errors.Is(err, os.ErrClosed) || errors.Is(err, io.ErrClosedPipe) || errors.Is(err, syscall.EPIPE)
}

// runPythonWorker owns the common schema-1 process and stream lifecycle used
// by framework adapters. Framework code receives canonical records only; it
// never reads WALDO indexes, Parquet files, or lifecycle state.
func runPythonWorker(ctx context.Context, label, python, program string, request Request, extraArguments ...string) (Observation, error) {
	if python == "" {
		return Observation{}, fmt.Errorf("%s Python runtime is required", label)
	}
	arguments := []string{"-c", program, request.ArtifactDirectory, request.ArtifactPrefix}
	arguments = append(arguments, extraArguments...)
	return runWorkerCommand(ctx, label, exec.CommandContext(ctx, python, arguments...), request)
}

func runWorkerCommand(ctx context.Context, label string, command *exec.Cmd, request Request) (Observation, error) {
	if request.Records == nil {
		return Observation{}, fmt.Errorf("%s backend received no canonical record stream", label)
	}
	request.Tokenizer = defaultedTokenizer(request.Tokenizer)
	if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
		return Observation{}, fmt.Errorf("create %s artifact directory: %w", label, err)
	}
	stdin, err := command.StdinPipe()
	if err != nil {
		return Observation{}, err
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		return Observation{}, err
	}
	var stderr cappedBuffer
	command.Stderr = &stderr
	command.WaitDelay = workerExitDrain
	if command.SysProcAttr == nil {
		command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}
	var skipped cappedBuffer
	if err := command.Start(); err != nil {
		return Observation{}, fmt.Errorf("start %s worker: %w", label, err)
	}

	type workerResult struct {
		observation Observation
		err         error
	}
	result := make(chan workerResult, 1)
	stopRecords := make(chan struct{})
	var stopRecordsOnce sync.Once
	if request.Resume != nil && request.Resume.Step >= request.Parameters.Steps {
		stopRecordsOnce.Do(func() { close(stopRecords) })
	}
	go func() {
		var observation Observation
		completed := false
		err := ReadWorkerOutputWithSkipped(stdout, &skipped, func(frame WorkerOutputFrame) error {
			switch frame.Kind {
			case "event":
				if frame.Event.Step >= request.Parameters.Steps {
					stopRecordsOnce.Do(func() { close(stopRecords) })
				}
				if request.Report != nil {
					request.Report(*frame.Event)
				}
			case "complete":
				if completed {
					return fmt.Errorf("%s worker returned more than one completion", label)
				}
				completed = true
				observation = *frame.Observation
			case "error":
				return &WorkerError{Message: frame.Error, Class: frame.ErrorClass}
			}
			return nil
		})
		if err == nil && !completed {
			err = fmt.Errorf("%s worker exited without a completion observation", label)
		}
		if err != nil && command.Process != nil {
			stopWorkerCommand(command)
		}
		result <- workerResult{observation: observation, err: err}
	}()

	records, evaluationRecords, tokenizerErr := tokenizedWorkerSources(request)
	if tokenizerErr != nil {
		_ = stdin.Close()
		_ = command.Process.Kill()
		_ = command.Wait()
		return Observation{}, tokenizerErr
	}
	if request.Parallelism.DataPlane == DataPlaneNodeLocal {
		records = prefetchRecords(records)
	}
	exited := make(chan error, 1)
	go func() {
		exitErr := awaitProcessExit(command)
		_ = stdin.Close()
		exited <- exitErr
	}()
	defer func() { go func() { _ = command.Wait() }() }()
	writeErr := writeWorkerInputUntil(ctx, stdin, workerBeginFromRequest(request), records, evaluationRecords, stopRecords)
	closeErr := stdin.Close()
	if errors.Is(closeErr, os.ErrClosed) {
		closeErr = nil
	}
	if writeStoppedByWorkerExit(writeErr) {
		writeErr = nil
	}
	var worker workerResult
	var waitErr error
	abandoned := ""
	select {
	case worker = <-result:
		waitErr = <-exited
	case waitErr = <-exited:
		select {
		case worker = <-result:
		case <-time.After(workerExitDrain):
			abandoned = terminateWorkerGroup(command)
			_ = stdout.Close()
			worker = <-result
		}
	}
	if abandoned != "" {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Observation{}, ctxErr
		}
		return Observation{}, fmt.Errorf("%s worker exited while leftover rank processes held its output stream open (%s)%s%s", label, abandoned, workerSkipped(skipped.String()), workerStderr(stderr.String()))
	}
	if writeErr != nil && !writeStoppedByWorkerExit(writeErr) {
		return Observation{}, fmt.Errorf("stream records to %s worker: %w%s%s", label, writeErr, workerSkipped(skipped.String()), workerStderr(stderr.String()))
	}
	if worker.err != nil {
		// CommandContext closes the worker pipes when cancellation kills the
		// process. The output reader consequently observes EOF before a complete
		// frame; preserve cancellation so the model lifecycle records an
		// interrupted, resumable run instead of a failed worker.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Observation{}, ctxErr
		}
		return Observation{}, fmt.Errorf("%s worker: %w%s%s", label, worker.err, workerSkipped(skipped.String()), workerStderr(stderr.String()))
	}
	if closeErr != nil && waitErr == nil {
		return Observation{}, fmt.Errorf("close %s worker input: %w", label, closeErr)
	}
	if waitErr != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Observation{}, ctxErr
		}
		return Observation{}, fmt.Errorf("%s worker exited: %w%s", label, waitErr, workerStderr(stderr.String()))
	}
	return worker.observation, nil
}

func defaultedTokenizer(spec TokenizerSpec) TokenizerSpec {
	if spec.Name == "" {
		return TokenizerSpec{Name: "byte", Revision: ByteTokenizerRevision, VocabularySize: 259, PadID: 0, BOSID: 1, EOSID: 2}
	}
	return spec
}

func workerBeginFromRequest(request Request) WorkerBegin {
	tokenizer := defaultedTokenizer(request.Tokenizer)
	begin := WorkerBegin{
		RunID: request.RunID, Stage: request.Stage, Objective: request.Objective,
		ArchitectureSHA256: request.ArchitectureSHA256, Architecture: request.Architecture,
		Parameters: request.Parameters, Parallelism: request.Parallelism,
		EvaluationSet: request.EvaluationSet, Tokenizer: tokenizer, DataNodeRank: request.DataNodeRank,
		PreparedCacheDirectory: request.PreparedCacheDirectory, PreparedCacheMaxBytes: request.PreparedCacheMaxBytes,
	}
	identity, _ := json.Marshal(struct {
		Architecture string                `json:"architecture_sha256"`
		Objective    string                `json:"objective"`
		Conversation ConversationTransform `json:"conversation"`
		Tokenizer    TokenizerSpec         `json:"tokenizer"`
		Corpus       corpus.BOM            `json:"corpus_bom"`
		Parameters   ResolvedParameters    `json:"parameters"`
		Parallelism  Parallelism           `json:"parallelism"`
		Evaluation   EvaluationSet         `json:"evaluation_set"`
	}{request.ArchitectureSHA256, request.Objective, request.Conversation, tokenizer, request.BOM, request.Parameters, request.Parallelism, request.EvaluationSet})
	sum := sha256.Sum256(identity)
	begin.PreparedIdentity = hex.EncodeToString(sum[:])
	if request.Initialization != nil {
		begin.Initialization = &WorkerInitialization{
			SourceType:  request.Initialization.SourceType,
			SourceID:    request.Initialization.SourceID,
			SourceRunID: request.Initialization.SourceRunID,
			Artifact:    request.Initialization.Artifact,
			Path:        request.Initialization.Path,
		}
	}
	if request.Resume != nil {
		begin.Resume = &WorkerResume{
			Step: request.Resume.Step, Tokens: request.Resume.Tokens,
			Checkpoint:  request.Resume.Checkpoint,
			Checkpoints: append([]Checkpoint(nil), request.Resume.Checkpoints...),
			Evaluations: append([]Evaluation(nil), request.Resume.Evaluations...),
			Paths:       append([]string(nil), request.Resume.Paths...),
		}
	}
	return begin
}

func tokenizedWorkerSources(request Request) (RecordSource, RecordSource, error) {
	records, evaluationRecords := request.Records, request.EvaluationRecords
	tokenizer := defaultedTokenizer(request.Tokenizer)
	if request.PreTokenize || tokenizer.Name != "byte" || request.Objective == "assistant-response-modeling" || request.Conversation.Template != "" {
		_, codec, err := ResolveTokenizer(tokenizer.Name, tokenizer.Revision, uint64(tokenizer.VocabularySize))
		if err != nil {
			return nil, nil, err
		}
		records = tokenizedRecordSource{source: records, codec: codec, objective: request.Objective, conversation: request.Conversation}
		if evaluationRecords != nil {
			evaluationRecords = tokenizedRecordSource{source: evaluationRecords, codec: codec, objective: request.Objective, conversation: request.Conversation}
		}
	}
	return records, evaluationRecords, nil
}

func runWorkerStreamJoin(ctx context.Context, label string, command *exec.Cmd, request Request) (Observation, error) {
	if err := os.MkdirAll(request.ArtifactDirectory, 0o755); err != nil {
		return Observation{}, fmt.Errorf("create %s artifact directory: %w", label, err)
	}
	var output cappedBuffer
	var stdin io.WriteCloser
	var err error
	if request.Records != nil {
		request.Tokenizer = defaultedTokenizer(request.Tokenizer)
		stdin, err = command.StdinPipe()
		if err != nil {
			return Observation{}, err
		}
	}
	command.Stdout = &output
	command.Stderr = &output
	command.WaitDelay = workerExitDrain
	if command.SysProcAttr == nil {
		command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}
	if err := command.Start(); err != nil {
		return Observation{}, fmt.Errorf("start %s secondary node: %w", label, err)
	}
	writeResult := make(chan error, 1)
	if stdin != nil {
		go func() {
			records, evaluations, tokenizeErr := tokenizedWorkerSources(request)
			if tokenizeErr != nil {
				_ = stdin.Close()
				writeResult <- tokenizeErr
				return
			}
			writeErr := WriteWorkerInput(ctx, stdin, workerBeginFromRequest(request), records, evaluations)
			closeErr := stdin.Close()
			if writeErr == nil && !errors.Is(closeErr, os.ErrClosed) {
				writeErr = closeErr
			}
			writeResult <- writeErr
		}()
	}
	defer func() { go func() { _ = command.Wait() }() }()
	waitErr := awaitProcessExit(command)
	if stdin != nil {
		if writeErr := <-writeResult; writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
			terminateWorkerGroup(command)
			return Observation{}, fmt.Errorf("stream node-local records to %s secondary node: %w%s", label, writeErr, workerStderr(output.String()))
		}
	}
	if waitErr != nil {
		terminateWorkerGroup(command)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Observation{}, ctxErr
		}
		return Observation{}, fmt.Errorf("%s secondary node exited: %w%s", label, waitErr, workerStderr(output.String()))
	}
	return Observation{}, nil
}
