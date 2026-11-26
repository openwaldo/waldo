// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package join

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// Every record pins its plan, so editing the run file starts the run afresh.

// SigningNamespace scopes record signatures to waldo model join.
const SigningNamespace = "waldo-join"

const (
	SubmissionKind = "waldo-join-submission"
	MergeKind      = "waldo-join-merge"
	ClaimKind      = "waldo-join-claim"
	RecordSchema   = 1
)

// Submission is one identity's trained unit for a round.
type Submission struct {
	Kind        string            `json:"kind"`
	Schema      int               `json:"schema"`
	Run         string            `json:"run"`
	Round       int               `json:"round"`
	Stage       string            `json:"stage"`
	Unit        string            `json:"unit"`
	Identity    string            `json:"identity"`
	Base        *string           `json:"base"`
	BaseSHA256  *string           `json:"base_sha256"`
	Model       string            `json:"model"`
	ModelSHA256 string            `json:"model_sha256"`
	Files       map[string]string `json:"files"`
	Tokens      int64             `json:"tokens"`
	URL         string            `json:"url"`
	CreatedAt   string            `json:"created_at"`
	Plan        string            `json:"plan_sha256"`
}

// Claim reserves a slot (<unit>-<n> or close-<n>) before the work; one that expires may be retaken.
type Claim struct {
	Kind      string `json:"kind"`
	Schema    int    `json:"schema"`
	Run       string `json:"run"`
	Round     int    `json:"round"`
	Stage     string `json:"stage"`
	Slot      string `json:"slot"`
	Attempt   int    `json:"attempt"`
	Identity  string `json:"identity"`
	CreatedAt string `json:"created_at"`
	Plan      string `json:"plan_sha256"`
}

func claimFile(slot string, attempt int) string {
	if attempt == 1 {
		return slot + ".yaml"
	}
	return fmt.Sprintf("%s.%d.yaml", slot, attempt)
}

// MergeRecord is one closer's close of a round.
type MergeRecord struct {
	Kind        string            `json:"kind"`
	Schema      int               `json:"schema"`
	Run         string            `json:"run"`
	Round       int               `json:"round"`
	Stage       string            `json:"stage"`
	Identity    string            `json:"identity"`
	Method      string            `json:"method"`
	Base        *string           `json:"base"`
	BaseSHA256  *string           `json:"base_sha256"`
	Accepted    []AcceptedUnit    `json:"accepted"`
	Dropped     []DroppedUnit     `json:"dropped"`
	Rejected    []DroppedUnit     `json:"rejected,omitempty"` // replicas set aside in units that were still judged
	Gate        *GateResult       `json:"gate,omitempty"`     // on a stage's gate rounds
	Evidence    []UnitEvidence    `json:"evidence,omitempty"`
	Model       string            `json:"model"`
	ModelSHA256 string            `json:"model_sha256"`
	Files       map[string]string `json:"files"`
	URL         string            `json:"url"`
	ClosedAt    string            `json:"closed_at"`
	Plan        string            `json:"plan_sha256"`
}

// GateResult is a closer's gate check; whether it passed is part of the decision.
type GateResult struct {
	After  string   `json:"after"`
	Passed bool     `json:"passed"`
	Value  *float64 `json:"heldout_loss,omitempty"`
	Report string   `json:"report,omitempty"` // the waldo model gate report, published with the merge
}

type AcceptedUnit struct {
	Unit        string `json:"unit"`
	Identity    string `json:"identity"`
	Model       string `json:"model"`
	ModelSHA256 string `json:"model_sha256"`
	Tokens      int64  `json:"tokens"`
}

// UnitEvidence is the round's start loss and each replica's held-out loss on one unit.
type UnitEvidence struct {
	Unit                string            `json:"unit"`
	StartLoss           float64           `json:"start_loss"`
	EvaluationSetSHA256 string            `json:"evaluation_set_sha256"`
	CorpusBOMSHA256     string            `json:"corpus_bom_sha256,omitempty"` // the unit's corpus selection, as every replica's run BOM must record it
	Replicas            []ReplicaEvidence `json:"replicas"`
}

type ReplicaEvidence struct {
	Identity    string  `json:"identity"`
	ModelSHA256 string  `json:"model_sha256"`
	HeldoutLoss float64 `json:"heldout_loss"`
}

type DroppedUnit struct {
	Unit       string   `json:"unit"`
	Reason     string   `json:"reason"`
	Identities []string `json:"identities,omitempty"`
}

// RoundRecord links a merge's weights to its run, plan, and accepted submissions' signed records.
type RoundRecord struct {
	Kind       string            `json:"kind"`
	Schema     int               `json:"schema"`
	Subject    string            `json:"subject"`
	Run        string            `json:"run"`
	Round      int               `json:"round"`
	Stage      string            `json:"stage"`
	PlanSHA256 string            `json:"plan_sha256"`
	BaseSHA256 *string           `json:"base_sha256"`
	Method     string            `json:"method"`
	Accepted   []RoundSubmission `json:"accepted"`
}

type RoundSubmission struct {
	Unit         string `json:"unit"`
	Identity     string `json:"identity"`
	ModelSHA256  string `json:"model_sha256"`
	Record       string `json:"record"` // its path in the runs repository
	RecordSHA256 string `json:"record_sha256"`
}

func writeAllowedSigners(path string, identities map[string]Identity) error {
	names := make([]string, 0, len(identities))
	for name := range identities {
		names = append(names, name)
	}
	sort.Strings(names)
	var lines strings.Builder
	for _, name := range names {
		fmt.Fprintf(&lines, "%s namespaces=%q %s\n", name, SigningNamespace, identities[name].Key)
	}
	return os.WriteFile(path, []byte(lines.String()), 0o644)
}

func writeSigned(path string, record any, key string) error {
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return err
	}
	_ = os.Remove(path + ".sig")
	if output, err := exec.Command("ssh-keygen", "-Y", "sign", "-f", key, "-n", SigningNamespace, path).CombinedOutput(); err != nil {
		return fmt.Errorf("sign %s: %w: %s", path, err, bytes.TrimSpace(output))
	}
	return nil
}

// readSigned decodes path if a plan identity's key signed it; ok is false otherwise.
func readSigned(path, allowedSigners string, identities map[string]Identity, record any, identityOf func() string) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}
	if err := json.Unmarshal(data, record); err != nil {
		return false, nil
	}
	identity := identityOf()
	if _, ok := identities[identity]; !ok {
		return false, nil
	}
	if _, err := os.Stat(path + ".sig"); err != nil {
		return false, nil
	}
	command := exec.Command("ssh-keygen", "-Y", "verify", "-f", allowedSigners, "-I", identity, "-n", SigningNamespace, "-s", path+".sig")
	command.Stdin = bytes.NewReader(data)
	return command.Run() == nil, nil
}
