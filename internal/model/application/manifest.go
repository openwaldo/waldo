// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// Package application owns declarative external trainer and tokenizer sources.
// It never executes application code.
package application

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

type Source struct {
	Repository string `json:"repository" yaml:"repository"`
	Commit     string `json:"commit" yaml:"commit"`
	Path       string `json:"path" yaml:"path"`
}

var commitPattern = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)

func (s Source) Validate() error {
	if strings.TrimSpace(s.Repository) == "" {
		return fmt.Errorf("source.repository is required")
	}
	if !commitPattern.MatchString(s.Commit) {
		return fmt.Errorf("source.commit must be a full 40-hex commit SHA")
	}
	if !relativePath(s.Path) {
		return fmt.Errorf("source.path must be a clean relative directory (use . for the root)")
	}
	return nil
}
func relativePath(s string) bool {
	return s != "" && !strings.ContainsAny(s, "\\:\x00") && !path.IsAbs(s) && path.Clean(s) == s && s != ".." && !strings.HasPrefix(s, "../")
}

type Entrypoint struct {
	Module string `json:"module,omitempty" yaml:"module,omitempty"`
	Script string `json:"script,omitempty" yaml:"script,omitempty"`
}
type Environment struct {
	Kind   string `json:"kind" yaml:"kind"`
	Lock   string `json:"lock" yaml:"lock"`
	Python string `json:"python" yaml:"python"`
}
type Commands struct {
	Probe   string `json:"probe" yaml:"probe"`
	Resolve string `json:"resolve,omitempty" yaml:"resolve,omitempty"`
	Train   string `json:"train,omitempty" yaml:"train,omitempty"`
	Serve   string `json:"serve,omitempty" yaml:"serve,omitempty"`
}
type Manifest struct {
	Kind            string                 `json:"kind" yaml:"kind"`
	Schema          int                    `json:"schema" yaml:"schema"`
	Name            string                 `json:"name" yaml:"name"`
	Protocol        int                    `json:"protocol" yaml:"protocol"`
	Entrypoint      Entrypoint             `json:"entrypoint" yaml:"entrypoint"`
	Commands        Commands               `json:"commands" yaml:"commands"`
	LaunchProfiles  []string               `json:"launch_profiles,omitempty" yaml:"launch_profiles,omitempty"`
	Objectives      []string               `json:"objectives,omitempty" yaml:"objectives,omitempty"`
	ArtifactProfile string                 `json:"artifact_profile,omitempty" yaml:"artifact_profile,omitempty"`
	Environments    map[string]Environment `json:"environments" yaml:"environments"`
}

// DecodeStrict accepts exactly one YAML or JSON document and rejects unknown
// fields, including fields nested in application-owned declarations.
func DecodeStrict(data []byte, target any) error {
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("{")) {
		d := json.NewDecoder(bytes.NewReader(data))
		d.DisallowUnknownFields()
		d.UseNumber()
		if err := d.Decode(target); err != nil {
			return err
		}
		var extra any
		if err := d.Decode(&extra); err != io.EOF {
			return fmt.Errorf("expected exactly one JSON document")
		}
	} else {
		d := yaml.NewDecoder(bytes.NewReader(data))
		d.KnownFields(true)
		if err := d.Decode(target); err != nil {
			return err
		}
		var extra any
		if err := d.Decode(&extra); err != io.EOF {
			return fmt.Errorf("expected exactly one YAML document")
		}
	}
	return nil
}
func Digest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func ParseManifest(data []byte, kind string) (Manifest, string, error) {
	var m Manifest
	if err := DecodeStrict(data, &m); err != nil {
		return m, "", err
	}
	if err := m.Validate(kind); err != nil {
		return m, "", err
	}
	return m, Digest(data), nil
}

var modulePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$`)
var pythonPattern = regexp.MustCompile(`^3\.[0-9]+\.[0-9]+$`)

func (m Manifest) Validate(kind string) error {
	if (kind != "waldo-trainer" && kind != "waldo-tokenizer") || m.Kind != kind || m.Schema != 1 || m.Protocol != 1 {
		return fmt.Errorf("unsupported manifest kind %q schema %d protocol %d (expected %s, schema 1, protocol 1)", m.Kind, m.Schema, m.Protocol, kind)
	}
	if strings.TrimSpace(m.Name) == "" {
		return fmt.Errorf("manifest name is required")
	}
	if (m.Entrypoint.Module == "") == (m.Entrypoint.Script == "") {
		return fmt.Errorf("entrypoint requires exactly one of module or script")
	}
	if m.Entrypoint.Module != "" && !modulePattern.MatchString(m.Entrypoint.Module) {
		return fmt.Errorf("invalid entrypoint module")
	}
	if m.Entrypoint.Script != "" && (!relativePath(m.Entrypoint.Script) || m.Entrypoint.Script == ".") {
		return fmt.Errorf("invalid entrypoint script")
	}
	validCommand := func(s string) bool {
		return s != "" && !strings.HasPrefix(s, "-") && !strings.ContainsAny(s, " \t\r\n\x00")
	}
	if !validCommand(m.Commands.Probe) {
		return fmt.Errorf("commands.probe must be a single command argument")
	}
	if kind == "waldo-trainer" {
		if !validCommand(m.Commands.Resolve) || !validCommand(m.Commands.Train) || m.Commands.Serve != "" {
			return fmt.Errorf("trainer commands must declare probe, resolve and train only")
		}
		if len(m.LaunchProfiles) == 0 || len(m.Objectives) == 0 || m.ArtifactProfile == "" {
			return fmt.Errorf("trainer launch_profiles, objectives and artifact_profile are required")
		}
		for _, v := range m.LaunchProfiles {
			if v != "single-process" && v != "distributed" {
				return fmt.Errorf("unsupported launch profile %q", v)
			}
		}
		for _, v := range m.Objectives {
			if v != "causal-language-modeling" && v != "assistant-response-modeling" {
				return fmt.Errorf("unsupported objective %q", v)
			}
		}
	} else if !validCommand(m.Commands.Serve) || m.Commands.Resolve != "" || m.Commands.Train != "" || len(m.LaunchProfiles) > 0 || len(m.Objectives) > 0 || m.ArtifactProfile != "" {
		return fmt.Errorf("tokenizer manifest must declare probe and serve only, without trainer capabilities")
	}
	if len(m.Environments) == 0 {
		return fmt.Errorf("named environments are required")
	}
	for name, e := range m.Environments {
		if name == "" || e.Kind != "python-lock" || !relativePath(e.Lock) || e.Lock == "." || !pythonPattern.MatchString(e.Python) {
			return fmt.Errorf("environment %q requires kind python-lock, a relative lock path and an exact Python 3.x.y version", name)
		}
	}
	return nil
}
