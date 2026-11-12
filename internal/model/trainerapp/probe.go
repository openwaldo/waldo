// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// Package trainerapp implements external trainer protocol 1. It does not create
// model/run state; its caller persists identity and verifies produced artifacts.
package trainerapp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"

	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/training"
)

const Protocol = 1
const FrameLimit = 16 << 20

type Schemas struct {
	Architecture json.RawMessage `json:"architecture"`
	Parameters   json.RawMessage `json:"parameters"`
}
type Capabilities struct {
	Objectives     []string `json:"objectives"`
	LaunchProfiles []string `json:"launch_profiles"`
	Resume         bool     `json:"resume"`
}
type Device struct {
	Name        string `json:"name"`
	Kind        string `json:"kind"`
	MemoryBytes uint64 `json:"memory_bytes"`
}
type Probe struct {
	Kind         string            `json:"kind"`
	Protocol     int               `json:"protocol"`
	Capabilities Capabilities      `json:"capabilities"`
	Frameworks   map[string]string `json:"frameworks"`
	Devices      []Device          `json:"devices"`
	Driver       string            `json:"driver"`
	CUDA         string            `json:"cuda"`
	Schemas      Schemas           `json:"schemas"`
}
type Client struct {
	request                  applaunch.Request
	probe                    Probe
	architecture, parameters *application.ParameterSchema
	evidence                 []applaunch.Evidence
}

// Open checks capabilities before callers create durable run state. A deadline
// or cancellation must be provided by the caller for all external commands.
func Open(ctx context.Context, request applaunch.Request, objectives []string) (*Client, error) {
	if request.Source.Manifest.Kind != "waldo-trainer" {
		return nil, fmt.Errorf("expected trainer application")
	}
	if request.Profile == "" {
		request.Profile = "single-process"
	}
	if request.World == (applaunch.World{}) {
		request.World = applaunch.World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}
	}
	c := &Client{request: request}
	var p Probe
	if err := c.command(ctx, "probe", map[string]any{"kind": "probe", "protocol": Protocol}, &p); err != nil {
		return nil, err
	}
	if p.Kind != "probe" || p.Protocol != Protocol {
		return nil, fmt.Errorf("probe requires kind probe and protocol 1")
	}
	if !p.Capabilities.Resume || len(p.Frameworks) == 0 || len(p.Devices) == 0 || p.Driver == "" || p.CUDA == "" {
		return nil, fmt.Errorf("probe requires resume support, frameworks, devices, driver and cuda (use unavailable when absent)")
	}
	for _, d := range p.Devices {
		if d.Name == "" || d.Kind == "" || d.MemoryBytes == 0 {
			return nil, fmt.Errorf("probe device requires name, kind and memory_bytes")
		}
	}
	for _, o := range objectives {
		if !slices.Contains(p.Capabilities.Objectives, o) || !slices.Contains(request.Source.Manifest.Objectives, o) {
			return nil, fmt.Errorf("trainer capability mismatch: objective %s", o)
		}
	}
	if !slices.Contains(p.Capabilities.LaunchProfiles, request.Profile) {
		return nil, fmt.Errorf("trainer capability mismatch: launch profile %s", request.Profile)
	}
	var err error
	c.architecture, err = application.CompileSchema(p.Schemas.Architecture, true)
	if err != nil {
		return nil, fmt.Errorf("probe architecture schema: %w", err)
	}
	c.parameters, err = application.CompileSchema(p.Schemas.Parameters, false)
	if err != nil {
		return nil, fmt.Errorf("probe parameters schema: %w", err)
	}
	c.probe = p
	return c, nil
}
func (c *Client) Probe() Probe {
	var p Probe
	b, _ := json.Marshal(c.probe)
	_ = json.Unmarshal(b, &p)
	p.Schemas = c.Schemas()
	return p
}
func (c *Client) Schemas() Schemas {
	return Schemas{c.architecture.Document(), c.parameters.Document()}
}
func (c *Client) Evidence() []applaunch.Evidence {
	b, _ := json.Marshal(c.evidence)
	var e []applaunch.Evidence
	_ = json.Unmarshal(b, &e)
	return e
}

type World struct {
	applaunch.World
	Devices []Device `json:"devices"`
}
type ResolveRequest struct {
	Kind         string                          `json:"kind"`
	Protocol     int                             `json:"protocol"`
	Architecture map[string]any                  `json:"architecture"`
	Parameters   map[string]any                  `json:"parameters"`
	Data         training.ExternalDataParameters `json:"data"`
	World        World                           `json:"world"`
}
type Batch struct {
	MicroBatchSequences int64 `json:"micro_batch_sequences"`
	AccumulationSteps   int64 `json:"accumulation_steps"`
	WorldSize           int64 `json:"world_size"`
}
type resolvedFrame struct {
	Kind       string         `json:"kind"`
	Protocol   int            `json:"protocol"`
	Parameters map[string]any `json:"parameters"`
	Batch      Batch          `json:"batch"`
}
type Resolved struct {
	Parameters       map[string]any `json:"parameters"`
	Recipe           map[string]any `json:"recipe"`
	Execution        map[string]any `json:"execution"`
	RecipeSHA256     string         `json:"recipe_sha256"`
	ParametersSHA256 string         `json:"parameters_sha256"`
	Batch            Batch          `json:"batch"`
}

func (c *Client) ResolveCompose(ctx context.Context, architecture, recipe, execution map[string]any, data training.ExternalDataParameters, world World) (Resolved, error) {
	parameters, err := c.parameters.ComposeParameters(recipe, execution)
	if err != nil {
		return Resolved{}, err
	}
	return c.Resolve(ctx, architecture, parameters, data, world)
}

// Resolve always performs both calls, including on a single-process host.
// Only RecipeSHA256 enters identity; the full parameter hash is run evidence.
func (c *Client) Resolve(ctx context.Context, architecture, parameters map[string]any, data training.ExternalDataParameters, world World) (Resolved, error) {
	var zero Resolved
	a, err := normalize(architecture)
	if err != nil {
		return zero, err
	}
	p, err := normalize(parameters)
	if err != nil {
		return zero, err
	}
	if err = c.architecture.Validate(a); err != nil {
		return zero, fmt.Errorf("architecture: %w", err)
	}
	if err = c.parameters.Validate(p); err != nil {
		return zero, fmt.Errorf("compose parameters: %w", err)
	}
	if world.WorldSize < 1 || world.Nodes < 1 || world.ProcessesPerNode < 1 || world.WorldSize/world.ProcessesPerNode != world.Nodes || world.WorldSize%world.ProcessesPerNode != 0 || world.NodeRank < 0 || world.NodeRank >= world.Nodes {
		return zero, fmt.Errorf("invalid resolve world shape")
	}
	world.Devices = append([]Device(nil), c.probe.Devices...)
	req := ResolveRequest{Kind: "resolve", Protocol: Protocol, Architecture: a, Parameters: p, Data: data, World: world}
	actual, err := c.resolve(ctx, req)
	if err != nil {
		return zero, err
	}
	req.World.World = applaunch.World{Nodes: 1, ProcessesPerNode: 1, WorldSize: 1}
	// Explicit decomposition may only fit the actual world. Re-resolve recipe
	// inputs with automatic execution values for the identity invariance check.
	req.Parameters, _, err = c.parameters.Split(p)
	if err != nil {
		return zero, err
	}
	single, err := c.resolve(ctx, req)
	if err != nil {
		return zero, fmt.Errorf("single-process resolve: %w", err)
	}
	if actual.RecipeSHA256 != single.RecipeSHA256 {
		return zero, fmt.Errorf("resolve recipe parameters change with world shape")
	}
	return actual, nil
}
func (c *Client) resolve(ctx context.Context, req ResolveRequest) (Resolved, error) {
	var result Resolved
	var frame resolvedFrame
	if err := c.command(ctx, "resolve", req, &frame); err != nil {
		return result, err
	}
	if frame.Kind != "resolve" || frame.Protocol != Protocol || frame.Parameters == nil {
		return result, fmt.Errorf("resolve requires kind, protocol and parameters")
	}
	if err := frame.Batch.validate(req.Data.GlobalBatchSequences, int64(req.World.WorldSize)); err != nil {
		return result, err
	}
	recipe, execution, err := c.parameters.Split(frame.Parameters)
	if err != nil {
		return result, err
	}
	result = Resolved{Parameters: frame.Parameters, Recipe: recipe, Execution: execution, RecipeSHA256: digest(recipe), ParametersSHA256: digest(frame.Parameters), Batch: frame.Batch}
	return result, nil
}
func (b Batch) validate(global, world int64) error {
	if global <= 0 || world <= 0 || b.WorldSize != world || b.MicroBatchSequences <= 0 || b.AccumulationSteps <= 0 || global%world != 0 || (global/world)%b.MicroBatchSequences != 0 || (global/world)/b.MicroBatchSequences != b.AccumulationSteps {
		return fmt.Errorf("resolve batch mismatch: micro-batch × accumulation × world size must equal global batch")
	}
	return nil
}
func normalize(v map[string]any) (map[string]any, error) {
	if v == nil {
		return nil, fmt.Errorf("parameters/architecture must be objects")
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	err = json.Unmarshal(b, &out)
	return out, err
}
func digest(v any) string { b, _ := json.Marshal(v); return application.Digest(b) }
func strict(b []byte, v any) error {
	if err := required(b, reflect.TypeOf(v).Elem()); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}
func write(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(b)+1 >= FrameLimit {
		return fmt.Errorf("protocol frame exceeds %d bytes", FrameLimit)
	}
	_, err = w.Write(append(b, '\n'))
	return err
}
func scanner(r io.Reader) *bufio.Scanner {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), FrameLimit)
	return s
}
func (c *Client) command(ctx context.Context, command string, input, output any) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r := c.request
	r.Command = command
	r.Args = nil
	p, err := applaunch.Start(ctx, r)
	if err != nil {
		return err
	}
	c.evidence = append(c.evidence, p.Evidence())
	done := make(chan error, 1)
	go func() { err := write(p.Stdin, input); p.Stdin.Close(); done <- err }()
	defer func() { cancel(); p.Stdin.Close(); p.Stdout.Close(); p.Wait(); <-done }()
	s := scanner(p.Stdout)
	if !s.Scan() {
		cancel()
		return errors.Join(fmt.Errorf("%s missing response", command), s.Err(), p.Wait())
	}
	var header struct {
		Kind string `json:"kind"`
	}
	if err = json.Unmarshal(s.Bytes(), &header); err != nil {
		return err
	}
	if header.Kind == "error" {
		return decodeError(s.Bytes())
	}
	if err = strict(s.Bytes(), output); err != nil {
		return fmt.Errorf("%s: %w", command, err)
	}
	if s.Scan() {
		return fmt.Errorf("%s emitted extra response", command)
	}
	if err = s.Err(); err != nil {
		return err
	}
	// Keep the result for deferred collection while still checking write errors.
	err = <-done
	done <- err
	return errors.Join(err, p.Wait())
}
func decodeError(b []byte) error {
	var f struct {
		Kind       string `json:"kind"`
		Protocol   int    `json:"protocol"`
		Error      string `json:"error"`
		ErrorClass string `json:"error_class"`
	}
	if err := strict(b, &f); err != nil {
		return err
	}
	if f.Protocol != Protocol || f.Error == "" || f.ErrorClass == "" {
		return fmt.Errorf("error frame requires protocol, error and error_class")
	}
	return &training.WorkerError{Message: f.Error, Class: f.ErrorClass}
}

// required distinguishes absent/null required fields from valid zero values.
func required(raw json.RawMessage, t reflect.Type) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == reflect.TypeOf(json.RawMessage{}) {
		return nil
	}
	if t.Kind() == reflect.Slice {
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return err
		}
		for _, item := range items {
			if err := required(item, t.Elem()); err != nil {
				return err
			}
		}
		return nil
	}
	if t.Kind() != reflect.Struct {
		return nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return err
	}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if f.Anonymous && tag == "" {
			if err := required(raw, f.Type); err != nil {
				return err
			}
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")
		if name == "-" {
			continue
		}
		if name == "" {
			name = f.Name
		}
		v, ok := obj[name]
		if !ok || bytes.Equal(v, []byte("null")) {
			if !strings.Contains(opts, "omitempty") {
				return fmt.Errorf("missing required field %s", name)
			}
			continue
		}
		if err := required(v, f.Type); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

// SchemaHashes are portable identity inputs; probe host facts are run-only.
func (c *Client) SchemaHashes() map[string]string {
	return map[string]string{"architecture": c.architecture.SHA256(), "parameters": c.parameters.SHA256()}
}
