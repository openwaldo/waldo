// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

// Package tokenizerapp drives tokenizer protocol 1 through the verified launcher.
package tokenizerapp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/model/application"
	"github.com/openwaldo/waldo/internal/model/pythonenv"
	"github.com/openwaldo/waldo/internal/training"
)

const frameLimit = 16 << 20

type Probe struct {
	Name           string         `json:"name"`
	VocabularySize int            `json:"vocabulary_size"`
	SpecialTokens  map[string]int `json:"special_tokens"`
	Deterministic  *bool          `json:"deterministic"`
	Schemas        struct {
		Configuration json.RawMessage `json:"configuration"`
	} `json:"schemas"`
}

// Identity is portable plan/run evidence. Launch observations remain separate.
type Identity struct {
	Source              application.ResolvedSource `json:"source"`
	Environment         pythonenv.Identity         `json:"environment"`
	Configuration       json.RawMessage            `json:"configuration"`
	ConfigurationSHA256 string                     `json:"configuration_sha256"`
	SchemaSHA256        string                     `json:"schema_sha256"`
	Probe               Probe                      `json:"probe"`
}

func (i Identity) SHA256() string { b, _ := json.Marshal(i); return application.Digest(b) }

type result struct {
	tokens []int
	err    error
}
type Client struct {
	p        *applaunch.Process
	cancel   context.CancelFunc
	done     chan struct{}
	slots    chan struct{}
	write    sync.Mutex
	mu       sync.Mutex
	next     uint64
	pending  map[uint64]chan result
	fatal    error
	identity Identity
	spec     training.TokenizerSpec
}

// Open probes and validates configuration before starting the long-lived codec.
// The caller owns ctx's deadline and must Close the client on every path.
func Open(ctx context.Context, request applaunch.Request, configuration map[string]any) (*Client, error) {
	if request.Source.Manifest.Kind != "waldo-tokenizer" {
		return nil, fmt.Errorf("expected tokenizer application")
	}
	if configuration == nil {
		return nil, fmt.Errorf("tokenizer configuration must be an object")
	}
	config, err := json.Marshal(configuration)
	if err != nil {
		return nil, err
	}
	request.Args = nil
	request.Command = "probe"
	probeCtx, cancel := context.WithCancel(ctx)
	p, err := applaunch.Start(probeCtx, request)
	if err != nil {
		cancel()
		return nil, err
	}
	defer func() { cancel(); p.Stdin.Close(); p.Stdout.Close(); p.Wait() }()
	writeDone := make(chan error, 1)
	go func() {
		e := json.NewEncoder(p.Stdin).Encode(struct {
			Configuration json.RawMessage `json:"configuration"`
		}{config})
		p.Stdin.Close()
		writeDone <- e
	}()
	scanner := bufio.NewScanner(p.Stdout)
	scanner.Buffer(make([]byte, 4096), frameLimit)
	var probe Probe
	if !scanner.Scan() {
		cancel()
		return nil, errors.Join(fmt.Errorf("tokenizer probe missing response"), scanner.Err(), p.Wait())
	}
	if err := strict(scanner.Bytes(), &probe); err != nil {
		return nil, fmt.Errorf("tokenizer probe: %w", err)
	}
	var special struct {
		SpecialTokens map[string]*int `json:"special_tokens"`
	}
	if err := json.Unmarshal(scanner.Bytes(), &special); err != nil {
		return nil, err
	}
	for name, id := range special.SpecialTokens {
		if id == nil {
			return nil, fmt.Errorf("tokenizer special token %s is null", name)
		}
	}
	if scanner.Scan() {
		return nil, fmt.Errorf("tokenizer probe emitted extra response")
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if err := errors.Join(<-writeDone, p.Wait()); err != nil {
		return nil, err
	}
	if probe.Name == "" || probe.VocabularySize < 1 || probe.Deterministic == nil {
		return nil, fmt.Errorf("tokenizer probe requires name, positive vocabulary_size and deterministic")
	}
	for _, key := range []string{"pad", "bos", "eos"} {
		if _, ok := probe.SpecialTokens[key]; !ok {
			return nil, fmt.Errorf("tokenizer probe missing special token %s", key)
		}
	}
	for key, id := range probe.SpecialTokens {
		if id < 0 || id >= probe.VocabularySize {
			return nil, fmt.Errorf("tokenizer special token %s outside vocabulary", key)
		}
	}
	schema, err := application.CompileSchema(probe.Schemas.Configuration, true)
	if err != nil {
		return nil, fmt.Errorf("tokenizer configuration schema: %w", err)
	}
	// Validate the JSON-normalized configuration, exactly as sent to Python.
	var normalized any
	if err := json.Unmarshal(config, &normalized); err != nil {
		return nil, err
	}
	if err := schema.Validate(normalized); err != nil {
		return nil, fmt.Errorf("tokenizer configuration: %w", err)
	}
	identity := Identity{Source: request.Source, Environment: request.Environment.Identity, Configuration: config, ConfigurationSHA256: application.Digest(config), SchemaSHA256: schema.SHA256(), Probe: probe}
	serveCtx, stop := context.WithCancel(ctx)
	request.Command = "serve"
	process, err := applaunch.Start(serveCtx, request)
	if err != nil {
		stop()
		return nil, err
	}
	c := &Client{p: process, cancel: stop, done: make(chan struct{}), slots: make(chan struct{}, 64), pending: map[uint64]chan result{}, identity: identity}
	c.spec = training.TokenizerSpec{Name: probe.Name, Revision: identity.SHA256(), VocabularySize: probe.VocabularySize, PadID: probe.SpecialTokens["pad"], BOSID: probe.SpecialTokens["bos"], EOSID: probe.SpecialTokens["eos"]}
	go c.read()
	if err := json.NewEncoder(process.Stdin).Encode(struct {
		Configuration json.RawMessage `json:"configuration"`
	}{config}); err != nil {
		c.fail(err)
		c.Close()
		return nil, err
	}
	return c, nil
}
func strict(data []byte, value any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}
func (c *Client) fail(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.fatal != nil {
		return
	}
	c.fatal = err
	for id, ch := range c.pending {
		ch <- result{err: fmt.Errorf("tokenizer request %d: %w", id, err)}
		delete(c.pending, id)
	}
	c.cancel()
}
func (c *Client) read() {
	defer close(c.done)
	scanner := bufio.NewScanner(c.p.Stdout)
	scanner.Buffer(make([]byte, 4096), frameLimit)
	for scanner.Scan() {
		var response struct {
			ID     uint64  `json:"id"`
			Tokens *[]*int `json:"tokens"`
			Error  string  `json:"error,omitempty"`
		}
		if err := strict(scanner.Bytes(), &response); err != nil {
			c.fail(fmt.Errorf("malformed response: %w", err))
			break
		}
		c.mu.Lock()
		ch, ok := c.pending[response.ID]
		c.mu.Unlock()
		if !ok {
			c.fail(fmt.Errorf("unexpected response id %d", response.ID))
			break
		}
		if response.Error != "" {
			c.fail(fmt.Errorf("request %d: %s", response.ID, response.Error))
			break
		}
		if response.Tokens == nil {
			c.fail(fmt.Errorf("request %d missing tokens", response.ID))
			break
		}
		valid := true
		tokens := make([]int, len(*response.Tokens))
		for i, id := range *response.Tokens {
			if id == nil || *id < 0 || *id >= c.spec.VocabularySize {
				valid = false
				break
			}
			tokens[i] = *id
		}
		if !valid {
			c.fail(fmt.Errorf("request %d token outside vocabulary", response.ID))
			break
		}
		c.mu.Lock()
		if _, ok := c.pending[response.ID]; ok {
			delete(c.pending, response.ID)
			ch <- result{tokens: tokens}
		}
		c.mu.Unlock()
	}
	err := scanner.Err()
	c.p.Stdout.Close()
	err = errors.Join(err, c.p.Wait())
	if err == nil {
		err = io.EOF
	}
	c.fail(err)
}

// EncodeChecked is the fallible extension used by WALDO data preparation.
// Concurrent calls pipeline up to 64 requests; responses may arrive out of order.
func (c *Client) EncodeChecked(text string) ([]int, error) {
	select {
	case c.slots <- struct{}{}:
	case <-c.done:
		return nil, c.Err()
	}
	defer func() { <-c.slots }()
	c.mu.Lock()
	if c.fatal != nil {
		err := c.fatal
		c.mu.Unlock()
		return nil, err
	}
	c.next++
	id := c.next
	ch := make(chan result, 1)
	c.pending[id] = ch
	c.mu.Unlock()
	frame, err := json.Marshal(struct {
		ID   uint64 `json:"id"`
		Text string `json:"text"`
	}{id, text})
	if err == nil && len(frame)+1 >= frameLimit {
		err = fmt.Errorf("request exceeds frame limit")
	}
	if err == nil {
		c.write.Lock()
		_, err = c.p.Stdin.Write(append(frame, '\n'))
		c.write.Unlock()
	}
	if err != nil {
		c.fail(err)
	}
	r := <-ch
	return r.tokens, r.err
}

// EncodeBatch pipelines a bounded batch, preserving input order.
func (c *Client) EncodeBatch(texts []string) ([][]int, error) {
	out := make([][]int, len(texts))
	errs := make([]error, len(texts))
	var wg sync.WaitGroup
	jobs := make(chan int)
	for n := 0; n < min(64, len(texts)); n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				out[i], errs[i] = c.EncodeChecked(texts[i])
			}
		}()
	}
	for i := range texts {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("batch record %d: %w", i, err)
		}
	}
	return out, nil
}

// ConcurrentSafe allows WALDO corpus workers to share this bounded pipeline.
func (c *Client) ConcurrentSafe() bool  { return true }
func (c *Client) Name() string          { return c.spec.Name }
func (c *Client) Encode(s string) []int { v, _ := c.EncodeChecked(s); return v }
func (c *Client) Count(s string) int    { return len(c.Encode(s)) }
func (c *Client) Decode([]int) string {
	c.fail(fmt.Errorf("external tokenizer decoding is unsupported"))
	return ""
}
func (c *Client) Err() error                   { c.mu.Lock(); defer c.mu.Unlock(); return c.fatal }
func (c *Client) Spec() training.TokenizerSpec { return c.spec }
func (c *Client) Identity() Identity {
	b, _ := json.Marshal(c.identity)
	var i Identity
	_ = json.Unmarshal(b, &i)
	i.Probe.Schemas.Configuration = bytes.Clone(c.identity.Probe.Schemas.Configuration)
	return i
}
func (c *Client) Evidence() applaunch.Evidence { return c.p.Evidence() }

// Close sends EOF and waits for a clean exit. A stuck server is cancelled
// after five seconds and its exit error is returned. Close is repeatable.
func (c *Client) Close() error {
	c.p.Stdin.Close()
	timer := time.AfterFunc(5*time.Second, c.cancel)
	defer timer.Stop()
	<-c.done
	c.cancel()
	return c.p.Wait()
}
