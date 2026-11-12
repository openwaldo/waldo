// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package application

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// ParameterSchema keeps the unfiltered probe document, including unknown x-*
// annotations. Compiled annotations also survive $ref resolution.
type ParameterSchema struct {
	raw          []byte
	schema       *jsonschema.Schema
	architecture bool
}

func (s *ParameterSchema) Document() []byte { return bytes.Clone(s.raw) }
func (s *ParameterSchema) SHA256() string   { return Digest(s.raw) }

type annotations map[string]any

func (annotations) Validate(*jsonschema.ValidatorContext, any) {}

type noSchemaDownloads struct{}

func (noSchemaDownloads) Load(url string) (any, error) {
	return nil, fmt.Errorf("schema reference %s is not in the received document", url)
}

// CompileSchema accepts a self-contained JSON Schema 2020-12 document from
// probe. It never reads a file or fetches a remote reference.
func CompileSchema(raw []byte, architecture bool) (*ParameterSchema, error) {
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	if obj, ok := doc.(map[string]any); ok {
		if dialect, ok := obj["$schema"]; ok && dialect != "https://json-schema.org/draft/2020-12/schema" {
			return nil, fmt.Errorf("schema must use JSON Schema 2020-12")
		}
	}
	c := jsonschema.NewCompiler()
	c.DefaultDraft(jsonschema.Draft2020)
	c.UseLoader(noSchemaDownloads{})
	// Capture all vendor annotations, not only the role understood by WALDO.
	metaCompiler := jsonschema.NewCompiler()
	if err := metaCompiler.AddResource("https://waldo.invalid/annotations-meta", true); err != nil {
		return nil, err
	}
	meta, err := metaCompiler.Compile("https://waldo.invalid/annotations-meta")
	if err != nil {
		return nil, err
	}
	c.RegisterVocabulary(&jsonschema.Vocabulary{URL: "https://waldo.invalid/annotations", Schema: meta, Compile: func(_ *jsonschema.CompilerContext, obj map[string]any) (jsonschema.SchemaExt, error) {
		a := annotations{}
		for k, v := range obj {
			if strings.HasPrefix(k, "x-") {
				a[k] = v
			}
		}
		if role, ok := a["x-waldo-role"]; ok && role != "recipe" && role != "execution" {
			return nil, fmt.Errorf("unknown x-waldo-role %v", role)
		}
		return a, nil
	}})
	c.AssertVocabs()
	const location = "https://waldo.invalid/probe-schema"
	if err := c.AddResource(location, doc); err != nil {
		return nil, err
	}
	compiled, err := c.Compile(location)
	if err != nil {
		return nil, err
	}
	if err := checkRoles(compiled, "", architecture, map[roleVisit]bool{}); err != nil {
		return nil, err
	}
	// Definitions that are not referenced are still received schema, and must
	// not hide an invalid role or an execution annotation in architecture.
	var checkDefinitions func(any, string) error
	checkDefinitions = func(value any, pointer string) error {
		obj, ok := value.(map[string]any)
		if !ok {
			return nil
		}
		for _, key := range []string{"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"} {
			children, _ := obj[key].(map[string]any)
			for name, child := range children {
				p := pointer + "/" + key + "/" + escapePointer(name)
				if key == "$defs" || key == "definitions" {
					sub, err := c.Compile(location + "#" + p)
					if err != nil {
						return err
					}
					if err := checkRoles(sub, "", architecture, map[roleVisit]bool{}); err != nil {
						return err
					}
				}
				if err := checkDefinitions(child, p); err != nil {
					return err
				}
			}
		}
		for _, key := range []string{"not", "if", "then", "else", "items", "contains", "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "propertyNames", "contentSchema"} {
			if err := checkDefinitions(obj[key], pointer+"/"+key); err != nil {
				return err
			}
		}
		for _, key := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
			children, _ := obj[key].([]any)
			for i, child := range children {
				if err := checkDefinitions(child, fmt.Sprintf("%s/%s/%d", pointer, key, i)); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := checkDefinitions(doc, ""); err != nil {
		return nil, err
	}
	return &ParameterSchema{raw: bytes.Clone(raw), schema: compiled, architecture: architecture}, nil
}
func escapePointer(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~", "~0"), "/", "~1")
}
func declaredRole(s *jsonschema.Schema) string {
	for _, ext := range s.Extensions {
		if a, ok := ext.(annotations); ok {
			if r, ok := a["x-waldo-role"].(string); ok {
				return r
			}
		}
	}
	return ""
}
func effectiveRole(s *jsonschema.Schema, seen map[*jsonschema.Schema]bool) (string, error) {
	if s == nil || seen[s] {
		return "", nil
	}
	seen[s] = true
	role := declaredRole(s)
	// $ref applies to the same instance as its siblings; a role there constrains
	// siblings too, rather than silently overwriting them.
	peers := append([]*jsonschema.Schema{s.Ref}, s.AllOf...)
	for _, peer := range peers {
		other, err := effectiveRole(peer, seen)
		if err != nil {
			return "", err
		}
		if other != "" {
			if role != "" && role != other {
				return "", fmt.Errorf("contradicting x-waldo-role at %s", s.Location)
			}
			role = other
		}
	}
	return role, nil
}

type roleVisit struct {
	schema    *jsonschema.Schema
	inherited string
}

func checkRoles(s *jsonschema.Schema, inherited string, architecture bool, seen map[roleVisit]bool) error {
	if s == nil {
		return nil
	}
	if s.DraftVersion != 2020 {
		return fmt.Errorf("schema at %s must use JSON Schema 2020-12", s.Location)
	}
	visit := roleVisit{s, inherited}
	if seen[visit] {
		return nil
	}
	seen[visit] = true
	if s.DynamicRef != nil || s.RecursiveRef != nil {
		return fmt.Errorf("dynamic schema references are not supported for parameter roles at %s", s.Location)
	}
	role, err := effectiveRole(s, map[*jsonschema.Schema]bool{})
	if err != nil {
		return err
	}
	if role != "" {
		if architecture && role == "execution" {
			return fmt.Errorf("architecture schema contains execution role at %s", s.Location)
		}
		if inherited != "" && inherited != role {
			return fmt.Errorf("contradicting nested x-waldo-role at %s: %s inside %s", s.Location, role, inherited)
		}
		inherited = role
	}
	for key, child := range s.Properties {
		if waldoOwnedParameter(key) && (inherited == "execution" || hasExecutionRole(child, map[*jsonschema.Schema]bool{})) {
			return fmt.Errorf("WALDO-owned parameter %s cannot be execution-only", key)
		}
	}
	for _, child := range schemaChildren(s) {
		if err := checkRoles(child, inherited, architecture, seen); err != nil {
			return err
		}
	}
	return nil
}

func hasExecutionRole(s *jsonschema.Schema, seen map[*jsonschema.Schema]bool) bool {
	if s == nil || seen[s] {
		return false
	}
	seen[s] = true
	if declaredRole(s) == "execution" {
		return true
	}
	for _, child := range schemaChildren(s) {
		if hasExecutionRole(child, seen) {
			return true
		}
	}
	return false
}

func waldoOwnedParameter(key string) bool {
	switch key {
	case "data", "profile", "tokens", "epochs", "steps", "sequence_length", "global_batch_sequences", "seed", "checkpoint_every", "evaluate_every", "shuffle_buffer_records", "shuffle_buffer_bytes", "evaluation_fraction", "evaluation_selection", "evaluation_max_records", "evaluation_max_bytes", "order", "evaluation", "requested_tokens", "planned_token_capacity":
		return true
	}
	return false
}

// ComposeParameters checks the identity/execution boundary before resolve.
// Schema defaults are not injected; omitted values remain the trainer's job.
func (s *ParameterSchema) ComposeParameters(recipe, execution map[string]any) (map[string]any, error) {
	merged := map[string]any{}
	for key, value := range recipe {
		merged[key] = value
	}
	for key, value := range execution {
		if _, exists := merged[key]; exists {
			return nil, fmt.Errorf("parameter %s occurs in both trainer and execution", key)
		}
		merged[key] = value
	}
	r, e, err := s.Split(merged)
	if err != nil {
		return nil, err
	}
	for key := range recipe {
		if _, exists := e[key]; exists {
			return nil, fmt.Errorf("execution parameter %s belongs in stage execution, not trainer", key)
		}
	}
	for key := range execution {
		if _, exists := r[key]; exists {
			return nil, fmt.Errorf("identity parameter %s belongs in stage trainer, not execution", key)
		}
	}
	return merged, nil
}
func schemaChildren(s *jsonschema.Schema) []*jsonschema.Schema {
	out := []*jsonschema.Schema{s.Ref, s.Not, s.If, s.Then, s.Else, s.PropertyNames, s.UnevaluatedProperties, s.Contains, s.Items2020, s.UnevaluatedItems, s.ContentSchema}
	out = append(out, s.AllOf...)
	out = append(out, s.AnyOf...)
	out = append(out, s.OneOf...)
	out = append(out, s.PrefixItems...)
	for _, v := range s.Properties {
		out = append(out, v)
	}
	for _, v := range s.PatternProperties {
		out = append(out, v)
	}
	for _, v := range s.DependentSchemas {
		out = append(out, v)
	}
	if v, ok := s.AdditionalProperties.(*jsonschema.Schema); ok {
		out = append(out, v)
	}
	return out
}

// ValidateArchitecture enforces recipe-only roles even when a caller compiled
// the document without the architecture flag.
func (s *ParameterSchema) ValidateArchitecture(value any) error {
	if !s.architecture {
		checked, err := CompileSchema(s.raw, true)
		if err != nil {
			return err
		}
		return checked.Validate(value)
	}
	return s.Validate(value)
}

func (s *ParameterSchema) Validate(value any) error {
	normalized, err := jsonValue(value)
	if err != nil {
		return err
	}
	return s.schema.Validate(normalized)
}
func jsonValue(value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return jsonschema.UnmarshalJSON(bytes.NewReader(data))
}

// Split validates a complete resolve result and projects its values by role.
// No default values are injected; resolve is responsible for completion.
func (s *ParameterSchema) Split(value map[string]any) (recipe, execution map[string]any, err error) {
	v, err := jsonValue(value)
	if err != nil {
		return nil, nil, err
	}
	if err = s.schema.Validate(v); err != nil {
		return nil, nil, err
	}
	r, e, rp, ep, err := splitValue(v, []*jsonschema.Schema{s.schema}, "")
	if err != nil {
		return nil, nil, err
	}
	recipe = map[string]any{}
	execution = map[string]any{}
	if rp {
		recipe = r.(map[string]any)
	}
	if ep {
		execution = e.(map[string]any)
	}
	return recipe, execution, nil
}
func activeSchemas(s *jsonschema.Schema, v any, seen map[*jsonschema.Schema]bool) []*jsonschema.Schema {
	if s == nil || seen[s] {
		return nil
	}
	seen[s] = true
	out := []*jsonschema.Schema{s}
	peers := append([]*jsonschema.Schema{s.Ref}, s.AllOf...)
	for _, sub := range append(append([]*jsonschema.Schema{}, s.AnyOf...), s.OneOf...) {
		if sub.Validate(v) == nil {
			peers = append(peers, sub)
		}
	}
	if s.If != nil {
		if s.If.Validate(v) == nil {
			peers = append(peers, s.Then)
		} else {
			peers = append(peers, s.Else)
		}
	}
	if obj, ok := v.(map[string]any); ok {
		for key, sub := range s.DependentSchemas {
			if _, exists := obj[key]; exists {
				peers = append(peers, sub)
			}
		}
	}
	for _, sub := range peers {
		out = append(out, activeSchemas(sub, v, seen)...)
	}
	return out
}
func splitValue(v any, schemas []*jsonschema.Schema, inherited string) (any, any, bool, bool, error) {
	active := []*jsonschema.Schema{}
	seen := map[*jsonschema.Schema]bool{}
	for _, s := range schemas {
		active = append(active, activeSchemas(s, v, seen)...)
	}
	role := inherited
	for _, s := range active {
		r := declaredRole(s)
		if r != "" {
			if role != "" && r != role {
				return nil, nil, false, false, fmt.Errorf("conflicting roles at %s", s.Location)
			}
			role = r
		}
	}
	if role == "execution" {
		return nil, v, false, true, nil
	}
	switch value := v.(type) {
	case map[string]any:
		r, e := map[string]any{}, map[string]any{}
		for key, item := range value {
			children := []*jsonschema.Schema{}
			for _, s := range active {
				matched := false
				if sub, ok := s.Properties[key]; ok {
					children = append(children, sub)
					matched = true
				}
				for pattern, sub := range s.PatternProperties {
					if pattern.MatchString(key) {
						children = append(children, sub)
						matched = true
					}
				}
				if !matched {
					if sub, ok := s.AdditionalProperties.(*jsonschema.Schema); ok {
						children = append(children, sub)
					}
				}
				// Unevaluated role annotations cannot safely classify a property without
				// the validator's evaluated-property set. Fail closed rather than hash an
				// execution value or omit a recipe value.
				if s.UnevaluatedProperties != nil {
					if hasRole(s.UnevaluatedProperties, map[*jsonschema.Schema]bool{}) {
						return nil, nil, false, false, fmt.Errorf("roles in unevaluatedProperties cannot be projected")
					}
				}
			}
			rv, ev, rp, ep, err := splitValue(item, children, role)
			if err != nil {
				return nil, nil, false, false, err
			}
			if rp {
				r[key] = rv
			}
			if ep {
				e[key] = ev
			}
		}
		return r, e, len(r) > 0 || len(value) == 0, len(e) > 0, nil
	case []any:
		r, e := make([]any, len(value)), make([]any, len(value))
		rp, ep := len(value) == 0, false
		for i, item := range value {
			children := []*jsonschema.Schema{}
			for _, s := range active {
				if i < len(s.PrefixItems) {
					children = append(children, s.PrefixItems[i])
				} else if s.Items2020 != nil {
					children = append(children, s.Items2020)
				}
				if s.Contains != nil && s.Contains.Validate(item) == nil {
					children = append(children, s.Contains)
				}
				if s.UnevaluatedItems != nil && hasRole(s.UnevaluatedItems, map[*jsonschema.Schema]bool{}) {
					return nil, nil, false, false, fmt.Errorf("roles in unevaluatedItems cannot be projected")
				}
			}
			rv, ev, hasR, hasE, err := splitValue(item, children, role)
			if err != nil {
				return nil, nil, false, false, err
			}
			r[i], e[i] = rv, ev
			rp, ep = rp || hasR, ep || hasE
		}
		return r, e, rp, ep, nil
	default:
		return v, nil, true, false, nil
	}
}
func hasRole(s *jsonschema.Schema, seen map[*jsonschema.Schema]bool) bool {
	if s == nil || seen[s] {
		return false
	}
	seen[s] = true
	if declaredRole(s) != "" {
		return true
	}
	for _, child := range schemaChildren(s) {
		if hasRole(child, seen) {
			return true
		}
	}
	return false
}
