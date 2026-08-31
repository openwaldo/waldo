// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/openwaldo/waldo/internal/corpus"
	"gopkg.in/yaml.v3"
)

const inputProfileMaximum = 1 << 20

const (
	ProfileRecordMap              = "record-map"
	ProfileDialoguePair           = "dialogue-pair"
	ProfileChatMessages           = "chat-messages"
	ProfileRankedConversationTree = "ranked-conversation-tree"
	ProfileBoundedText            = "bounded-text"
	ProfileXMLRecord              = "xml-record"
)

// InputProfile pins the expected physical format and describes how one input
// record becomes canonical text. WALDO still probes the bytes and rejects a
// source-directory manifest whose declared format does not match them.
type InputProfile struct {
	Format        string               `json:"format,omitempty" yaml:"format,omitempty"`
	Type          string               `json:"type" yaml:"type"`
	OnEmpty       string               `json:"on_empty,omitempty" yaml:"on_empty,omitempty"`
	NUL           string               `json:"nul,omitempty" yaml:"nul,omitempty"`
	MainContent   map[string]any       `json:"main_content,omitempty" yaml:"main_content,omitempty"`
	Fields        ProfileFields        `json:"fields,omitempty" yaml:"fields,omitempty"`
	Tree          ConversationTree     `json:"tree,omitempty" yaml:"tree,omitempty"`
	Messages      ChatMessagesMapping  `json:"messages,omitempty" yaml:"messages,omitempty"`
	Bounds        TextBounds           `json:"bounds,omitempty" yaml:"bounds,omitempty"`
	XML           XMLMapping           `json:"xml,omitempty" yaml:"xml,omitempty"`
	LicensePolicy corpus.LicensePolicy `json:"license_policy,omitempty" yaml:"license_policy,omitempty"`
}

// withDefaults makes corpus-visible conversion choices explicit in the plan.
// Structured records commonly contain stray NUL bytes from upstream exports;
// replacing them with spaces is loss-minimizing and keeps one bad byte from
// aborting an otherwise valid corpus. Callers may request strict rejection with
// nul: error.
func (profile InputProfile) withDefaults() InputProfile {
	if profile.recordProfile() && profile.NUL == "" {
		profile.NUL = "space"
	}
	return profile
}

type ChatMessagesMapping struct {
	Role        string            `json:"role,omitempty" yaml:"role,omitempty"`
	Content     string            `json:"content,omitempty" yaml:"content,omitempty"`
	System      string            `json:"system,omitempty" yaml:"system,omitempty"`
	Tools       string            `json:"tools,omitempty" yaml:"tools,omitempty"`
	RoleAliases map[string]string `json:"role_aliases,omitempty" yaml:"role_aliases,omitempty"`
}

type ProfileFields struct {
	Text         []string          `json:"text,omitempty" yaml:"text,omitempty"`
	TextFallback []string          `json:"text_fallback,omitempty" yaml:"text_fallback,omitempty"`
	ID           string            `json:"id,omitempty" yaml:"id,omitempty"`
	Date         string            `json:"date,omitempty" yaml:"date,omitempty"`
	Language     string            `json:"language,omitempty" yaml:"language,omitempty"`
	License      string            `json:"license,omitempty" yaml:"license,omitempty"`
	Source       string            `json:"source,omitempty" yaml:"source,omitempty"`
	Context      string            `json:"context,omitempty" yaml:"context,omitempty"`
	Response     string            `json:"response,omitempty" yaml:"response,omitempty"`
	Tools        string            `json:"tools,omitempty" yaml:"tools,omitempty"`
	Meta         map[string]string `json:"meta,omitempty" yaml:"meta,omitempty"`
}

type TextBounds struct {
	StartPattern string `json:"start_pattern,omitempty" yaml:"start_pattern,omitempty"`
	EndPattern   string `json:"end_pattern,omitempty" yaml:"end_pattern,omitempty"`
}

type XMLMapping struct {
	Exclude      []string `json:"exclude,omitempty" yaml:"exclude,omitempty"`
	SourcePrefix string   `json:"source_prefix,omitempty" yaml:"source_prefix,omitempty"`
	OnMalformed  string   `json:"on_malformed,omitempty" yaml:"on_malformed,omitempty"`
}

// LoadInputProfile reads one strict standalone YAML or JSON profile for direct
// local ingestion. Source-directory manifests use the same InputProfile shape.
func LoadInputProfile(path string) (InputProfile, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return InputProfile{}, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Size() > inputProfileMaximum {
		return InputProfile{}, fmt.Errorf("input profile must be a regular non-symlink file no larger than %d bytes", inputProfileMaximum)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return InputProfile{}, err
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	var profile InputProfile
	if err := decoder.Decode(&profile); err != nil {
		return InputProfile{}, err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple YAML documents are not allowed")
		}
		return InputProfile{}, err
	}
	if err := profile.Validate(); err != nil {
		return InputProfile{}, err
	}
	if profile.Type == "" {
		return InputProfile{}, fmt.Errorf("input profile type is required")
	}
	return profile, nil
}

type ConversationTree struct {
	Root          string `json:"root,omitempty" yaml:"root,omitempty"`
	Replies       string `json:"replies,omitempty" yaml:"replies,omitempty"`
	Text          string `json:"text,omitempty" yaml:"text,omitempty"`
	Rank          string `json:"rank,omitempty" yaml:"rank,omitempty"`
	MissingRank   string `json:"missing_rank,omitempty" yaml:"missing_rank,omitempty"`
	Role          string `json:"role,omitempty" yaml:"role,omitempty"`
	AssistantRole string `json:"assistant_role,omitempty" yaml:"assistant_role,omitempty"`
}

func (profile InputProfile) Validate() error {
	if profile.Format != "" {
		switch profile.Format {
		case "text", "markdown", "mbox", "json", "jsonl", "parquet", "xml", "pdf", "epub":
		default:
			return fmt.Errorf("unsupported input format %q", profile.Format)
		}
	}
	policy, err := corpus.NewLicensePolicy(profile.LicensePolicy.Include, profile.LicensePolicy.Exclude)
	if err != nil {
		return err
	}
	if len(policy.Include) != len(profile.LicensePolicy.Include) || len(policy.Exclude) != len(profile.LicensePolicy.Exclude) {
		return fmt.Errorf("license_policy patterns must be non-empty and unique")
	}
	if profile.OnEmpty != "" && profile.OnEmpty != "error" && profile.OnEmpty != "skip" {
		return fmt.Errorf("on_empty must be error or skip")
	}
	if profile.OnEmpty != "" && profile.Type != ProfileRecordMap && profile.Type != ProfileDialoguePair && profile.Type != ProfileChatMessages && profile.Type != ProfileBoundedText {
		return fmt.Errorf("on_empty is supported only for record-map, dialogue-pair, chat-messages, and bounded-text")
	}
	if profile.NUL != "" && profile.NUL != "error" && profile.NUL != "space" {
		return fmt.Errorf("nul must be error or space")
	}
	if profile.NUL != "" && !profile.recordProfile() {
		return fmt.Errorf("nul is supported only for record profiles")
	}
	if profile.MainContent != nil {
		if !profile.recordProfile() {
			return fmt.Errorf("main_content is supported only for record profiles")
		}
		if len(profile.MainContent) == 0 {
			return fmt.Errorf("main_content requires at least one field and value")
		}
		for path, value := range profile.MainContent {
			if err := validateFieldPath(path); err != nil {
				return fmt.Errorf("main_content: %w", err)
			}
			if _, ok := mainContentScalar(value); !ok {
				return fmt.Errorf("main_content value must be a string, number, or boolean")
			}
		}
	}
	switch profile.Type {
	case "":
		if profile.OnEmpty != "" || profile.NUL != "" || profile.MainContent != nil || !profile.Fields.empty() || profile.Tree != (ConversationTree{}) || !profile.Messages.empty() || profile.Bounds != (TextBounds{}) || !profile.XML.empty() || len(profile.LicensePolicy.Include) > 0 || len(profile.LicensePolicy.Exclude) > 0 {
			return fmt.Errorf("input profile fields require a type")
		}
		return nil
	case ProfileRecordMap:
		if profile.Format != "" && profile.Format != "json" && profile.Format != "jsonl" && profile.Format != "parquet" {
			return fmt.Errorf("record-map requires format json, jsonl, or parquet")
		}
		if len(profile.Fields.Text) == 0 {
			return fmt.Errorf("record-map requires fields.text")
		}
		if profile.Fields.Context != "" || profile.Fields.Response != "" || profile.Fields.Tools != "" || profile.Tree != (ConversationTree{}) || !profile.Messages.empty() || profile.Bounds != (TextBounds{}) || !profile.XML.empty() {
			return fmt.Errorf("record-map accepts text, id, date, language, license, source, and meta fields only")
		}
		for name, path := range profile.Fields.Meta {
			if strings.TrimSpace(name) == "" || strings.TrimSpace(path) == "" {
				return fmt.Errorf("record-map meta names and paths must be non-empty")
			}
		}
	case ProfileDialoguePair:
		if profile.Format != "" && profile.Format != "json" && profile.Format != "jsonl" && profile.Format != "parquet" {
			return fmt.Errorf("dialogue-pair requires format json, jsonl, or parquet")
		}
		if len(profile.Fields.Text) == 0 || profile.Fields.Response == "" {
			return fmt.Errorf("dialogue-pair requires fields.text and fields.response")
		}
		if len(profile.Fields.TextFallback) > 0 || profile.Fields.Source != "" || profile.Tree != (ConversationTree{}) || !profile.Messages.empty() || profile.Bounds != (TextBounds{}) || !profile.XML.empty() {
			return fmt.Errorf("dialogue-pair does not accept tree fields")
		}
		for name, path := range profile.Fields.Meta {
			if strings.TrimSpace(name) == "" || strings.TrimSpace(path) == "" {
				return fmt.Errorf("dialogue-pair meta names and paths must be non-empty")
			}
		}
	case ProfileChatMessages:
		if profile.Format != "" && profile.Format != "json" && profile.Format != "jsonl" && profile.Format != "parquet" {
			return fmt.Errorf("chat-messages requires format json, jsonl, or parquet")
		}
		if profile.Messages.Role == "" || profile.Messages.Content == "" {
			return fmt.Errorf("chat-messages requires messages.role and messages.content")
		}
		if len(profile.Fields.Text) > 0 || len(profile.Fields.TextFallback) > 0 || profile.Fields.Context != "" || profile.Fields.Response != "" || profile.Fields.Tools != "" || profile.Tree != (ConversationTree{}) || profile.Bounds != (TextBounds{}) || !profile.XML.empty() {
			return fmt.Errorf("chat-messages accepts identity, source, and meta fields plus messages only")
		}
		seenAliases := map[string]bool{}
		for source, target := range profile.Messages.RoleAliases {
			if strings.TrimSpace(source) == "" || !validChatRole(target) {
				return fmt.Errorf("chat-messages role_aliases must map non-empty names to system, user, assistant, or tool")
			}
			canonicalSource := strings.ToLower(strings.TrimSpace(source))
			if seenAliases[canonicalSource] {
				return fmt.Errorf("chat-messages role_aliases contains duplicate case-insensitive source %q", canonicalSource)
			}
			seenAliases[canonicalSource] = true
		}
		for name, path := range profile.Fields.Meta {
			if strings.TrimSpace(name) == "" || strings.TrimSpace(path) == "" {
				return fmt.Errorf("chat-messages meta names and paths must be non-empty")
			}
		}
	case ProfileRankedConversationTree:
		if profile.Format != "" && profile.Format != "json" && profile.Format != "jsonl" {
			return fmt.Errorf("ranked-conversation-tree requires format json or jsonl")
		}
		if profile.Tree.Replies == "" || profile.Tree.Text == "" || profile.Tree.Rank == "" {
			return fmt.Errorf("ranked-conversation-tree requires tree.replies, tree.text, and tree.rank")
		}
		if len(profile.Fields.Text) > 0 || len(profile.Fields.TextFallback) > 0 || profile.Fields.Context != "" || profile.Fields.Response != "" || profile.Fields.Source != "" || len(profile.Fields.Meta) > 0 || !profile.Messages.empty() || profile.Bounds != (TextBounds{}) || !profile.XML.empty() {
			return fmt.Errorf("ranked-conversation-tree text comes from the tree mapping")
		}
		if profile.Tree.MissingRank != "" && profile.Tree.MissingRank != "source-order" {
			return fmt.Errorf("ranked-conversation-tree tree.missing_rank must be source-order")
		}
	case ProfileBoundedText:
		if profile.Format != "" && profile.Format != "text" && profile.Format != "markdown" {
			return fmt.Errorf("bounded-text requires format text or markdown")
		}
		if profile.Bounds.StartPattern == "" || profile.Bounds.EndPattern == "" {
			return fmt.Errorf("bounded-text requires bounds.start_pattern and bounds.end_pattern")
		}
		if _, err := regexp.Compile(profile.Bounds.StartPattern); err != nil {
			return fmt.Errorf("invalid bounds.start_pattern: %w", err)
		}
		if _, err := regexp.Compile(profile.Bounds.EndPattern); err != nil {
			return fmt.Errorf("invalid bounds.end_pattern: %w", err)
		}
		if !profile.Fields.empty() || profile.Tree != (ConversationTree{}) || !profile.Messages.empty() || !profile.XML.empty() {
			return fmt.Errorf("bounded-text accepts bounds and on_empty only")
		}
	case ProfileXMLRecord:
		if profile.Format != "" && profile.Format != "xml" {
			return fmt.Errorf("xml-record requires format xml")
		}
		if len(profile.Fields.Text) == 0 {
			return fmt.Errorf("xml-record requires fields.text")
		}
		if len(profile.Fields.TextFallback) > 0 || profile.Fields.Context != "" || profile.Fields.Response != "" || profile.Tree != (ConversationTree{}) || !profile.Messages.empty() || profile.Bounds != (TextBounds{}) {
			return fmt.Errorf("xml-record accepts text, id, date, language, license, source, and meta fields only")
		}
		if profile.XML.OnMalformed != "" && profile.XML.OnMalformed != "error" && profile.XML.OnMalformed != "skip" {
			return fmt.Errorf("xml-record xml.on_malformed must be error or skip")
		}
		for _, selector := range profile.xmlSelectors() {
			if err := validateXMLSelector(selector); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported input profile %q", profile.Type)
	}
	if profile.Type != ProfileXMLRecord {
		for _, path := range profile.paths() {
			if err := validateFieldPath(path); err != nil {
				return err
			}
		}
	}
	return nil
}

func (fields ProfileFields) empty() bool {
	return len(fields.Text) == 0 && len(fields.TextFallback) == 0 && fields.ID == "" && fields.Date == "" && fields.Language == "" &&
		fields.License == "" && fields.Source == "" && fields.Context == "" && fields.Response == "" && fields.Tools == "" && len(fields.Meta) == 0
}

func (mapping XMLMapping) empty() bool {
	return len(mapping.Exclude) == 0 && mapping.SourcePrefix == "" && mapping.OnMalformed == ""
}

func (mapping ChatMessagesMapping) empty() bool {
	return mapping.Role == "" && mapping.Content == "" && mapping.System == "" && mapping.Tools == "" && len(mapping.RoleAliases) == 0
}

func (profile InputProfile) paths() []string {
	paths := append([]string(nil), profile.Fields.Text...)
	paths = append(paths, profile.Fields.TextFallback...)
	paths = append(paths, profile.Fields.ID, profile.Fields.Date, profile.Fields.Language,
		profile.Fields.License, profile.Fields.Source, profile.Fields.Context, profile.Fields.Response, profile.Fields.Tools,
		profile.Tree.Root, profile.Tree.Replies, profile.Tree.Text, profile.Tree.Rank,
		profile.Tree.Role, profile.Messages.Role, profile.Messages.Content, profile.Messages.System, profile.Messages.Tools)
	for _, path := range profile.Fields.Meta {
		paths = append(paths, path)
	}
	for path := range profile.MainContent {
		paths = append(paths, path)
	}
	return paths
}

func (profile InputProfile) xmlSelectors() []string {
	selectors := append([]string(nil), profile.Fields.Text...)
	selectors = append(selectors, profile.Fields.ID, profile.Fields.Date, profile.Fields.Language,
		profile.Fields.License, profile.Fields.Source)
	for _, selector := range profile.Fields.Meta {
		selectors = append(selectors, selector)
	}
	selectors = append(selectors, profile.XML.Exclude...)
	return selectors
}

func validateXMLSelector(selector string) error {
	if selector == "" {
		return nil
	}
	if !strings.HasPrefix(selector, "/") || selector == "/" || strings.HasSuffix(selector, "/") {
		return fmt.Errorf("XML selector %q must be an absolute XPath", selector)
	}
	parts, err := splitXPath(selector)
	if err != nil {
		return err
	}
	for position, part := range parts {
		if part == "" { // The empty segment in // is the descendant axis.
			if position == len(parts)-1 || (position > 0 && parts[position-1] == "") {
				return fmt.Errorf("invalid XML selector %q", selector)
			}
			continue
		}
		if strings.HasPrefix(part, "@") {
			if position != len(parts)-1 || !validXMLName(strings.TrimPrefix(part, "@")) {
				return fmt.Errorf("invalid XML selector %q", selector)
			}
			continue
		}
		if !validXMLName(part) {
			return fmt.Errorf("invalid XML selector %q", selector)
		}
	}
	return nil
}

var xmlQName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*(?::[A-Za-z_][A-Za-z0-9_.-]*)?$`)

func validXMLName(value string) bool {
	if value == "*" {
		return true
	}
	if strings.HasPrefix(value, "{") {
		end := strings.LastIndexByte(value, '}')
		return end > 1 && end < len(value)-1 && xmlQName.MatchString(value[end+1:])
	}
	return xmlQName.MatchString(value)
}

func splitXPath(selector string) ([]string, error) {
	var parts []string
	start := 1
	braces := 0
	for position := 1; position < len(selector); position++ {
		switch selector[position] {
		case '{':
			braces++
		case '}':
			braces--
			if braces < 0 {
				return nil, fmt.Errorf("invalid XML selector %q", selector)
			}
		case '/':
			if braces == 0 {
				parts = append(parts, selector[start:position])
				start = position + 1
			}
		}
	}
	if braces != 0 {
		return nil, fmt.Errorf("invalid XML selector %q", selector)
	}
	return append(parts, selector[start:]), nil
}

func validateFieldPath(path string) error {
	if path == "" {
		return nil
	}
	for _, segment := range strings.Split(path, ".") {
		name := strings.TrimSuffix(segment, "[]")
		if name == "" || strings.ContainsAny(name, "[]") {
			return fmt.Errorf("invalid declarative field path %q", path)
		}
	}
	return nil
}

func (profile InputProfile) recordProfile() bool {
	return profile.Type == ProfileRecordMap || profile.Type == ProfileDialoguePair || profile.Type == ProfileChatMessages || profile.Type == ProfileRankedConversationTree
}

func validChatRole(role string) bool {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "user", "assistant", "tool":
		return true
	default:
		return false
	}
}
