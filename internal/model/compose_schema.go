// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"bytes"
	"fmt"
	"io"

	"gopkg.in/yaml.v3"
)

// normalizeComposeDocument translates the schema-2 envelope into the durable
// compose representation. Native hashes retain schema-1 meaning; Transformers
// carries a distinct provider specification rather than native execution.
func normalizeComposeDocument(data []byte) ([]byte, error) {
	var document yaml.Node
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(&document); err != nil {
		return nil, err
	}
	var extra yaml.Node
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple YAML documents are not allowed")
		}
		return nil, err
	}
	if len(document.Content) != 1 || document.Content[0].Kind != yaml.MappingNode {
		return nil, fmt.Errorf("model compose must be a mapping")
	}
	root := document.Content[0]
	for i := 0; i < len(root.Content); i += 2 {
		if root.Content[i].Value != "training" {
			continue
		}
		value := root.Content[i+1]
		for j := 0; value.Kind == yaml.MappingNode && j < len(value.Content); j += 2 {
			if value.Content[j].Value == "engine" && value.Content[j+1].Value == "huggingface-transformers" {
				return normalizeTransformersCompose(data)
			}
		}
	}
	var schema int
	for i := 0; i < len(root.Content); i += 2 {
		if root.Content[i].Value == "schema" {
			if err := root.Content[i+1].Decode(&schema); err != nil {
				return nil, err
			}
		}
	}
	if schema != 2 {
		return data, nil
	}
	if err := knownYAMLFields(root, map[string]bool{"kind": true, "schema": true, "training": true, "base": true, "architecture": true, "interaction": true, "stages": true}); err != nil {
		return nil, err
	}
	engineFound, architectureFound := false, false
	var normalized []*yaml.Node
	for i := 0; i < len(root.Content); i += 2 {
		key, value := root.Content[i], root.Content[i+1]
		switch key.Value {
		case "schema":
			value.Value = "1"
		case "training":
			if err := knownYAMLFields(value, map[string]bool{"engine": true}); err != nil {
				return nil, fmt.Errorf("training: %w", err)
			}
			if len(value.Content) != 2 || value.Content[1].Tag != "!!str" || value.Content[1].Value != "waldo-native" {
				return nil, fmt.Errorf("schema-2 training.engine must be waldo-native or huggingface-transformers")
			}
			engineFound = true
			continue
		case "architecture":
			if err := knownYAMLFields(value, map[string]bool{"provider": true, "config": true}); err != nil {
				return nil, fmt.Errorf("architecture: %w", err)
			}
			var provider string
			var config *yaml.Node
			for j := 0; j < len(value.Content); j += 2 {
				switch value.Content[j].Value {
				case "provider":
					if value.Content[j+1].Tag != "!!str" {
						return nil, fmt.Errorf("architecture.provider must be a string")
					}
					provider = value.Content[j+1].Value
				case "config":
					config = value.Content[j+1]
				}
			}
			if provider != "waldo-native" || config == nil || config.Kind != yaml.MappingNode {
				return nil, fmt.Errorf("schema-2 architecture requires provider waldo-native and a config mapping")
			}
			value = config
			architectureFound = true
		}
		normalized = append(normalized, key, value)
	}
	if !engineFound || !architectureFound {
		return nil, fmt.Errorf("schema-2 native compose requires training.engine and architecture.provider/config")
	}
	root.Content = normalized
	return yaml.Marshal(&document)
}
