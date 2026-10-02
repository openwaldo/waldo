// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestSchema2NativeMatchesSchema1(t *testing.T) {
	legacy, _, err := LoadCompose("../../composes/holding/tool-use.yaml")
	if err != nil {
		t.Fatal(err)
	}
	modern, _, err := LoadCompose("../../docs/examples/planned-schema2-tool-use.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(legacy, modern) {
		t.Fatal("schema-2 translation changes the native compose contract")
	}
	data, err := os.ReadFile("../../docs/examples/planned-schema2-tool-use.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := yaml.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "compose.json")
	if err := os.WriteFile(path, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	fromJSON, _, err := LoadCompose(path)
	if err != nil || !reflect.DeepEqual(legacy, fromJSON) {
		t.Fatalf("JSON translation differs: %v", err)
	}
}

func TestSchema2NativeFailsClosed(t *testing.T) {
	data, err := os.ReadFile("../../docs/examples/planned-schema2-tool-use.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{
		"unknown engine":         strings.Replace(string(data), "engine: waldo-native", "engine: huggingface-transformers", 1),
		"unknown provider":       strings.Replace(string(data), "provider: waldo-native", "provider: other", 1),
		"unknown nested config":  strings.Replace(string(data), "hidden_size: 1152", "hidden_size_typo: 1152", 1),
		"duplicate engine":       strings.Replace(string(data), "engine: waldo-native", "engine: waldo-native\n  engine: waldo-native", 1),
		"unknown training field": strings.Replace(string(data), "engine: waldo-native", "engine: waldo-native\n  ignored: true", 1),
		"missing engine":         strings.Replace(string(data), "engine: waldo-native", "", 1),
		"extra document":         string(data) + "\n---\n{}\n",
		"trainer payload":        string(data) + "    trainer:\n      class: Trainer\n",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "compose.yaml")
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadCompose(path); err == nil {
				t.Fatal("invalid schema-2 compose accepted")
			}
		})
	}
}
