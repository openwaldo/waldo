// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package join

import (
	"fmt"
	"slices"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Budget struct {
	Key   string // "tokens", "steps", or "epochs"
	Value int64
}

func stageKind(stage string) string {
	if stage == "posttrain" {
		return "fine-tuning"
	}
	return "pre-training"
}

// unitCompose cuts one piece of work from the run file: the round's stage, base, data, seed, and budget.
func (plan Plan) unitCompose(stage, unit, base string, seed int, budget Budget) (string, error) {
	compose := clone(plan.Compose)
	mapDelete(compose, "join")
	mapDelete(compose, "base")
	if base != "" {
		// A unit inherits its base's architecture; a pulled base records no initialization or dropout to compare.
		mapSet(compose, "base", mapping("model", scalar(base)))
		mapDelete(compose, "architecture")
	}
	selected := stageNode(compose, stageKind(stage))
	if selected == nil {
		return "", fmt.Errorf("the run file has no %s stage for %s rounds", stageKind(stage), stage)
	}
	mapSet(compose, "stages", sequence(selected))
	parameters := ensureMapping(selected, "parameters")
	for _, key := range []string{"tokens", "steps", "epochs"} {
		mapDelete(parameters, key)
	}
	mapSet(parameters, budget.Key, scalar(strconv.FormatInt(budget.Value, 10)))
	if budget.Key == "steps" {
		mapSet(parameters, "epochs", scalar("1000"))
	}
	mapSet(parameters, "seed", scalar(strconv.Itoa(seed)))
	// A unit holds out the same records every round, so its losses compare; units hold out by their own
	// seeds, so a bootstrap over every unit's data may train on records a later unit holds out.
	mapSet(parameters, "evaluation_seed", scalar(strconv.Itoa(pieceSeed(plan.Name, unit))))
	if budget != probeBudget {
		// A round's work is its last weights; the best held-out checkpoint can be the round's start.
		mapSet(parameters, "artifact_selection", scalar("final"))
	}
	if budget == probeBudget {
		for _, key := range []string{"warmup_steps", "warmdown_steps", "checkpoint_every", "evaluate_every"} {
			mapDelete(parameters, key)
		}
	}
	if stage == "posttrain" {
		corpora := mapGet(selected, "corpora")
		if corpora == nil || corpora.Kind != yaml.SequenceNode || len(corpora.Content) == 0 {
			return "", fmt.Errorf("the fine-tuning stage needs a corpus for its units to slice")
		}
		index := slices.Index(plan.Units, unit)
		if index < 0 {
			return "", fmt.Errorf("unit %s is not in the run", unit)
		}
		slice := mapping("index", scalar(strconv.Itoa(index)), "count", scalar(strconv.Itoa(len(plan.Units))))
		slice.Style = yaml.FlowStyle
		mapSet(ensureMapping(corpora.Content[0], "filter"), "slice", slice)
	} else {
		var sources []string
		for _, name := range plan.Units {
			if unit == "all" || name == unit {
				sources = append(sources, plan.UnitData[name]...)
			}
		}
		include := sequence()
		for _, source := range sources {
			include.Content = append(include.Content, scalar(source))
		}
		include.Style = yaml.FlowStyle
		mapSet(ensureMapping(ensureMapping(selected, "filter"), "sources"), "include", include)
	}
	data, err := yaml.Marshal(compose)
	return string(data), err
}

var probeBudget = Budget{"tokens", 16384}

func (plan Plan) budget(stage string) (Budget, error) {
	switch stage {
	case "bootstrap":
		return Budget{"tokens", plan.Rounds.BootstrapTokens}, nil
	case "posttrain":
		return Budget{"epochs", int64(plan.Rounds.EpochsPerRound)}, nil
	}
	steps, err := plan.pretrainSteps()
	return Budget{"steps", steps}, err
}

func (plan Plan) pretrainSteps() (int64, error) {
	parameters := mapGet(stageNode(plan.Compose, "pre-training"), "parameters")
	batch, sequence := intAt(parameters, "batch_size"), intAt(parameters, "sequence_length")
	if batch <= 0 || sequence <= 0 {
		return 0, fmt.Errorf("the pre-training stage needs batch_size and sequence_length to size pretraining rounds")
	}
	return (plan.Rounds.TokensPerRound + batch*sequence - 1) / (batch * sequence), nil
}

func (plan Plan) vocabulary() int64 {
	return intAt(mapGet(plan.Compose, "architecture"), "vocabulary_size")
}

func (plan Plan) interaction() (string, error) {
	node := mapGet(plan.Compose, "interaction")
	if node == nil {
		return "", nil
	}
	data, err := yaml.Marshal(mapping("interaction", clone(node)))
	return string(data), err
}

func stageNode(compose *yaml.Node, kind string) *yaml.Node {
	stages := mapGet(compose, "stages")
	if stages == nil {
		return nil
	}
	for _, stage := range stages.Content {
		if node := mapGet(stage, "type"); node != nil && node.Value == kind {
			return stage
		}
	}
	return nil
}

func mapGet(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for index := 0; index+1 < len(node.Content); index += 2 {
		if node.Content[index].Value == key {
			return node.Content[index+1]
		}
	}
	return nil
}

func mapSet(node *yaml.Node, key string, value *yaml.Node) {
	for index := 0; index+1 < len(node.Content); index += 2 {
		if node.Content[index].Value == key {
			node.Content[index+1] = value
			return
		}
	}
	node.Content = append(node.Content, scalar(key), value)
}

func mapDelete(node *yaml.Node, key string) {
	for index := 0; index+1 < len(node.Content); index += 2 {
		if node.Content[index].Value == key {
			node.Content = append(node.Content[:index], node.Content[index+2:]...)
			return
		}
	}
}

func ensureMapping(node *yaml.Node, key string) *yaml.Node {
	if child := mapGet(node, key); child != nil && child.Kind == yaml.MappingNode {
		return child
	}
	child := mapping()
	mapSet(node, key, child)
	return child
}

func intAt(node *yaml.Node, key string) int64 {
	value, err := strconv.ParseInt(valueOf(mapGet(node, key)), 10, 64)
	if err != nil {
		return 0
	}
	return value
}

func valueOf(node *yaml.Node) string {
	if node == nil {
		return ""
	}
	return node.Value
}

func scalar(value string) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Value: value}
}

func mapping(pairs ...any) *yaml.Node {
	node := &yaml.Node{Kind: yaml.MappingNode}
	for index := 0; index+1 < len(pairs); index += 2 {
		node.Content = append(node.Content, scalar(pairs[index].(string)), pairs[index+1].(*yaml.Node))
	}
	return node
}

func sequence(items ...*yaml.Node) *yaml.Node {
	return &yaml.Node{Kind: yaml.SequenceNode, Content: items}
}

func clone(node *yaml.Node) *yaml.Node {
	if node == nil {
		return nil
	}
	copied := *node
	copied.Content = make([]*yaml.Node, len(node.Content))
	for index, child := range node.Content {
		copied.Content[index] = clone(child)
	}
	return &copied
}
