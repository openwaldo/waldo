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

// Budget is how much one piece of work trains, as compose parameters.
type Budget struct {
	Key   string // "tokens", "steps", or "epochs"
	Value int64
}

// stageKind is the compose stage type a round's stage trains.
func stageKind(stage string) string {
	if stage == "posttrain" {
		return "fine-tuning"
	}
	return "pre-training"
}

// unitCompose cuts one piece of work from the run file: the round's stage
// alone, starting from base, with the unit's data, seed, and budget. Everything
// else in the compose stands as the run file wrote it.
func (plan Plan) unitCompose(stage, unit, base string, seed int, budget Budget) (string, error) {
	compose := clone(plan.Compose)
	mapDelete(compose, "join")
	mapDelete(compose, "base")
	if base != "" {
		mapSet(compose, "base", mapping("model", scalar(base)))
		// The bootstrap trained the run's tokenizer into its architecture; every later unit inherits
		// that architecture from its base rather than training the tokenizer again.
		if mapGet(mapGet(mapGet(compose, "architecture"), "tokenizer"), "training") != nil {
			mapDelete(compose, "architecture")
		}
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
		// Fixed steps over a small unit repeat its data rather than stop short.
		mapSet(parameters, "epochs", scalar("1000"))
	}
	mapSet(parameters, "seed", scalar(strconv.Itoa(seed)))
	// A unit holds out the same records every round, so its losses compare across rounds and
	// no round trains on what another measures; the training shuffle still changes each round.
	mapSet(parameters, "evaluation_seed", scalar(strconv.Itoa(pieceSeed(plan.Name, unit))))
	if budget != probeBudget {
		// A round's work is its last weights: picking the best held-out checkpoint can hand back
		// the round's starting weights when a few records misjudge it.
		mapSet(parameters, "artifact_selection", scalar("final"))
	}
	if budget == probeBudget {
		// A probe only evaluates, so schedule lengths that must fit within a real run's steps don't apply.
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

// probeBudget is the one step a closer's evaluate-only probe declares.
var probeBudget = Budget{"tokens", 16384}

// budget is what a unit of a round's stage trains.
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

// pretrainSteps converts tokens_per_round into the fixed steps every replica trains.
func (plan Plan) pretrainSteps() (int64, error) {
	parameters := mapGet(stageNode(plan.Compose, "pre-training"), "parameters")
	batch, sequence := intAt(parameters, "batch_size"), intAt(parameters, "sequence_length")
	if batch <= 0 || sequence <= 0 {
		return 0, fmt.Errorf("the pre-training stage needs batch_size and sequence_length to size pretraining rounds")
	}
	return (plan.Rounds.TokensPerRound + batch*sequence - 1) / (batch * sequence), nil
}

// vocabulary is the architecture's vocabulary size, whose uniform loss a bootstrap starts from.
func (plan Plan) vocabulary() int64 {
	return intAt(mapGet(plan.Compose, "architecture"), "vocabulary_size")
}

// interaction is the run file's interaction block as compose YAML, so merged models chat like trained ones.
func (plan Plan) interaction() (string, error) {
	node := mapGet(plan.Compose, "interaction")
	if node == nil {
		return "", nil
	}
	data, err := yaml.Marshal(mapping("interaction", clone(node)))
	return string(data), err
}

// stageNode is the compose's first stage of a type, or nil.
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

// ensureMapping is the mapping under key, created when missing.
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

// clone deep-copies a node tree, so editing one unit's compose leaves the run file untouched.
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
