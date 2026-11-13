// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0
package applaunch

import "testing"

// Each world-shape rule independently refuses a launch that torchrun would
// otherwise start with ranks inconsistent with WALDO's data partitioning.
func TestDistributedWorldShapeRefusals(t *testing.T) {
	master := func(w World) World { w.MasterAddress, w.MasterPort = "192.0.2.1", 29500; return w }
	r := fixture(t, "")
	r.Command, r.Profile = "train", "distributed"
	// An explicit interface keeps the result independent of host routing.
	r.NCCLInterface = "eth0"
	r.World = master(World{Nodes: 2, NodeRank: 1, ProcessesPerNode: 2, WorldSize: 4})
	if _, err := Prepare(r); err != nil {
		t.Fatalf("valid world refused: %v", err)
	}
	for name, world := range map[string]World{
		"world below nodes":    master(World{Nodes: 2, ProcessesPerNode: 2, WorldSize: 2}),
		"world not multiple":   master(World{Nodes: 2, ProcessesPerNode: 2, WorldSize: 5}),
		"rank equals nodes":    master(World{Nodes: 2, NodeRank: 2, ProcessesPerNode: 1, WorldSize: 2}),
		"negative rank":        master(World{Nodes: 2, NodeRank: -1, ProcessesPerNode: 1, WorldSize: 2}),
		"missing master":       {Nodes: 2, NodeRank: 1, ProcessesPerNode: 1, WorldSize: 2},
		"missing master host":  {Nodes: 2, NodeRank: 1, ProcessesPerNode: 1, WorldSize: 2, MasterPort: 29500},
		"missing master port":  {Nodes: 2, NodeRank: 1, ProcessesPerNode: 1, WorldSize: 2, MasterAddress: "192.0.2.1"},
		"master port too high": {Nodes: 2, NodeRank: 1, ProcessesPerNode: 1, WorldSize: 2, MasterAddress: "192.0.2.1", MasterPort: 65536},
	} {
		r.World = world
		if e, err := Prepare(r); err == nil {
			t.Fatalf("%s accepted: %v", name, e.Argv)
		}
	}
}
