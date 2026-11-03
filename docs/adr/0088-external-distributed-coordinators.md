# ADR 0088: External distributed node-local coordinators

Status: accepted

## Decision

External protocol-1 training uses torchrun and the existing hostfile/Fuzzball
binary staging and readiness transport. The native worker protocol remains
unchanged. External node plans carry immutable application, corpus, execution
and evaluation pins; secondaries verify them before acknowledging readiness.

The data plane is always `node-local-cache`. Each coordinator constructs the
canonical stream and assigns sequence ordinal modulo world size to a global
rank. It emits its local ranks' sequences and an explicit exclusive global
ordinal at each micro-batch boundary. Local leaders fan out records only within
their node. The primary ledger accounts for the full canonical stream, including
sequences assigned to other nodes, so global reported consumption remains
verifiable without transporting corpus data between nodes.

World shape, host list, launch arguments, transport settings and data plane are
run provenance only. Resume preserves topology. Distributed checkpoint begin
hashes normalize node rank and the staged initialization file's host path;
immutable initialization artifact facts remain pinned. Existing single-process
checkpoint hashes are unchanged.

Secondary coordinators consume verified checkpoint bundles, emit no lifecycle
records, and report process exit only. Global rank zero alone declares
checkpoints, evaluations and published artifacts. Native compatibility broadcast
is not exposed to external trainers.
