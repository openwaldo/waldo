# WALDO developer documentation

This directory contains the programmatic contracts, architecture, and
developer requirements that change with the WALDO source.

User quickstarts live in the separate
[`openwaldo/docs`](https://github.com/openwaldo/docs) repository. CLI help is
the authority for available commands and flags.

## Start here

- [Architecture](ARCHITECTURE.md): package boundaries and data flow.
- [Command reference](UX.md): current command organization and behavior.
- [Compatibility](COMPATIBILITY.md): supported persistent and public formats.
- [Testing](TESTING.md): required local and end-to-end checks.
- [Contributing](CONTRIBUTING.md): development and DCO requirements.

## Data and ingestion contracts

Read these in order:

1. [Ingestion](INGESTION.md): trust boundary and supported workflow.
2. [Ingestion manifest](INGESTION-MANIFEST.md): exact handoff schema.
3. [Ingestion architecture](INGESTION-DESIGN.md): implementation ownership and
   data flow.
4. [Fetcher contract](FETCHER-CONTRACT.md): acquisition responsibilities.
5. [OpenWALDO corpus BOM](OPENWALDO-BOM.md): resolved corpus provenance.

## Model contracts

- [Model compose](MODEL-COMPOSE.md): portable training-plan schema.
- [Model lifecycle](MODEL-LIFECYCLE.md): managed models, runs, and backends.
- [Model exports](MODEL-EXPORTS.md): release formats and quantization.
- [Foundation and sparse-MoE plan](FOUNDATION-MOE-PLAN.md): planned model
  lineages, native artifacts, and NeMo/Megatron execution.
- [Reference compose strategy](../composes/README.md): capability ladder,
  promotion gates, and corpus requirements for model experiments.
- [EU GPAI disclosure](EU-GPAI-DISCLOSURE.md): regulatory JSON projection.

## Project direction and decisions

- [Vision](VISION.md): product purpose and non-goals.
- [Roadmap](ROADMAP.md): implemented and pending work.
- [ADRs](adr/README.md): active architectural decisions.

ADRs record durable design constraints. Maintained contracts above provide the
complete implementation requirements.
