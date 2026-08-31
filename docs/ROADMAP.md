# Roadmap

This page separates shipped behavior from pending work. Git history and ADRs
contain completed implementation history.

## Implemented

- Managed and explicit Git index workflows.
- Corpus inspection, verification, audit, ingestion, update, and export.
- Manifest-backed ingestion with built-in text, Markdown, mbox, JSON, JSONL,
  Parquet, XML, PDF, and EPUB adapters.
- Canonical text and structured-conversation Parquet records.
- Local and S3 lookaside publication, verification, and explicit removal.
- Corpus, run, origin, model, release, and EU disclosure BOMs.
- Model composes, structured conversation training, assistant-response
  modeling, and interaction templates.
- MLX, PyTorch, single-node TorchTitan, and multi-node TorchTitan training.
- Native WALDO, Hugging Face, MLX, GGUF, and Ollama model exports.
- Optional Sigstore signing.

## Pending

- Built-in tree-aware document adapters, including LaTeX.
- Multimodal ingestion.
- Broader model architecture and tokenizer compatibility.
- Preference-training objectives.
- PyTorch generation.
- Hugging Face publication.
- Editable EU template rendering.
- Implemented lookaside mirroring.

Track active implementation work in issues rather than expanding this page
into a design backlog.
