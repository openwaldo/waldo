# ADR 0034: Keep ingestion profiles corpus-neutral

Status: accepted

## Decision

Physical containers and logical record mappings are separate facts. Current
built-in formats are text, Markdown, mbox, JSON, JSONL, Parquet, XML, PDF, and
EPUB.
Compressed JSONL and mbox are streamed.

Current logical profiles are `record-map`, `dialogue-pair`, `chat-messages`,
`ranked-conversation-tree`, `bounded-text`, and `xml-record`. JSON accepts one
object or one top-level array and streams array elements. Profiles are strict,
versioned plan facts and never recognize a particular corpus.

Per-record licenses are normalized and retained with raw evidence. Profile
policies for empty records, NUL handling, primary-content classification,
record bounds, and license selection participate in plan identity.

Adding a physical format requires a reviewed built-in adapter. WALDO does not
run an executable or external converter named by a manifest.
