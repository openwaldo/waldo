# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

# The rung-0007 preparation helper must hand `waldo index ingest` rows the declared adapter
# accepts. Six interaction-contract categories carry typed multi-turn messages, including
# assistant tool calls whose text content is empty; chat-messages rejects those. The full view
# therefore uses the source repository's prompt/response projection with dialogue-pair, which
# is also the representation of the published shard. This test needs only the standard library.

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELPER = os.path.join(
    REPO_ROOT, "composes", "0007-interaction-assistant-350m", "prepare-corpora.py"
)


def load_helper():
    spec = importlib.util.spec_from_file_location("rung_0007_prepare_corpora", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Shaped like openwaldo/post-training-data rows: the second message is an assistant tool
# call with empty text content.
TOOL_ROW = {
    "id": "tool-success-000000",
    "category": "tool-success",
    "split": "train",
    "prompt": "The lookup result reports status healthy. Summarize that result.",
    "response": "The service is healthy, according to the lookup.",
    "messages": [
        {"role": "user", "content": "Check the service."},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup"}]},
        {"role": "tool", "content": "{\"status\":\"healthy\"}"},
        {"role": "assistant", "content": "The service is healthy, according to the lookup."},
    ],
}
FORMATTING_ROW = {
    "id": "formatting-000000",
    "category": "formatting",
    "split": "train",
    "prompt": "Return the two names as a JSON array.",
    "response": "[\"Ada\",\"Lin\"]",
    "messages": [
        {"role": "user", "content": "Return the two names as a JSON array."},
        {"role": "assistant", "content": "[\"Ada\",\"Lin\"]"},
    ],
}


class PrepareCorporaTest(unittest.TestCase):
    def setUp(self):
        self.helper = load_helper()

    def write(self, rows, mapping):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.helper.write_source(directory, "corpus", "Title", "Description", rows, mapping)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        written = [
            json.loads(line)
            for line in (directory / "raw" / "training.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return manifest, written

    def test_full_view_uses_the_prompt_response_projection(self):
        rows = [self.helper.pair_row(row) for row in (TOOL_ROW, FORMATTING_ROW)]
        manifest, written = self.write(rows, self.helper.PAIR_INPUT)

        mapping = manifest["sources"][0]["input"]
        self.assertEqual(mapping["type"], "dialogue-pair")
        self.assertEqual(mapping["fields"], {"id": "id", "text": ["prompt"], "response": "response"})
        self.assertNotIn("messages", mapping)

        self.assertEqual(len(written), 2)
        for row in written:
            self.assertNotIn("messages", row)
            self.assertTrue(row["prompt"])
            self.assertTrue(row["response"])
        self.assertEqual(written[0]["prompt"], TOOL_ROW["prompt"])
        self.assertEqual(written[0]["response"], TOOL_ROW["response"])

    def test_formatting_view_keeps_typed_messages_without_empty_content(self):
        rows = [self.helper.messages_row(FORMATTING_ROW)]
        manifest, written = self.write(rows, self.helper.MESSAGES_INPUT)

        mapping = manifest["sources"][0]["input"]
        self.assertEqual(mapping["type"], "chat-messages")
        self.assertEqual(mapping["fields"], {"id": "id", "meta": {"category": "category"}})
        for message in written[0]["messages"]:
            self.assertTrue(message["content"])

    def test_raw_tree_evidence_matches_the_written_file(self):
        manifest, written = self.write([self.helper.pair_row(TOOL_ROW)], self.helper.PAIR_INPUT)
        self.assertEqual(manifest["raw"]["file_count"], 1)
        self.assertEqual(len(written), 1)
        self.assertGreater(manifest["raw"]["byte_count"], 0)


if __name__ == "__main__":
    unittest.main()
