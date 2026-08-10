# Angular corpus preprocessing

The normalized corpus uses schema `normalized_document_v2` and a hybrid parser:

1. `markdown-it-py` parses CommonMark blocks and the GFM table extension;
2. an Angular adapter extracts `docs-*` elements without changing source line numbers;
3. deterministic normalization assigns block metadata and sentence IDs;
4. corpus validation rejects structural leaks and invalid provenance.

Install the package and test dependencies, then run from the repository root:

```bash
python -m pip install -e ".[test]"
python -m rag_chunking.cli.preprocess \
  --input data/raw/angular \
  --output data/processed/angular
```

When the package has not been installed, set `PYTHONPATH=src` before the command.
The CLI discovers `.md` files recursively in stable relative-path order and writes
byte-deterministic `documents.jsonl` and `manifest.json` files.

## Normalized structures

The parser emits these block types:

- `heading`, `paragraph`, `code_block`, `list`, `blockquote`, and `table`;
- `callout` for Angular admonitions and `docs-callout` prose;
- `html_block` for CommonMark raw HTML blocks;
- `code_reference` for external self-closing `docs-code path="..."` elements;
- `custom_block` for standalone Angular UI/media references.

Each parsed block records one-based inclusive `source_line_start` and
`source_line_end`. Angular wrappers such as workflows, steps, tabs, and cards are
stored in `metadata.container_path`. Tables retain parsed header, alignment, row,
and row/column-count metadata; lists retain item marker, nesting level, and text.
Links and images are retained in block metadata. Link-reference definitions are
stored once in document metadata and do not become visible paragraph content.

## External code references

The Markdown snapshot does not contain the files referenced by Angular's
self-closing `docs-code` elements. They are therefore represented explicitly as
unresolved `code_reference` blocks rather than raw tags or invented code. The
manifest reports the unresolved count. A future collection stage may resolve
these paths from a pinned Angular source snapshot without changing the parser
contract.

## Validation and migration

Validation checks schema version, source order and line spans, heading levels,
table metadata, code-reference state, raw Angular-tag leakage, identity, coverage,
and serialization round trips. Parser warnings and unknown Angular tags are
reported in the manifest audit section.

Readers reject old normalized artifacts without `normalized_document_v2`.
Regenerate all processed documents, chunks, and prompt caches after a parser or
schema change; cache identity includes the normalized document hash.
