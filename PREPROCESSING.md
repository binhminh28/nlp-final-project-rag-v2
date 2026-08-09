# Angular corpus preprocessing

Run from the repository root:

```bash
python -m rag_chunking.cli.preprocess --input data/raw/angular --output data/processed/angular
```

When the package has not been installed, set `PYTHONPATH=src` (PowerShell:
`$env:PYTHONPATH = "src"`) before the command. The CLI recursively discovers
`.md` files in stable relative-path order and writes deterministic
`documents.jsonl` and `manifest.json` files.

By default, a source-file error is printed and processing continues so all bad
files appear in one run. Validation fails and no output is written if any file
fails. Pass `--fail-fast` to stop at the first source error.

The parser uses only the Python standard library. It recognizes standard
Markdown structures plus Angular's `docs-*` elements. HTML comments and closing
wrapper tags are source-only and omitted; semantic attributes and wrapper body
content are preserved. Paragraph whitespace is normalized conservatively, while
fenced and `<docs-code>` code retains line breaks and indentation.
