from rag_chunking.data.preprocess import preprocess_markdown, segment_sentences


def parse(markdown: str):
    return preprocess_markdown(markdown, doc_id="angular:test.md")


def test_heading_levels_and_order() -> None:
    blocks, _ = parse("# Angular\n## Components\n### Inputs\n")

    assert [(block.type, block.level, block.text) for block in blocks] == [
        ("heading", 1, "Angular"),
        ("heading", 2, "Components"),
        ("heading", 3, "Inputs"),
    ]


def test_heading_keeps_literal_hash_and_removes_spaced_closing_marker() -> None:
    blocks, _ = parse("## Using C#\n## Closed heading ##\n")
    assert [block.text for block in blocks] == ["Using C#", "Closed heading"]


def test_multiline_paragraph_normalizes_without_losing_inline_content() -> None:
    blocks, _ = parse(
        "Angular uses   [dependency injection](guide/di) and `signals`\n"
        "to build applications. A second sentence follows.\n"
    )

    assert len(blocks) == 1
    assert blocks[0].text == (
        "Angular uses [dependency injection](guide/di) and `signals` "
        "to build applications. A second sentence follows."
    )
    assert [sentence.text for sentence in blocks[0].sentences] == [
        "Angular uses [dependency injection](guide/di) and `signals` to build applications.",
        "A second sentence follows.",
    ]


def test_fenced_code_preserves_blank_lines_indentation_and_info() -> None:
    markdown = """```ts {header:"Example", linenums}
const x = 1;

if (x) {
  console.log(x);
}
```
"""
    blocks, _ = parse(markdown)

    assert len(blocks) == 1
    assert blocks[0].type == "code_block"
    assert blocks[0].language == "ts"
    assert blocks[0].metadata["info"] == 'ts {header:"Example", linenums}'
    assert blocks[0].text == "const x = 1;\n\nif (x) {\n  console.log(x);\n}"
    assert blocks[0].sentences == []


def test_mixed_document_preserves_structural_order() -> None:
    markdown = """# Angular

Intro paragraph.

- Components
- Services

```ts
const angular = true;
```

## Components

Component details.
"""
    blocks, _ = parse(markdown)

    assert [block.type for block in blocks] == [
        "heading",
        "paragraph",
        "list",
        "code_block",
        "heading",
        "paragraph",
    ]


def test_front_matter_is_metadata_not_prose() -> None:
    markdown = """---
title: Signals
draft: false
weight: 3
---
# Signals

Reactive state.
"""
    blocks, front_matter = parse(markdown)

    assert front_matter == {"title": "Signals", "draft": False, "weight": 3}
    assert [block.text for block in blocks] == ["Signals", "Reactive state."]


def test_angular_docs_code_is_a_code_block() -> None:
    markdown = """<docs-code header="Example" language="ts" linenums>
@Component({
  selector: 'app-root',
})
</docs-code>
"""
    blocks, _ = parse(markdown)

    assert len(blocks) == 1
    assert blocks[0].type == "code_block"
    assert blocks[0].language == "ts"
    assert blocks[0].text == "@Component({\n  selector: 'app-root',\n})"
    assert blocks[0].metadata["header"] == "Example"
    assert blocks[0].metadata["linenums"] == "true"


def test_angular_wrapper_preserves_title_and_inner_prose() -> None:
    markdown = """<docs-callout important title="Important detail">
Keep this semantic documentation text.
</docs-callout>
"""
    blocks, _ = parse(markdown)

    assert [block.type for block in blocks] == ["custom_block", "paragraph"]
    assert blocks[0].text == "Important detail"
    assert blocks[1].text == "Keep this semantic documentation text."


def test_decorative_header_is_the_angular_page_heading() -> None:
    blocks, _ = parse(
        '<docs-decorative-header title="What is Angular?" imgSrc="image.svg">\n'
        "Page introduction.\n"
        "</docs-decorative-header>\n"
    )
    assert [(block.type, block.level, block.text) for block in blocks] == [
        ("heading", 1, "What is Angular?"),
        ("paragraph", None, "Page introduction."),
    ]


def test_markdown_hard_line_break_is_preserved() -> None:
    blocks, _ = parse("First line.  \nSecond line.\n")
    assert blocks[0].text == "First line.  \nSecond line."


def test_blockquote_table_and_list_are_distinct_blocks() -> None:
    markdown = """> Documentation note.

| Name | Meaning |
| --- | --- |
| DI | Dependency injection |

1. First
2. Second
"""
    blocks, _ = parse(markdown)
    assert [block.type for block in blocks] == ["blockquote", "table", "list"]
    assert blocks[-1].ordered is True


def test_sentence_ids_are_stable_and_skip_headings_and_code() -> None:
    markdown = """# Heading

First sentence. Second sentence!

```text
Not. Prose.
```
"""
    first, _ = parse(markdown)
    second, _ = parse(markdown)

    first_ids = [sentence.sentence_id for block in first for sentence in block.sentences]
    second_ids = [sentence.sentence_id for block in second for sentence in block.sentences]
    assert first_ids == ["angular:test.md:s000000", "angular:test.md:s000001"]
    assert first_ids == second_ids


def test_sentence_segmenter_keeps_lowercase_continuation() -> None:
    assert segment_sentences("Use e.g. a signal. Then update it.") == [
        "Use e.g. a signal.",
        "Then update it.",
    ]
