from rag_chunking.data.parser_v2 import parse_markdown


def parse(markdown: str):
    return parse_markdown(markdown, doc_id="angular:v2.md")


def test_commonmark_setext_and_indented_code_have_source_spans() -> None:
    result = parse("Title\n=====\n\n    const value = 1;\n")

    assert [(block.type, block.text) for block in result.blocks] == [
        ("heading", "Title"),
        ("code_block", "const value = 1;"),
    ]
    assert result.blocks[0].level == 1
    assert (result.blocks[0].source_line_start, result.blocks[0].source_line_end) == (1, 2)
    assert result.blocks[1].metadata["syntax"] == "indented"


def test_inline_control_comment_does_not_hide_decorative_header() -> None:
    result = parse(
        '<docs-decorative-header title="Signals"> <!-- markdownlint-disable-line -->\n'
        "Reactive state.\n"
        "</docs-decorative-header>\n"
    )

    assert [(block.type, block.text) for block in result.blocks] == [
        ("heading", "Signals"),
        ("paragraph", "Reactive state."),
    ]
    assert result.blocks[0].metadata["syntax"] == "docs-decorative-header"


def test_self_closing_docs_code_is_an_explicit_unresolved_reference() -> None:
    result = parse(
        '<docs-code header="Example (<tag>)"\n'
        '  path="examples/example.ts" language="ts"/>\n'
    )

    block = result.blocks[0]
    assert block.type == "code_reference"
    assert block.text == "Referenced code: Example (<tag>)"
    assert block.language == "ts"
    assert block.metadata["path"] == "examples/example.ts"
    assert block.metadata["resolved"] is False
    assert result.metadata["audit"]["unresolved_code_references"] == 1


def test_reference_definitions_are_document_metadata_not_visible_blocks() -> None:
    result = parse(
        '[Guide]: /guide "Guide title"\n\n'
        "Read [the guide][Guide].\n"
    )

    assert [block.text for block in result.blocks] == ["Read [the guide][Guide]."]
    assert result.metadata["link_definitions"]["GUIDE"]["href"] == "/guide"
    assert result.blocks[0].metadata["links"] == [
        {"href": "/guide", "title": "Guide title"}
    ]


def test_nested_list_items_are_structured_in_metadata() -> None:
    result = parse("- Parent\n  - Child\n1. Ordered sibling\n")

    lists = [block for block in result.blocks if block.type == "list"]
    assert lists[0].metadata["items"][:2] == [
        {"level": 0, "marker": "-", "ordered": False, "text": "Parent"},
        {"level": 1, "marker": "-", "ordered": False, "text": "Child"},
    ]


def test_table_nested_in_list_is_retained_as_structured_metadata() -> None:
    result = parse(
        "1. Compare the options.\n\n"
        "   | Name | Meaning |\n"
        "   | :--- | ---: |\n"
        "   | A | Value |\n"
    )

    block = result.blocks[0]
    assert block.type == "list"
    assert block.metadata["nested_tables"][0]["header"] == ["Name", "Meaning"]
    assert block.metadata["nested_tables"][0]["rows"] == [["A", "Value"]]


def test_table_metadata_handles_alignment_and_escaped_pipe() -> None:
    result = parse(
        "| Name | Meaning |\n"
        "| :--- | ---: |\n"
        r"| a\|b | value |" "\n"
    )

    block = result.blocks[0]
    assert block.type == "table"
    assert block.metadata["header"] == ["Name", "Meaning"]
    assert block.metadata["alignments"] == ["left", "right"]
    assert block.metadata["rows"] == [["a|b", "value"]]


def test_ragged_table_rows_are_padded_and_audited() -> None:
    result = parse(
        "| Name | Type | Default |\n"
        "| --- | --- | --- |\n"
        "| project | string |\n"
    )

    block = result.blocks[0]
    assert block.metadata["rows"] == [["project", "string", ""]]
    assert block.metadata["ragged_rows"] == [
        {"row_index": 0, "source_column_count": 2}
    ]
    assert any("ragged table" in warning for warning in result.metadata["audit"]["warnings"])


def test_angular_container_path_preserves_workflow_and_step() -> None:
    result = parse(
        "<docs-workflow>\n"
        '<docs-step title="Install">\n'
        "Run the command.\n"
        "</docs-step>\n"
        "</docs-workflow>\n"
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].metadata["container_path"] == [
        {"type": "workflow"},
        {"type": "step", "title": "Install"},
    ]


def test_admonition_and_html_are_distinct_block_types() -> None:
    result = parse(
        "IMPORTANT: Preserve this warning.\n\n"
        "<div>\nVisible HTML content.\n</div>\n"
    )

    assert [block.type for block in result.blocks] == ["callout", "html_block"]
    assert result.blocks[0].metadata["callout_kind"] == "important"
    assert "Visible HTML content" in result.blocks[1].text


def test_image_keeps_alt_text_and_source_metadata() -> None:
    result = parse("See ![Architecture diagram](assets/architecture.png).\n")

    assert result.blocks[0].metadata["images"] == [
        {"src": "assets/architecture.png", "alt": "Architecture diagram"}
    ]
