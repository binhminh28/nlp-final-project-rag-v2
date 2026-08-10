"""CommonMark/GFM parser with source-aware Angular documentation adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .models import DocumentBlock, Sentence


PARSER_NAME = "markdown-it-py:commonmark+table/angular-v2"

_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_DOCS_OPEN_RE = re.compile(r"^[ \t]*<docs-(?P<tag>[a-z0-9-]+)\b", re.I)
_DOCS_CLOSE_RE = re.compile(r"^[ \t]*</docs-(?P<tag>[a-z0-9-]+)>[ \t]*$", re.I)
_CONTROL_COMMENT_RE = re.compile(
    r"<!--\s*(?:markdownlint|vale|prettier)(?:-|\s)[\s\S]*?-->", re.I
)
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_ADMONITION_RE = re.compile(
    r"^(?P<kind>NOTE|TIP|IMPORTANT|WARNING|CAUTION|HELPFUL):[ \t]*(?P<body>[\s\S]*)$",
    re.I,
)
_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-+*]|\d+[.)])[ \t]+(?P<text>.*)$"
)
_BLOCK_HTML_TAG_RE = re.compile(
    r"</?(?:div|ul|ol|li|p|section|article|details|summary|table|thead|tbody|tr|th|td)\b[^>]*>",
    re.I,
)
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])[ \t]+(?=(?:[\"'`(\[]*[A-Z0-9]))"
)

_MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")


@dataclass(slots=True)
class AngularElement:
    tag: str
    attrs: dict[str, str]
    source_start: int
    source_end: int
    content_start: int
    content_end: int
    kind: str
    text: str = ""

    def context_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"type": self.tag}
        for key in ("title", "label", "header", "name", "path", "language"):
            if key in self.attrs:
                value[key] = self.attrs[key]
        return value


@dataclass(slots=True)
class ParsedMarkdown:
    blocks: list[DocumentBlock]
    front_matter: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_markdown(markdown: str, *, doc_id: str) -> ParsedMarkdown:
    """Parse one Markdown document into deterministic v2 blocks."""

    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    body, front_matter, line_offset = _split_front_matter(normalized)
    original_lines = body.split("\n")
    sanitized_lines, events, containers, warnings = _extract_angular(original_lines)
    environment: dict[str, Any] = {}
    tokens = _MARKDOWN.parse("\n".join(sanitized_lines), environment)
    blocks = _tokens_to_blocks(
        tokens, sanitized_lines, events, containers, line_offset=line_offset
    )
    for block in blocks:
        if block.type == "table" and block.metadata.get("ragged_rows"):
            warnings.append(
                f"normalized ragged table rows at line {block.source_line_start}"
            )
        if block.type == "list":
            for nested in block.metadata.get("nested_tables", []):
                if nested.get("ragged_rows"):
                    warnings.append(
                        f"normalized ragged nested table rows at line {block.source_line_start}"
                    )
    _assign_sentences(blocks, doc_id)

    references = _reference_metadata(environment.get("references", {}), line_offset)
    unresolved = sum(block.type == "code_reference" for block in blocks)
    unknown_tags = sorted(
        {
            str(block.metadata.get("angular_tag"))
            for block in blocks
            if block.type == "custom_block" and block.metadata.get("unknown_angular_tag")
        }
    )
    metadata: dict[str, Any] = {
        "parser": PARSER_NAME,
        "link_definitions": references,
        "audit": {
            "unresolved_code_references": unresolved,
            "unknown_angular_tags": unknown_tags,
            "warnings": warnings,
        },
    }
    return ParsedMarkdown(blocks, front_matter, metadata)


def segment_sentences(text: str) -> list[str]:
    """Lightweight deterministic segmentation for prose-like blocks."""

    sentences: list[str] = []
    for part in text.split("\n"):
        stripped = part.strip()
        if not stripped:
            continue
        sentences.extend(
            piece.strip()
            for piece in _SENTENCE_BOUNDARY_RE.split(stripped)
            if piece.strip()
        )
    return sentences


def _extract_angular(
    source_lines: list[str],
) -> tuple[list[str], list[AngularElement], list[AngularElement], list[str]]:
    """Blank Angular wrapper syntax while retaining exact source line positions."""

    lines = [_CONTROL_COMMENT_RE.sub("", line) for line in source_lines]
    events: list[AngularElement] = []
    containers: list[AngularElement] = []
    stack: list[AngularElement] = []
    warnings: list[str] = []
    in_fence = False
    fence_character = ""
    fence_length = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_character = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not fence_match.group(2).strip()
            ):
                in_fence = False
            index += 1
            continue
        if in_fence:
            index += 1
            continue

        close_match = _DOCS_CLOSE_RE.match(line)
        if close_match:
            tag = close_match.group("tag").lower()
            lines[index] = ""
            match_position = next(
                (
                    position
                    for position in range(len(stack) - 1, -1, -1)
                    if stack[position].tag == tag
                ),
                None,
            )
            if match_position is not None:
                element = stack.pop(match_position)
                element.content_end = index
                element.source_end = index + 1
            elif tag not in {"decorative-header", "code"}:
                warnings.append(f"unmatched closing docs-{tag} at line {index + 1}")
            index += 1
            continue

        open_match = _DOCS_OPEN_RE.match(line)
        if open_match is None:
            index += 1
            continue

        tag = open_match.group("tag").lower()
        tag_start = index
        tag_lines = [line]
        combined = "\n".join(tag_lines)
        tag_end = _opening_tag_end(combined)
        while tag_end is None and index + 1 < len(lines):
            index += 1
            tag_lines.append(lines[index])
            combined = "\n".join(tag_lines)
            tag_end = _opening_tag_end(combined)
        if tag_end is None:
            raise ValueError(f"Unclosed <docs-{tag}> opening tag at line {tag_start + 1}")
        opening_tag = combined[: tag_end + 1]
        remainder = combined[tag_end + 1 :]
        attrs = _parse_tag_attributes(opening_tag)
        self_closing = opening_tag.rstrip().endswith("/>")
        inline_close = re.search(rf"</docs-{re.escape(tag)}>\s*$", remainder, re.I)
        if inline_close:
            remainder = remainder[: inline_close.start()]

        for line_index in range(tag_start, index + 1):
            lines[line_index] = ""
        if remainder.strip():
            lines[index] = remainder.strip()

        if tag == "code" and not self_closing and inline_close is None:
            close_index = index + 1
            while close_index < len(lines) and not re.match(
                r"^[ \t]*</docs-code>[ \t]*$", lines[close_index], re.I
            ):
                close_index += 1
            if close_index >= len(lines):
                raise ValueError(f"Unclosed <docs-code> block at line {tag_start + 1}")
            events.append(
                AngularElement(
                    tag="code",
                    attrs=attrs,
                    source_start=tag_start,
                    source_end=close_index + 1,
                    content_start=index + 1,
                    content_end=close_index,
                    kind="code_block",
                    text="\n".join(source_lines[index + 1 : close_index]),
                )
            )
            for line_index in range(index + 1, close_index + 1):
                lines[line_index] = ""
            index = close_index + 1
            continue

        if tag == "decorative-header":
            title = attrs.get("title")
            if title:
                events.append(
                    AngularElement(
                        tag=tag,
                        attrs=attrs,
                        source_start=tag_start,
                        source_end=index + 1,
                        content_start=index + 1,
                        content_end=index + 1,
                        kind="heading",
                        text=title,
                    )
                )
            else:
                warnings.append(f"decorative header without title at line {tag_start + 1}")
            index += 1
            continue

        if self_closing or inline_close is not None:
            events.append(
                AngularElement(
                    tag=tag,
                    attrs=attrs,
                    source_start=tag_start,
                    source_end=index + 1,
                    content_start=index + 1,
                    content_end=index + 1,
                    kind="code_reference"
                    if tag == "code" and attrs.get("path")
                    else "custom",
                )
            )
        else:
            previous_same = next(
                (
                    position
                    for position in range(len(stack) - 1, -1, -1)
                    if stack[position].tag == tag
                ),
                None,
            )
            if previous_same is not None:
                implicitly_closed = stack[previous_same:]
                del stack[previous_same:]
                for previous in implicitly_closed:
                    previous.content_end = tag_start
                    previous.source_end = tag_start
                    warnings.append(
                        f"implicitly closed docs-{previous.tag} before line {tag_start + 1}"
                    )
            container = AngularElement(
                tag=tag,
                attrs=attrs,
                source_start=tag_start,
                source_end=len(lines),
                content_start=index + 1,
                content_end=len(lines),
                kind="container",
            )
            containers.append(container)
            stack.append(container)
        index += 1

    for element in stack:
        warnings.append(f"unclosed docs-{element.tag} at line {element.source_start + 1}")
    return lines, events, containers, warnings


def _parse_tag_attributes(tag: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    pattern = re.compile(
        r"([:\w-]+)(?:\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+)))?"
    )
    match = _DOCS_OPEN_RE.match(tag)
    remainder = tag[match.end() :] if match else tag
    for attribute in pattern.finditer(remainder):
        key = attribute.group(1)
        value = next(
            (item for item in attribute.groups()[1:] if item is not None), "true"
        )
        attributes[key] = value.rstrip("/")
    return attributes


def _opening_tag_end(value: str) -> int | None:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {chr(34), chr(39)}:
            quote = character
        elif character == ">":
            return index
    return None


def _tokens_to_blocks(
    tokens: list[Token],
    lines: list[str],
    events: list[AngularElement],
    containers: list[AngularElement],
    *,
    line_offset: int,
) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open" and token.map:
            inline = tokens[index + 1]
            metadata = _inline_metadata(inline)
            blocks.append(
                _make_block(
                    "heading",
                    _normalize_prose(inline.content),
                    token.map,
                    containers,
                    line_offset,
                    level=int(token.tag[1:]),
                    metadata=metadata,
                )
            )
            index += 3
            continue
        if token.type == "paragraph_open" and token.map:
            inline = tokens[index + 1]
            text = _normalize_prose(_HTML_COMMENT_RE.sub("", inline.content))
            text = _normalize_prose(_BLOCK_HTML_TAG_RE.sub(" ", text))
            if text:
                metadata = _inline_metadata(inline)
                block_type = "paragraph"
                callout = _ADMONITION_RE.match(text)
                if callout:
                    block_type = "callout"
                    metadata.update(
                        {
                            "callout_kind": callout.group("kind").lower(),
                            "callout_syntax": "angular_admonition",
                        }
                    )
                if _inside_container(token.map, containers, "callout"):
                    block_type = "callout"
                    callout_container = _innermost_container(
                        token.map, containers, "callout"
                    )
                    metadata.update(
                        {
                            "callout_kind": _callout_kind(callout_container),
                            "callout_syntax": "docs-callout",
                        }
                    )
                blocks.append(
                    _make_block(
                        block_type,
                        text,
                        token.map,
                        containers,
                        line_offset,
                        metadata=metadata,
                    )
                )
            index += 3
            continue
        if token.type in {"fence", "code_block"} and token.map:
            info = token.info.strip()
            language = info.split(maxsplit=1)[0] if info else None
            metadata = {"syntax": "fence" if token.type == "fence" else "indented"}
            if info and info != language:
                metadata["info"] = info
            blocks.append(
                _make_block(
                    "code_block",
                    token.content.rstrip("\n"),
                    token.map,
                    containers,
                    line_offset,
                    language=language,
                    metadata=metadata,
                )
            )
            index += 1
            continue
        if token.type in {"bullet_list_open", "ordered_list_open"} and token.map:
            raw = _source_slice(lines, token.map)
            metadata = {
                "items": _parse_list_items(raw),
                "nested_tables": _nested_table_metadata(raw),
            }
            blocks.append(
                _make_block(
                    "list",
                    raw,
                    token.map,
                    containers,
                    line_offset,
                    ordered=token.type == "ordered_list_open",
                    metadata=metadata,
                )
            )
            index = _after_matching_close(tokens, index)
            continue
        if token.type == "blockquote_open" and token.map:
            raw = _source_slice(lines, token.map)
            text = _normalize_prose(
                "\n".join(re.sub(r"^[ \t]*>[ \t]?", "", line) for line in raw.split("\n"))
            )
            blocks.append(
                _make_block(
                    "blockquote", text, token.map, containers, line_offset
                )
            )
            index = _after_matching_close(tokens, index)
            continue
        if token.type == "table_open" and token.map:
            raw = _source_slice(lines, token.map)
            blocks.append(
                _make_block(
                    "table",
                    raw,
                    token.map,
                    containers,
                    line_offset,
                    metadata=_table_metadata(raw),
                )
            )
            index = _after_matching_close(tokens, index)
            continue
        if token.type == "html_block" and token.map:
            raw = token.content.strip()
            if raw and not _HTML_COMMENT_RE.fullmatch(raw):
                blocks.append(
                    _make_block(
                        "html_block",
                        raw,
                        token.map,
                        containers,
                        line_offset,
                        metadata={"syntax": "commonmark_html_block"},
                    )
                )
            index += 1
            continue
        index += 1

    blocks.extend(_event_block(event, containers, line_offset) for event in events)
    blocks.sort(
        key=lambda block: (
            block.source_line_start or 0,
            0 if block.type == "heading" else 1,
            block.source_line_end or 0,
        )
    )
    return blocks


def _make_block(
    block_type: str,
    text: str,
    source_map: list[int],
    containers: list[AngularElement],
    line_offset: int,
    *,
    level: int | None = None,
    language: str | None = None,
    ordered: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentBlock:
    value = dict(metadata or {})
    context = _container_path(source_map, containers)
    if context:
        value["container_path"] = context
    return DocumentBlock(
        type=block_type,  # type: ignore[arg-type]
        text=text,
        level=level,
        language=language,
        ordered=ordered,
        source_line_start=source_map[0] + line_offset + 1,
        source_line_end=source_map[1] + line_offset,
        metadata=value,
    )


def _event_block(
    event: AngularElement, containers: list[AngularElement], line_offset: int
) -> DocumentBlock:
    source_map = [event.source_start, event.source_end]
    metadata: dict[str, Any] = {
        "angular_tag": event.tag,
        "attributes": event.attrs,
    }
    if event.kind == "heading":
        metadata["syntax"] = "docs-decorative-header"
        return _make_block(
            "heading",
            event.text,
            source_map,
            containers,
            line_offset,
            level=1,
            metadata=metadata,
        )
    if event.kind == "code_block":
        metadata.update({"syntax": "docs-code", **event.attrs})
        return _make_block(
            "code_block",
            event.text,
            source_map,
            containers,
            line_offset,
            language=event.attrs.get("language"),
            metadata=metadata,
        )
    if event.kind == "code_reference":
        path = event.attrs["path"]
        label = event.attrs.get("header") or Path(path).name
        metadata.update(
            {
                "path": path,
                "language": event.attrs.get("language"),
                "resolved": False,
                "synthetic_text": True,
            }
        )
        return _make_block(
            "code_reference",
            f"Referenced code: {label}",
            source_map,
            containers,
            line_offset,
            language=event.attrs.get("language"),
            metadata=metadata,
        )

    meaningful = next(
        (
            event.attrs[key]
            for key in ("title", "label", "header", "alt", "name")
            if event.attrs.get(key)
        ),
        None,
    )
    if meaningful is None:
        reference = next(
            (
                event.attrs[key]
                for key in ("href", "src", "path")
                if event.attrs.get(key)
            ),
            None,
        )
        meaningful = f"{event.tag.replace('-', ' ').title()}: {reference}" if reference else event.tag.replace("-", " ").title()
    metadata["synthetic_text"] = True
    metadata["unknown_angular_tag"] = event.tag not in {
        "card",
        "card-container",
        "callout",
        "code-multifile",
        "nav-card",
        "nav-link",
        "pill",
        "pill-row",
        "step",
        "tab",
        "tab-group",
        "video",
        "workflow",
    }
    return _make_block(
        "custom_block",
        meaningful,
        source_map,
        containers,
        line_offset,
        metadata=metadata,
    )


def _container_path(
    source_map: list[int], containers: list[AngularElement]
) -> list[dict[str, object]]:
    start = source_map[0]
    matching = [
        item
        for item in containers
        if item.content_start <= start < item.content_end
    ]
    matching.sort(key=lambda item: (item.content_start, -item.content_end))
    return [item.context_dict() for item in matching]


def _inside_container(
    source_map: list[int], containers: list[AngularElement], tag: str
) -> bool:
    return _innermost_container(source_map, containers, tag) is not None


def _innermost_container(
    source_map: list[int], containers: list[AngularElement], tag: str
) -> AngularElement | None:
    matching = [
        item
        for item in containers
        if item.tag == tag and item.content_start <= source_map[0] < item.content_end
    ]
    return max(matching, key=lambda item: item.content_start, default=None)


def _callout_kind(container: AngularElement | None) -> str:
    if container is None:
        return "note"
    for key in ("important", "helpful", "warning", "tip", "note"):
        if key in container.attrs:
            return key
    return container.attrs.get("type", "note").lower()


def _inline_metadata(token: Token) -> dict[str, Any]:
    links: list[dict[str, str]] = []
    images: list[dict[str, str]] = []
    for child in token.children or []:
        if child.type == "link_open":
            item = {"href": str(child.attrs.get("href", ""))}
            if child.attrs.get("title"):
                item["title"] = str(child.attrs["title"])
            links.append(item)
        elif child.type == "image":
            item = {
                "src": str(child.attrs.get("src", "")),
                "alt": child.content,
            }
            if child.attrs.get("title"):
                item["title"] = str(child.attrs["title"])
            images.append(item)
    metadata: dict[str, Any] = {}
    if links:
        metadata["links"] = links
    if images:
        metadata["images"] = images
    return metadata


def _table_metadata(text: str) -> dict[str, Any]:
    lines = text.split("\n")
    if len(lines) < 2:
        return {"header": [], "alignments": [], "rows": []}
    header = _split_table_row(lines[0])
    delimiter = _split_table_row(lines[1])
    alignments: list[str] = []
    for cell in delimiter:
        stripped = cell.strip()
        if stripped.startswith(":") and stripped.endswith(":"):
            alignments.append("center")
        elif stripped.endswith(":"):
            alignments.append("right")
        elif stripped.startswith(":"):
            alignments.append("left")
        else:
            alignments.append("default")
    source_rows = [_split_table_row(line) for line in lines[2:]]
    ragged_rows = [
        {"row_index": index, "source_column_count": len(row)}
        for index, row in enumerate(source_rows)
        if len(row) != len(header)
    ]
    rows = [
        (row + [""] * max(0, len(header) - len(row)))[: len(header)]
        for row in source_rows
    ]
    return {
        "header": header,
        "alignments": alignments,
        "rows": rows,
        "column_count": len(header),
        "row_count": len(rows),
        "ragged_rows": ragged_rows,
    }


def _nested_table_metadata(text: str) -> list[dict[str, Any]]:
    lines = text.split("\n")
    tables: list[dict[str, Any]] = []
    index = 0
    delimiter_cell = re.compile(r"[ \t]*:?-{3,}:?[ \t]*")
    while index + 1 < len(lines):
        header = lines[index]
        delimiter = lines[index + 1]
        cells = delimiter.strip().strip("|").split("|") if "|" in delimiter else []
        if "|" not in header or not cells or not all(
            delimiter_cell.fullmatch(cell) for cell in cells
        ):
            index += 1
            continue
        end = index + 2
        while end < len(lines) and lines[end].strip() and "|" in lines[end]:
            end += 1
        nested = _table_metadata("\n".join(line.strip() for line in lines[index:end]))
        nested.update({"line_start": index, "line_end": end})
        tables.append(nested)
        index = end
    return tables


def _split_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith(r"\|"):
        value = value[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    code_fence = 0
    index = 0
    while index < len(value):
        character = value[index]
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
            current.append(character)
        elif character == "`":
            run = 1
            while index + run < len(value) and value[index + run] == "`":
                run += 1
            if code_fence == 0:
                code_fence = run
            elif code_fence == run:
                code_fence = 0
            current.extend("`" * run)
            index += run - 1
        elif character == "|" and code_fence == 0:
            cells.append("".join(current).strip().replace(r"\|", "|"))
            current = []
        else:
            current.append(character)
        index += 1
    cells.append("".join(current).strip().replace(r"\|", "|"))
    return cells


def _parse_list_items(text: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for line in text.split("\n"):
        match = _LIST_ITEM_RE.match(line)
        if not match:
            continue
        indent = match.group("indent").replace("\t", "    ")
        marker = match.group("marker")
        items.append(
            {
                "level": len(indent) // 2,
                "marker": marker,
                "ordered": marker[0].isdigit(),
                "text": match.group("text").strip(),
            }
        )
    return items


def _source_slice(lines: list[str], source_map: list[int]) -> str:
    return "\n".join(line.rstrip() for line in lines[source_map[0] : source_map[1]]).strip()


def _after_matching_close(tokens: list[Token], start: int) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        depth += tokens[index].nesting
        if depth == 0:
            return index + 1
    return len(tokens)


def _normalize_prose(text: str) -> str:
    output: list[str] = []
    for line in text.split("\n"):
        hard_break = line.endswith("  ")
        stripped = re.sub(r"[ \t]+", " ", line.strip())
        if not stripped:
            continue
        separator = "\n" if output and output[-1].endswith("  ") else (" " if output else "")
        output.append(separator + stripped + ("  " if hard_break else ""))
    return "".join(output).strip()


def _assign_sentences(blocks: Iterable[DocumentBlock], doc_id: str) -> None:
    counter = 0
    for block in blocks:
        if block.type not in {"paragraph", "list", "blockquote", "callout"}:
            continue
        for text in segment_sentences(block.text):
            block.sentences.append(
                Sentence(sentence_id=f"{doc_id}:s{counter:06d}", text=text)
            )
            counter += 1


def _reference_metadata(
    values: dict[str, Any], line_offset: int
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for label in sorted(values):
        source = values[label]
        item: dict[str, Any] = {"href": source.get("href", "")}
        if source.get("title"):
            item["title"] = source["title"]
        if source.get("map"):
            item["source_line_start"] = source["map"][0] + line_offset + 1
            item["source_line_end"] = source["map"][1] + line_offset
        references[label] = item
    return references


def _split_front_matter(markdown: str) -> tuple[str, dict[str, Any], int]:
    lines = markdown.split("\n")
    if not lines or lines[0].strip() != "---":
        return markdown, {}, 0
    try:
        closing = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration:
        return markdown, {}, 0
    raw_lines = lines[1:closing]
    metadata: dict[str, Any] = {}
    unparsed: list[str] = []
    for line in raw_lines:
        if not line.strip():
            continue
        if ":" not in line or line[:1].isspace():
            unparsed.append(line)
            continue
        key, raw_value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(raw_value.strip())
    if unparsed:
        metadata["_unparsed"] = "\n".join(unparsed)
    return "\n".join(lines[closing + 1 :]), metadata, closing + 1


def _parse_scalar(value: str) -> Any:
    if not value:
        return None
    import json

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
        if lowered in {"null", "~"}:
            return None
        return value.strip(chr(34) + chr(39))
