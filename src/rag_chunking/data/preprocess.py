"""Conservative Markdown normalization for the Angular documentation dialect."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .models import DocumentBlock, Sentence


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_LIST_RE = re.compile(r"^[ \t]*(?P<marker>[-+*]|\d+[.)])[ \t]+(?P<text>.*)$")
_BLOCKQUOTE_RE = re.compile(r"^[ \t]*>[ \t]?(.*)$")
_DOCS_TAG_RE = re.compile(r"^[ \t]*<(?P<closing>/)?docs-(?P<name>[a-z0-9-]+)\b", re.I)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])[ \t]+(?=(?:[\"'`(\[]*[A-Z0-9]))"
)


def preprocess_markdown(
    markdown: str, *, doc_id: str
) -> tuple[list[DocumentBlock], dict[str, Any]]:
    """Convert Markdown to ordered structural blocks and front-matter metadata."""

    # Newlines are normalized once for deterministic JSON across operating systems.
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    body, front_matter = _split_front_matter(normalized)
    lines = body.split("\n")
    blocks: list[DocumentBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = _normalize_prose_lines(paragraph_lines)
        paragraph_lines.clear()
        if text:
            blocks.append(DocumentBlock(type="paragraph", text=text))

    index = 0
    while index < len(lines):
        line = lines[index]

        if not line.strip():
            flush_paragraph()
            index += 1
            continue

        if "<!--" in line:
            flush_paragraph()
            comment_lines = [line]
            while "-->" not in "\n".join(comment_lines) and index + 1 < len(lines):
                index += 1
                comment_lines.append(lines[index])
            remainder = _HTML_COMMENT_RE.sub("", "\n".join(comment_lines)).strip()
            if remainder:
                paragraph_lines.append(remainder)
            index += 1
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            flush_paragraph()
            fence = fence_match.group(1)
            info = fence_match.group(2).strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines):
                closing = _FENCE_RE.match(lines[index])
                if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence):
                    break
                code_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise ValueError("Unclosed fenced code block")
            language = info.split(maxsplit=1)[0] if info else None
            metadata = {"info": info} if info and info != language else {}
            blocks.append(
                DocumentBlock(
                    type="code_block",
                    text="\n".join(code_lines),
                    language=language,
                    metadata=metadata,
                )
            )
            index += 1
            continue

        docs_tag_match = _DOCS_TAG_RE.match(line)
        if docs_tag_match and not docs_tag_match.group("closing"):
            flush_paragraph()
            tag_name = docs_tag_match.group("name").lower()
            tag_lines = [line]
            while ">" not in "\n".join(tag_lines) and index + 1 < len(lines):
                index += 1
                tag_lines.append(lines[index])
            opening_tag = "\n".join(tag_lines)
            if tag_name == "code" and not opening_tag.rstrip().endswith("/>"):
                code_lines = []
                index += 1
                while index < len(lines) and not re.match(r"^[ \t]*</docs-code>[ \t]*$", lines[index], re.I):
                    code_lines.append(lines[index])
                    index += 1
                if index >= len(lines):
                    raise ValueError("Unclosed <docs-code> block")
                attributes = _parse_tag_attributes(opening_tag)
                blocks.append(
                    DocumentBlock(
                        type="code_block",
                        text="\n".join(code_lines),
                        language=attributes.get("language"),
                        metadata={"syntax": "docs-code", **attributes},
                    )
                )
                index += 1
                continue

            attributes = _parse_tag_attributes(opening_tag)
            if tag_name == "decorative-header" and attributes.get("title"):
                # Angular uses this component as the page's visual H1.
                blocks.append(
                    DocumentBlock(
                        type="heading",
                        level=1,
                        text=attributes["title"],
                        metadata={"syntax": "docs-decorative-header"},
                    )
                )
                index += 1
                continue
            semantic_text = _custom_tag_text(tag_name, attributes, opening_tag)
            blocks.append(
                DocumentBlock(
                    type="custom_block",
                    text=semantic_text,
                    metadata={"tag": tag_name, **attributes},
                )
            )
            index += 1
            continue

        if docs_tag_match and docs_tag_match.group("closing"):
            # Closing wrapper tags carry structure for rendering, but no content.
            flush_paragraph()
            index += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            heading_text = re.sub(r"[ \t]+#+[ \t]*$", "", heading_match.group(2)).strip()
            blocks.append(
                DocumentBlock(
                    type="heading",
                    level=len(heading_match.group(1)),
                    text=heading_text,
                )
            )
            index += 1
            continue

        if re.match(r"^[ \t]{0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*$", line):
            flush_paragraph()
            index += 1
            continue

        if _BLOCKQUOTE_RE.match(line):
            flush_paragraph()
            quote_lines: list[str] = []
            while index < len(lines):
                quote_match = _BLOCKQUOTE_RE.match(lines[index])
                if not quote_match:
                    break
                quote_lines.append(quote_match.group(1))
                index += 1
            blocks.append(
                DocumentBlock(type="blockquote", text=_normalize_prose_lines(quote_lines))
            )
            continue

        list_match = _LIST_RE.match(line)
        if list_match:
            flush_paragraph()
            ordered = list_match.group("marker")[0].isdigit()
            list_lines = [line.rstrip()]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip():
                    break
                if _FENCE_RE.match(candidate) or _HEADING_RE.match(candidate):
                    break
                candidate_list = _LIST_RE.match(candidate)
                if candidate_list or candidate.startswith(("  ", "\t")):
                    list_lines.append(candidate.rstrip())
                    index += 1
                    continue
                break
            blocks.append(
                DocumentBlock(type="list", text="\n".join(list_lines), ordered=ordered)
            )
            continue

        if index + 1 < len(lines) and _is_table_header(line, lines[index + 1]):
            flush_paragraph()
            table_lines = [line.rstrip(), lines[index + 1].rstrip()]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index].rstrip())
                index += 1
            blocks.append(DocumentBlock(type="table", text="\n".join(table_lines)))
            continue

        paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    _assign_sentences(blocks, doc_id)
    return blocks, front_matter


def segment_sentences(text: str) -> list[str]:
    """Lightweight deterministic segmentation for prose-like blocks."""

    sentences: list[str] = []
    for part in text.split("\n"):
        stripped = part.strip()
        if not stripped:
            continue
        sentences.extend(piece.strip() for piece in _SENTENCE_BOUNDARY_RE.split(stripped) if piece.strip())
    return sentences


def _assign_sentences(blocks: Iterable[DocumentBlock], doc_id: str) -> None:
    counter = 0
    for block in blocks:
        if block.type not in {"paragraph", "list", "blockquote"}:
            continue
        for text in segment_sentences(block.text):
            block.sentences.append(Sentence(sentence_id=f"{doc_id}:s{counter:06d}", text=text))
            counter += 1


def _normalize_prose_lines(lines: Iterable[str]) -> str:
    output: list[str] = []
    for line in lines:
        hard_break = line.endswith("  ")
        stripped = re.sub(r"[ \t]+", " ", line.strip())
        if not stripped:
            continue
        separator = "\n" if output and output[-1].endswith("  ") else (" " if output else "")
        output.append(separator + stripped + ("  " if hard_break else ""))
    return "".join(output).strip()


def _split_front_matter(markdown: str) -> tuple[str, dict[str, Any]]:
    lines = markdown.split("\n")
    if not lines or lines[0].strip() != "---":
        return markdown, {}
    try:
        closing = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        return markdown, {}
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
    return "\n".join(lines[closing + 1 :]), metadata


def _parse_scalar(value: str) -> Any:
    if not value:
        return None
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
        return value.strip("\"'")


def _parse_tag_attributes(tag: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    pattern = re.compile(r"([:\w-]+)(?:\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+)))?")
    match = _DOCS_TAG_RE.match(tag)
    remainder = tag[match.end() :] if match else tag
    for attribute in pattern.finditer(remainder):
        key = attribute.group(1)
        if key == "/":
            continue
        value = next((item for item in attribute.groups()[1:] if item is not None), "true")
        attributes[key] = value.rstrip("/")
    return attributes


def _custom_tag_text(tag_name: str, attributes: dict[str, str], raw_tag: str) -> str:
    meaningful = [attributes[key] for key in ("title", "label", "header") if key in attributes]
    if meaningful:
        return " | ".join(meaningful)
    return re.sub(r"\s+", " ", raw_tag.strip())


def _is_table_header(line: str, separator: str) -> bool:
    if "|" not in line or "|" not in separator:
        return False
    cells = separator.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"[ \t]*:?-{3,}:?[ \t]*", cell) for cell in cells)
