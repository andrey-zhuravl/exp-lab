from __future__ import annotations

import json
from typing import Any, Iterable, List


def safe_load(stream: Any) -> Any:
    text = stream.read() if hasattr(stream, "read") else stream
    text = text.strip()
    if not text:
        return None
    if text[0] in "[{":
        return json.loads(text)

    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    index = 0

    def indentation(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def parse_block(indent: int) -> Any:
        nonlocal index
        mapping: dict[str, Any] = {}
        while index < len(lines):
            line = lines[index]
            current_indent = indentation(line)
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError("Invalid indentation in YAML input")
            stripped = line.strip()
            if stripped.startswith("- "):
                return parse_list(indent)
            if ":" not in stripped:
                raise ValueError(f"Invalid line: {stripped}")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            index += 1
            if raw_value == "":
                if index >= len(lines):
                    mapping[key] = None
                    continue
                next_line = lines[index]
                next_indent = indentation(next_line)
                if next_indent <= current_indent:
                    mapping[key] = None
                else:
                    if next_line.strip().startswith("- "):
                        mapping[key] = parse_list(next_indent)
                    else:
                        mapping[key] = parse_block(next_indent)
            else:
                mapping[key] = _parse_scalar(raw_value)
        return mapping

    def parse_list(indent: int) -> List[Any]:
        nonlocal index
        items: List[Any] = []
        while index < len(lines):
            line = lines[index]
            current_indent = indentation(line)
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError("Invalid list indentation")
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            value_part = stripped[2:].strip()
            index += 1
            if value_part == "":
                if index >= len(lines):
                    items.append(None)
                else:
                    next_line = lines[index]
                    next_indent = indentation(next_line)
                    if next_indent > current_indent:
                        if next_line.strip().startswith("- "):
                            items.append(parse_list(next_indent))
                        else:
                            items.append(parse_block(next_indent))
                    else:
                        items.append(None)
            else:
                items.append(_parse_scalar(value_part))
        return items

    index = 0
    return parse_block(0)


def safe_dump(data: Any, stream: Any, sort_keys: bool = False) -> None:
    lines: List[str] = []

    def dump_value(value: Any, indent: int) -> None:
        if isinstance(value, dict):
            keys = sorted(value.keys()) if sort_keys else value.keys()
            for key in keys:
                item = value[key]
                if isinstance(item, (dict, list)):
                    lines.append(" " * indent + f"{key}:")
                    dump_value(item, indent + 2)
                else:
                    lines.append(" " * indent + f"{key}: {_format_scalar(item)}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(" " * indent + "-")
                    dump_value(item, indent + 2)
                else:
                    lines.append(" " * indent + f"- {_format_scalar(item)}")
        else:
            lines.append(" " * indent + _format_scalar(value))

    dump_value(data, 0)
    text = "\n".join(lines) + "\n"
    stream.write(text)


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            parts = _split_inline(inner)
            return [_parse_scalar(part.strip()) for part in parts]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        parts = _split_inline(inner)
        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid inline mapping: {value}")
            key, val = part.split(":", 1)
            result[key.strip()] = _parse_scalar(val.strip())
        return result
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "~"}:
        return None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_inline(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        current.append(char)
    final = "".join(current).strip()
    if final:
        parts.append(final)
    return parts


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{key}: {_format_scalar(val)}" for key, val in value.items())
        return "{" + items + "}"
    text = str(value)
    if not text or any(ch in text for ch in "#,:{}[] "):
        escaped = text.replace('"', '\"')
        return f'"{escaped}"'
    return text
