#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def content_text(content):
    chunks = []
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text", "text"}:
            value = item.get("text")
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).strip()


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: extract_transcript.py OUTPUT INPUT...")
    output = Path(sys.argv[1])
    records = []
    seen = set()
    for name in sys.argv[2:]:
        path = Path(name)
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "response_item":
                    continue
                payload = obj.get("payload", {})
                if payload.get("type") != "message":
                    continue
                role = payload.get("role")
                if role not in {"user", "assistant"}:
                    continue
                text = content_text(payload.get("content"))
                if not text:
                    continue
                key = payload.get("id") or (obj.get("timestamp"), role, text)
                if key in seen:
                    continue
                seen.add(key)
                records.append((obj.get("timestamp", ""), role, text, path.name))
    records.sort(key=lambda item: item[0])
    lines = [
        "# 规划 Go2 视觉导航毕业课题会话文本",
        "",
        "本文件由 Codex 原始 JSONL 记录自动提取，仅包含用户和助手可见消息，不包含工具调用和内部记录。",
        "",
    ]
    for timestamp, role, text, source in records:
        label = "用户" if role == "user" else "助手"
        lines.extend([f"## {label}  {timestamp}", "", text, ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"messages={len(records)} output={output}")


if __name__ == "__main__":
    main()
