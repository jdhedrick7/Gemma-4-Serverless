#!/usr/bin/env python3
"""Idempotently register a correct `gemma4` chat template in an installed
SpecForge (no shell-escaping hazards — run as a file, not an inline heredoc).

Gemma-4 turn format (verified live vs the tokenizer):
    <bos><|turn>user\nUUU<turn|>\n<|turn>model\nAAA<turn|>\n
so: assistant_header="<|turn>model\n", user_header="<|turn>user\n",
    end_of_turn_token="<turn|>\n", system_prompt="".
SpecForge's built-in "gemma" (<start_of_turn>/<end_of_turn>) is Gemma-2/3 and
would yield empty loss masks → dead finetune.
"""
from pathlib import Path

TPL = Path("/workspace/SpecForge/specforge/data/template.py")


def main() -> None:
    s = TPL.read_text()
    if 'name="gemma4"' in s:
        print("gemma4 already present")
        return
    anchor = 'TEMPLATE_REGISTRY.register(\n    name="gemma",'
    if anchor not in s:
        raise SystemExit("anchor 'gemma' template not found — inspect template.py")
    block = (
        'TEMPLATE_REGISTRY.register(\n'
        '    name="gemma4",\n'
        '    template=ChatTemplate(\n'
        '        assistant_header="<|turn>model\\n",\n'
        '        user_header="<|turn>user\\n",\n'
        '        system_prompt="",\n'
        '        end_of_turn_token="<turn|>\\n",\n'
        '    ),\n'
        ')\n\n'
    )
    TPL.write_text(s.replace(anchor, block + anchor, 1))

    # verify it loads and round-trips
    import importlib
    import specforge.data.template as m

    importlib.reload(m)
    t = m.TEMPLATE_REGISTRY.get("gemma4")
    assert t.assistant_header == "<|turn>model\n", repr(t.assistant_header)
    assert t.end_of_turn_token == "<turn|>\n", repr(t.end_of_turn_token)
    print("gemma4 REGISTERED + verified:", repr(t.assistant_header), repr(t.end_of_turn_token))


if __name__ == "__main__":
    main()
