"""
language.py — Natural language command parser for FrankaVisionPick.

Supports two backends (auto-detected):
  1. LM Studio local server (free, no key needed)  ← preferred
  2. Anthropic Claude Haiku (cloud, needs --api-key) ← fallback

LM Studio setup:
  - Open LM Studio → Local Server tab → Start Server (port 1234)
  - Load any model (gemma-4, qwen, etc.)

Usage:
    from language import parse_command
    order = parse_command("pick the blue pillar first, skip the yellow block")
    # → [1, 0, 2]
"""

import os
import json
import re

OBJ_DESCRIPTIONS = [
    "0 = red square cube     (4x4x4 cm,  base area 16 cm²)",
    "1 = blue tall pillar    (3x3x7 cm,  base area  9 cm²,  tallest)",
    "2 = green flat tile     (5x5x3.5 cm, base area 25 cm², widest — best stack base)",
    "3 = yellow medium block (4x4x5.5 cm, base area 16 cm²)",
]

_SYSTEM = f"""You control a Franka Panda robot arm that picks objects from Bin A and places them in Bin B.

Objects available:
{chr(10).join(OBJ_DESCRIPTIONS)}

The user gives a natural language instruction about which objects to pick and in what order.

Respond with ONLY a JSON object — no explanation, no markdown, no extra text:
{{
  "order": [list of indices 0-3 in pick order],
  "reason": "one short sentence"
}}

Rules:
- If the user only mentions some objects, append the unmentioned ones at the end in default order (0,1,2,3).
- If the user says to skip / ignore an object, omit it from the list entirely.
- If the instruction is ambiguous, use default order [0,1,2,3].

Examples:
  "pick the red one first"         -> {{"order":[0,1,2,3], "reason":"red first, rest default"}}
  "blue pillar then green tile"    -> {{"order":[1,2,0,3], "reason":"blue then green as requested"}}
  "skip the yellow block"          -> {{"order":[0,1,2],   "reason":"all except yellow"}}
  "tallest first"                  -> {{"order":[1,3,0,2], "reason":"sorted by height: 7cm,5.5cm,4cm,3.5cm"}}
  "only pick the green and red"    -> {{"order":[2,0],     "reason":"only green and red as requested"}}
"""

# LM Studio server address — change port if you moved it
LM_STUDIO_URL = "http://localhost:1234/v1"


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    return text


def _validate(order) -> bool:
    return (
        isinstance(order, list)
        and all(isinstance(x, int) and 0 <= x <= 3 for x in order)
        and len(set(order)) == len(order)
    )


def _call_lmstudio(text: str) -> list | None:
    """Call LM Studio local server (OpenAI-compatible API)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[LANG] openai package not found — run: pip install openai")
        return None

    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

        # Auto-detect whichever model is loaded in LM Studio
        models = client.models.list()
        model_id = models.data[0].id if models.data else "local-model"
        print(f"[LANG] Calling LM Studio ({model_id}) …")

        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": text},
            ],
            max_tokens=512,   # thinking mode needs more tokens
            temperature=0.1,
        )

        msg_dict = resp.choices[0].message.model_dump()
        raw      = msg_dict.get("content") or ""

        # Qwen3 thinking mode puts the chain-of-thought in reasoning_content
        # and sometimes leaves content empty. Extract the last list from reasoning.
        if not raw.strip():
            reasoning = msg_dict.get("reasoning_content") or ""
            print(f"[LANG] content empty — searching reasoning ({len(reasoning)} chars)")
            matches = re.findall(r'\[\s*[\d,\s]+\]', reasoning)
            if matches:
                candidate = matches[-1]
                print(f"[LANG] Extracted from reasoning: {candidate}")
                try:
                    order = json.loads(candidate)
                    if _validate(order):
                        print(f"[LANG] ✓ Order from reasoning: {order}")
                        return order
                except Exception:
                    pass
            print("[LANG] ✗ LM Studio returned empty response and no order in reasoning.")
            return None

        raw = raw.strip()
        print(f"[LANG] Raw response: {raw}")
        raw = _strip_fences(raw)
        result = json.loads(raw)
        order = result["order"]

        if _validate(order):
            print(f"[LANG] ✓ Reason: {result.get('reason', '')}")
            return order

        print("[LANG] ✗ Unexpected format from LM Studio.")
        return None

    except ConnectionRefusedError:
        print("[LANG] ✗ LM Studio server not running.")
        print("[LANG]   Open LM Studio → Local Server → Start Server")
        return None
    except Exception as e:
        print(f"[LANG] ✗ LM Studio error: {e}")
        return None


def _call_anthropic(text: str, api_key: str) -> list | None:
    """Call Anthropic Claude Haiku (cloud fallback)."""
    try:
        import anthropic
    except ImportError:
        print("[LANG] anthropic package not found.")
        return None

    try:
        print("[LANG] Calling Claude Haiku (cloud) …")
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )

        raw = msg.content[0].text.strip()
        print(f"[LANG] Raw response: {raw}")
        raw = _strip_fences(raw)
        result = json.loads(raw)
        order = result["order"]

        if _validate(order):
            print(f"[LANG] ✓ Reason: {result.get('reason', '')}")
            return order

        print("[LANG] ✗ Unexpected format from Anthropic.")
        return None

    except Exception as e:
        print(f"[LANG] ✗ Anthropic error: {e}")
        return None


def parse_command(text: str, api_key: str = None, default: list = None) -> list:
    """
    Parse a natural language pick instruction → list of object indices.

    Priority:
      1. LM Studio local server (no key needed, free)
      2. Anthropic Claude Haiku (if --api-key passed or ANTHROPIC_API_KEY set)
      3. Default order [0,1,2,3]
    """
    if default is None:
        default = [0, 1, 2, 3]

    # 1. Try LM Studio first
    order = _call_lmstudio(text)
    if order is not None:
        return order

    # 2. Fall back to Anthropic if a key is available
    key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    if key:
        print(f"[LANG] Falling back to Anthropic (****{key[-4:]}) …")
        order = _call_anthropic(text, key)
        if order is not None:
            return order

    print("[LANG] All backends failed — using default order [0,1,2,3].")
    return default
