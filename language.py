"""
language.py — Natural language command parser for FrankaVisionPick.

Sends the user's instruction to Claude and gets back a pick order.

Usage:
    from language import parse_command
    order = parse_command("pick the blue pillar first, skip the yellow block")
    # → [1, 2, 0]   (indices into OBJ_DEFS)
"""

import os
import json

# Object descriptions shown to the LLM
OBJ_DESCRIPTIONS = [
    "0 = red square cube     (4×4×4 cm)",
    "1 = blue tall pillar    (3×3×7 cm)",
    "2 = green flat tile     (5×5×3.5 cm)",
    "3 = yellow medium block (4×4×5.5 cm)",
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
  "pick the red one first"         → {{"order":[0,1,2,3], "reason":"red first, rest default"}}
  "blue pillar then green tile"    → {{"order":[1,2,0,3], "reason":"blue then green as requested"}}
  "skip the yellow block"          → {{"order":[0,1,2],   "reason":"all except yellow"}}
  "tallest first"                  → {{"order":[1,3,0,2], "reason":"sorted by height: 7cm,5.5cm,4cm,3.5cm"}}
  "only pick the green and red"    → {{"order":[2,0],     "reason":"only green and red as requested"}}
"""


def parse_command(text: str, api_key: str = None, default: list = None) -> list:
    """
    Parse a natural language pick instruction and return a list of object indices.

    api_key can be passed directly or read from ANTHROPIC_API_KEY env var.
    Falls back to `default` (= [0,1,2,3]) if the key is missing or the call fails.
    """
    if default is None:
        default = [0, 1, 2, 3]

    key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    if not key:
        print("[LANG] ✗ No API key found.")
        print("[LANG]   Pass it with:  --api-key sk-ant-...")
        print("[LANG]   Or set env var: set ANTHROPIC_API_KEY=sk-ant-...")
        print("[LANG]   Falling back to default order [0,1,2,3].")
        return default

    print(f"[LANG] API key found (****{key[-4:]})")

    try:
        import anthropic
        print("[LANG] Calling Claude Haiku …")
        client = anthropic.Anthropic(api_key=key)

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )

        raw    = msg.content[0].text.strip()
        print(f"[LANG] Raw response: {raw}")
        result = json.loads(raw)
        order  = result["order"]

        # Validate — must be a list of unique ints in 0-3
        if (isinstance(order, list)
                and all(isinstance(x, int) and 0 <= x <= 3 for x in order)
                and len(set(order)) == len(order)):
            print(f"[LANG] ✓ Reason: {result.get('reason', '')}")
            return order

        print(f"[LANG] ✗ Unexpected format — using default order.")
        return default

    except Exception as e:
        print(f"[LANG] ✗ Error: {e}")
        print("[LANG]   Falling back to default order [0,1,2,3].")
        return default
