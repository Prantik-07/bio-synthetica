"""Run locally or in Colab to verify protocol parsing and rewards.

Usage:
  python training/debug_parse.py
  # or from repo root:
  python -m training.debug_parse
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment.bio_synthetica_env import BioSyntheticaEnv, normalize_protocol


def main():
    env = BioSyntheticaEnv()

    cases = [
        (
            "clean protocol",
            '''scan("A1")
scan("B1")
pipette("A1", "B1", volume=100)
report_complete()''',
        ),
        (
            "markdown wrapped",
            '''Here is the code:
```python
scan("A1")
scan("B1")
pipette("A1", "B1", volume=50)
report_complete()
```''',
        ),
        (
            "def wrapped",
            '''def protocol():
    scan("A1")
    scan("B1")
    pipette("A1", "B1", volume=50)
    report_complete()
''',
        ),
        (
            "bad — no scan",
            '''pipette("A1", "B1", volume=50)
report_complete()''',
        ),
    ]

    print("=== normalize_protocol + parse ===\n")
    for name, raw in cases:
        norm = normalize_protocol(raw)
        pr = env.parse_protocol(norm)
        print(f"--- {name} ---")
        print(f"  syntax_pass: {pr['syntax_pass']}  n_actions: {len(pr['actions'])}")
        if pr["actions"]:
            print(f"  first: {pr['actions'][0]}  last: {pr['actions'][-1]}")
        print()

    print("=== env.step rewards ===\n")
    for name, raw in cases:
        env.reset()
        _, reward, _, info = env.step(raw)
        print(f"{name}: reward={reward:.3f}  syntax_pass={info['syntax_pass']}  "
              f"violations={len(info['violations'])}")
    print("\nDone.")


if __name__ == "__main__":
    main()
