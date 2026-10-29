"""Tests for scripts/train_longrun.py adapter_config_mismatch: a saved adapter's r, alpha and target modules against a
run's LoRA settings, with target_modules saved as a list or as Unsloth's single regex. No torch needed.

  python tests/test_init_adapter_config.py        (or: python -m pytest tests/test_init_adapter_config.py)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from train_longrun import adapter_config_mismatch  # noqa: E402

SEVEN = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = {"r": 16, "lora_alpha": 32, "target_modules": SEVEN}
# The form Unsloth wrote into round 2's adapter_config.json (29 Oct 2026): language/text filter plus the projections.
_PARTS = r"(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)"
_PROJ = r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
UNSLOTH_REGEX = (r"(?:.*?(?:language|text).*?" + _PARTS + r".*?\." + _PROJ + r")|(?:\bmodel\.layers\.[\d]{1,}"
                 r"(?!.*\.experts\.\d)\..*?" + _PARTS + r".*?\." + _PROJ + ")")


def test_unsloth_regex_with_the_same_projections_matches():
    ac = {"r": 16, "lora_alpha": 32, "target_modules": UNSLOTH_REGEX}
    assert adapter_config_mismatch(ac, LORA) is None


def test_list_in_any_order_matches():
    ac = {"r": 16, "lora_alpha": 32, "target_modules": list(reversed(SEVEN))}
    assert adapter_config_mismatch(ac, LORA) is None


def test_regex_missing_a_projection_is_refused():
    ac = {"r": 16, "lora_alpha": 32, "target_modules": UNSLOTH_REGEX.replace("|down_proj", "")}
    have, want = adapter_config_mismatch(ac, LORA)
    assert "down_proj" not in have["target_modules"] and "down_proj" in want["target_modules"]


def test_name_inside_a_longer_name_does_not_count():
    # "o_proj" must not be found inside "x_o_proj_y"-style longer names
    ac = {"r": 16, "lora_alpha": 32, "target_modules": UNSLOTH_REGEX.replace("|o_proj", "|out_o_proj2")}
    have, _ = adapter_config_mismatch(ac, LORA)
    assert "o_proj" not in have["target_modules"]


def test_rank_or_alpha_change_is_refused():
    assert adapter_config_mismatch({"r": 64, "lora_alpha": 32, "target_modules": UNSLOTH_REGEX}, LORA)
    assert adapter_config_mismatch({"r": 16, "lora_alpha": 16, "target_modules": SEVEN}, LORA)


def test_list_with_an_extra_module_is_refused():
    ac = {"r": 16, "lora_alpha": 32, "target_modules": SEVEN + ["in_proj_qkvz"]}
    assert adapter_config_mismatch(ac, LORA)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
