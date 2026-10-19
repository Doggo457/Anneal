"""Read-only inventory of Qwen3.8 GGUF files: Ollama models (via their manifests) plus loose .gguf paths.

Usage: python inventory_qwen.py [extra.gguf ...]
"""

import collections
import glob
import json
import os
import sys

from gguf import GGUFReader, GGUFValueType

# llama.cpp's LLAMA_FTYPE names for general.file_type
FTYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
          12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
          19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
          26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
          36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4"}


def scalar(r, key):
    f = r.fields.get(key)
    if f is None or not f.types:
        return None
    part = f.parts[f.data[0]]
    if f.types[0] == GGUFValueType.STRING:
        return bytes(part).decode("utf-8", "replace")
    return part.tolist()[0]


def describe(path):
    r = GGUFReader(path, "r")
    arch = scalar(r, "general.architecture")
    by_bytes = collections.Counter()
    n_el = 0
    for t in r.tensors:
        by_bytes[t.tensor_type.name] += int(t.n_bytes)
        n_el += int(t.n_elements)
    total = sum(by_bytes.values())
    mix = ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in by_bytes.most_common(3))
    ft = scalar(r, "general.file_type")
    return {
        "name": scalar(r, "general.name"), "arch": arch, "file_type": f"{ft} ({FTYPES.get(ft, '?')})",
        "bpw": round(8 * total / n_el, 2), "params_b": round(n_el / 1e9, 2), "mix": mix,
        "mtp_layers": scalar(r, f"{arch}.nextn_predict_layers"),
        "ctx": scalar(r, f"{arch}.context_length"), "gib": round(os.path.getsize(path) / 2**30, 2),
    }


def ollama_models():
    root = os.path.join(os.path.expanduser("~"), ".ollama", "models")
    for mf in glob.glob(os.path.join(root, "manifests", "registry.ollama.ai", "*", "*", "*")):
        parts = mf.split(os.sep)
        name = f"{parts[-2]}:{parts[-1]}"
        with open(mf, encoding="utf-8") as fh:
            layers = json.load(fh)["layers"]
        for layer in layers:
            if layer["mediaType"] == "application/vnd.ollama.image.model":
                yield f"ollama {name}", os.path.join(root, "blobs", layer["digest"].replace(":", "-"))


if __name__ == "__main__":
    items = list(ollama_models()) + [(f"file {p}", p) for p in sys.argv[1:]]
    for label, path in items:
        try:
            d = describe(path)
        except Exception as exc:
            print(f"{label}\n    could not read: {exc}")
            continue
        print(f"{label}\n    {json.dumps(d)}")
