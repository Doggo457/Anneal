"""Read-only summary of a GGUF file: architecture, key metadata, tensor quant types, MTP / nextn tensors.

Usage: python inspect_gguf.py <path-to-gguf> [--all-keys]
"""

import collections
import sys

from gguf import GGUFReader, GGUFValueType


def value(field):
    if not field.types:
        return None
    if field.types[0] == GGUFValueType.ARRAY:
        n = len(field.data)
        if field.types[-1] == GGUFValueType.STRING:
            items = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data[:8]]
        else:
            items = [field.parts[i].tolist()[0] for i in field.data[:8]]
        return f"[{n} items] {items}{' ...' if n > 8 else ''}"
    part = field.parts[field.data[0]]
    if field.types[0] == GGUFValueType.STRING:
        return bytes(part).decode("utf-8", "replace")
    return part.tolist()[0] if len(part) == 1 else part.tolist()


def main(path, all_keys=False):
    r = GGUFReader(path, "r")
    arch = value(r.fields["general.architecture"]) if "general.architecture" in r.fields else "?"
    print(f"file: {path}\narchitecture: {arch}\ntensors: {len(r.tensors)}")
    for name, field in r.fields.items():
        if name.startswith("GGUF.") or name.startswith("tokenizer.ggml.") and not all_keys:
            continue
        if all_keys or name.startswith("general.") or name.startswith(arch + ".") or "quant" in name or "mtp" in name:
            print(f"  {name} = {value(field)}")

    by_type = collections.Counter()
    bytes_by_type = collections.Counter()
    for t in r.tensors:
        by_type[t.tensor_type.name] += 1
        bytes_by_type[t.tensor_type.name] += int(t.n_bytes)
    total = sum(bytes_by_type.values())
    print("tensor types (count, GiB, share of bytes):")
    for k, n in by_type.most_common():
        print(f"  {k:8s} {n:5d}  {bytes_by_type[k] / 2**30:7.2f}  {100 * bytes_by_type[k] / total:5.1f}%")
    n_params = sum(int(t.n_elements) for t in r.tensors)
    print(f"parameters in file: {n_params / 1e9:.2f} B, {8 * total / n_params:.2f} bits/weight on average")

    mtp = [t for t in r.tensors if any(s in t.name for s in ("nextn", "mtp", ".eh_proj", "enorm", "hnorm",
                                                              "shared_head"))]
    print(f"MTP / nextn tensors: {len(mtp)}")
    for t in mtp[:25]:
        print(f"  {t.name}  {t.tensor_type.name}  {list(t.shape)}")
    blocks = sorted({int(t.name.split(".")[1]) for t in r.tensors if t.name.startswith("blk.")})
    if blocks:
        print(f"blocks: blk.{blocks[0]} .. blk.{blocks[-1]} ({len(blocks)})")
        last = [t.name for t in r.tensors if t.name.startswith(f"blk.{blocks[-1]}.")]
        print(f"  tensors in last block: {last[:20]}")


if __name__ == "__main__":
    main(sys.argv[1], "--all-keys" in sys.argv)
