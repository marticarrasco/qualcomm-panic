"""Which tensors does each fine-tuned GGUF change with respect to the stock one, and with respect to each other?

A minimal GGUF reader with the standard library only (read-only).
Usage, from the Datathon folder:  <venv python> -X utf8 mesures/ft_gguf_diff.py
"""
import hashlib
import mmap
import os
import re
import struct
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.abspath(os.path.join(HERE, ".."))
FILES = {
    "stock": os.path.join(TOP, "models", "Qwen_Qwen3.5-0.8B-Q8_0.gguf"),
    "ft1": os.path.join(TOP, "Qwen3.5-0.8B-M3Kang-PoT-Q8_0.gguf"),
    "ft2": os.path.join(TOP, "Qwen3.5-0.8B-M3Kang-PoT-v2-Q8_0.gguf"),
    "ft3": os.path.join(TOP, "Qwen3.5-0.8B-M3Kang-PoT-v3-Q8_0.gguf"),
}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read(path):
    f = open(path, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, m, pos)[0]
        pos += struct.calcsize(fmt)
        return v

    def string():
        nonlocal pos
        n = take("<Q")
        s = m[pos:pos + n].decode("utf-8", "replace")
        pos += n
        return s

    def value(t):
        if t in SCALAR:
            return take(SCALAR[t])
        if t == 8:
            return string()
        if t == 9:
            et, n = take("<I"), take("<Q")
            out = [value(et) for _ in range(n)]
            return out if n <= 16 else f"<array of {n}>"
        raise ValueError(f"unknown type {t}")

    assert m[:4] == b"GGUF", path
    pos = 4
    version, n_tensors, n_kv = take("<I"), take("<Q"), take("<Q")
    kv = {}
    for _ in range(n_kv):
        key = string()
        kv[key] = value(take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        infos.append((name, tuple(dims), take("<I"), take("<Q")))
    align = kv.get("general.alignment", 32)
    base = (pos + align - 1) // align * align
    order = sorted(infos, key=lambda t: t[3])
    ends = [t[3] for t in order[1:]] + [len(m) - base]
    tensors = {}
    for (name, dims, ttype, off), end in zip(order, ends):
        tensors[name] = (dims, ttype, hashlib.sha1(m[base + off:base + end]).hexdigest())
    return kv, tensors


data = {tag: read(path) for tag, path in FILES.items()}
for tag, (kv, tensors) in data.items():
    print(f"{tag}: {len(tensors)} tensors, architecture {kv.get('general.architecture')}, "
          f"block count {next((v for k, v in kv.items() if k.endswith('.block_count')), '?')}, "
          f"quantization types {dict(Counter(t[1] for t in tensors.values()))}")


def kind(name):
    return re.sub(r"\d+", "N", name)


def diff(a, b):
    ta, tb = data[a][1], data[b][1]
    shared = set(ta) & set(tb)
    changed = sorted(n for n in shared if ta[n][2] != tb[n][2])
    print(f"\n{a} vs {b}: tensors only in {a} {len(set(ta) - set(tb))}, only in {b} {len(set(tb) - set(ta))}, "
          f"shared {len(shared)}, with different bytes {len(changed)}")
    by_kind = Counter(kind(n) for n in changed)
    same_kind = Counter(kind(n) for n in shared if ta[n][2] == tb[n][2])
    for k, c in sorted(by_kind.items()):
        print(f"   changed {c:3d} (same {same_kind.get(k, 0):3d})  {k}")
    untouched = sorted(k for k in same_kind if k not in by_kind)
    print("   kinds never changed:", untouched)
    return set(changed)


c1, c2, c3 = diff("stock", "ft1"), diff("stock", "ft2"), diff("stock", "ft3")
print(f"\nchanged by ft1 and not by ft2: {len(c1 - c2)} | by ft2 and not by ft1: {len(c2 - c1)} | "
      f"by ft3 and not by ft2: {len(c3 - c2)} | by ft2 and not by ft3: {len(c2 - c3)}")
for n in sorted(c1 ^ c2)[:20]:
    print("   differs in the set:", n)

# Fine-tuned files against each other: a tensor with the same bytes in two of them was trained in neither.
print("\nTensor types (0 = F32, 8 = Q8_0) of the kinds that differ from stock:")
for k in ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_a", "blk.0.attn_qkv.weight", "blk.0.ffn_up.weight"):
    print("  ", k, {tag: (data[tag][1][k][0], data[tag][1][k][1]) for tag in data if k in data[tag][1]})
type_diff = Counter(kind(n) for n in data["stock"][1] if data["stock"][1][n][1] != data["ft2"][1][n][1])
print("   kinds stored with another type in ft2 than in stock:", dict(type_diff))
for a, b in (("ft1", "ft2"), ("ft2", "ft3")):
    ta, tb = data[a][1], data[b][1]
    same = Counter(kind(n) for n in ta if ta[n][2] == tb[n][2])
    other = Counter(kind(n) for n in ta if ta[n][2] != tb[n][2])
    print(f"\n{a} vs {b}: different bytes in {sum(other.values())} tensors: {dict(sorted(other.items()))}")
    print(f"   identical in both: {dict(sorted((k, c) for k, c in same.items() if k.startswith('blk.N.ssm') or k.startswith('blk.N.attn') or k.startswith('blk.N.ffn')))}")
