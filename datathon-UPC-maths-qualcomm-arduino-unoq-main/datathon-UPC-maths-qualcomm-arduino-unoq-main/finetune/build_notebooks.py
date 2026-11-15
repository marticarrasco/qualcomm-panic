"""Write the Kaggle notebooks of this folder from the cell lists below.

Usage, on the laptop:  python finetune/build_notebooks.py
The notebooks hold no M3Kang data, so they can be committed.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def notebook(cells):
    out = []
    for kind, source in cells:
        lines = source.strip("\n").split("\n")
        text = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": text}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells):
    path = HERE / name
    path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


# Shared by both notebooks: llama.cpp at the board's commit, and the server flags of the board.
COMMON = r'''
import hashlib, json, os, subprocess, sys, time, urllib.request

WORK = "/kaggle/working"
LLAMA = f"{WORK}/llama.cpp"
LLAMA_REF = "2cbe0ec429b849a88664e0f62388fc92a5abd699"   # config.env: LLAMA_CPP_REF
BASE_REPO = "Qwen/Qwen3.5-0.8B"
REF_REPO = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"            # config.env: LLM_REPO
REF_REVISION = "f36b1ea49a332ede8fe5f389bbf5b3575ef71f48"
REF_FILE = "Qwen_Qwen3.5-0.8B-Q8_0.gguf"
REF_SHA256 = "7182e2362766bb9569209bbc24cf1a4cdfbb8ab161babdb2080c84fa62c08c2f"
SERVER = f"{LLAMA}/build/bin/llama-server"
PORT = 8091
os.chdir(WORK)


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {r.returncode}: {cmd}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def post(path, payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as resp:
        return json.loads(resp.read())


class Server:
    """llama-server with the flags of board/systemd/math-llm.service."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.log = open(f"{WORK}/server.log", "w")
        self.proc = subprocess.Popen(
            [SERVER, "-m", self.model, "--host", "127.0.0.1", "--port", str(PORT), "--threads", "4",
             "--ctx-size", "4096", "--parallel", "1", "--cache-ram", "0", "--ctx-checkpoints", "2",
             "--jinja", "--reasoning", "off", "--reasoning-budget", "0", "--device", "none", "-ngl", "0"],
            stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(180):
            if self.proc.poll() is not None:
                break
            try:
                if get("/health").get("status") == "ok":
                    return self
            except Exception:
                time.sleep(1)
        self.__exit__()
        print(open(f"{WORK}/server.log").read()[-3000:])
        raise RuntimeError(f"llama-server did not start with {self.model}")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
'''

GET_LLAMA = r'''
# llama.cpp at the commit every board runs. The build runs in the background while the next cells download.
if not os.path.exists(f"{LLAMA}/convert_hf_to_gguf.py"):
    sh(f"mkdir -p {LLAMA} && cd {LLAMA} && git init -q && "
       f"git remote add origin https://github.com/ggml-org/llama.cpp.git && "
       f"git fetch -q --depth 1 origin {LLAMA_REF} && git checkout -q FETCH_HEAD")
sh(f"cd {LLAMA} && git log -1 --format=%h")
sh("which cmake || pip install -q cmake")
build = subprocess.Popen(
    f"cd {LLAMA} && cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > {WORK}/build.log 2>&1 && "
    f"cmake --build build --config Release -j4 --target llama-server >> {WORK}/build.log 2>&1", shell=True)
'''

WAIT_BUILD = r'''
print("waiting for the llama-server build...", flush=True)
if build.wait() != 0 or not os.path.exists(SERVER):
    print(open(f"{WORK}/build.log").read()[-4000:])
    raise RuntimeError("the llama.cpp build failed")
sh(f"{SERVER} --version 2>&1 | head -2")
'''

GATE0 = [
    ("markdown", r'''
# Gate 0: can the board's llama.cpp convert Qwen3.5 0.8B?

Before training anything, this notebook converts the **unmodified** `Qwen/Qwen3.5-0.8B` checkpoint to a Q8_0 GGUF
with `convert_hf_to_gguf.py` at the commit pinned in `config.env`, and compares the result with the GGUF the board
runs today (`bartowski/Qwen_Qwen3.5-0.8B-GGUF`). If the two files behave the same, a fine-tuned checkpoint can
take the same path.

**Kaggle settings:** Internet **on**. No GPU and no dataset needed. Run all the cells (about 15 minutes) and read
the `VERDICT` block at the end.
'''),
    ("code", COMMON),
    ("code", GET_LLAMA),
    ("code", r'''
# The converter needs the gguf package of the same commit, and a transformers that knows the Qwen3.5 tokenizer.
import numpy as np
sh(f"pip install -q -U 'transformers>=5' sentencepiece protobuf huggingface_hub 'numpy=={np.__version__}'")
sh(f"pip install -q --no-deps {LLAMA}/gguf-py")
import importlib.metadata as md
print({p: md.version(p) for p in ("transformers", "gguf", "torch", "numpy", "huggingface_hub")})
'''),
    ("code", r'''
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

base_sha = HfApi().model_info(BASE_REPO).sha
print("base revision:", base_sha)
snapshot_download(BASE_REPO, revision=base_sha, local_dir=f"{WORK}/base")
sh(f"ls -la {WORK}/base")
print(json.dumps({k: v for k, v in json.load(open(f"{WORK}/base/config.json")).items()
                  if k in ("architectures", "model_type", "transformers_version")}, indent=1))
'''),
    ("code", r'''
# The conversion itself. --outtype q8_0 writes Q8_0 directly, so llama-quantize is not needed.
OURS = f"{WORK}/ours-q8_0.gguf"
sh(f"cd {WORK} && python {LLAMA}/convert_hf_to_gguf.py base --outtype q8_0 --outfile {OURS} 2>&1 | tail -25")
print("converted:", os.path.getsize(OURS) // (1 << 20), "MB")
'''),
    ("code", r'''
REF = hf_hub_download(REF_REPO, REF_FILE, revision=REF_REVISION, local_dir=f"{WORK}/ref")
ref_ok = sha256(REF) == REF_SHA256
print("reference:", os.path.getsize(REF) // (1 << 20), "MB, SHA256 matches config.env:", ref_ok)
'''),
    ("code", r'''
# 1. Compare what is inside the two files.
from collections import Counter
from gguf import GGUFReader


def value(field):
    try:
        return field.contents()
    except Exception:
        return None


def describe(path):
    reader = GGUFReader(path)
    kv = {name: value(field) for name, field in reader.fields.items() if not name.startswith("GGUF.")}
    tensors = {t.name: (tuple(int(x) for x in t.shape), t.tensor_type.name,
                        hashlib.sha1(t.data.tobytes()).hexdigest()) for t in reader.tensors}
    return kv, tensors


kv_ours, t_ours = describe(OURS)
kv_ref, t_ref = describe(REF)
report = {"base_revision": base_sha}

report["architecture"] = (kv_ours.get("general.architecture"), kv_ref.get("general.architecture"))
print("architecture (ours, reference):", report["architecture"])
print("tensors (ours, reference):", len(t_ours), len(t_ref))
only_ours, only_ref = sorted(set(t_ours) - set(t_ref)), sorted(set(t_ref) - set(t_ours))
print("tensor names only in ours:", only_ours[:10], "| only in the reference:", only_ref[:10])
shared = sorted(set(t_ours) & set(t_ref))
shape_diff = [n for n in shared if t_ours[n][0] != t_ref[n][0]]
type_diff = [n for n in shared if t_ours[n][1] != t_ref[n][1]]
same_bytes = sum(t_ours[n][2] == t_ref[n][2] for n in shared)
print("types in ours:     ", dict(Counter(t[1] for t in t_ours.values())))
print("types in reference:", dict(Counter(t[1] for t in t_ref.values())))
print(f"shared tensors {len(shared)}: different shape {len(shape_diff)}, different type {len(type_diff)}, "
      f"byte-identical {same_bytes}")
for n in type_diff[:12]:
    print("   type differs:", n, t_ours[n][1], "vs", t_ref[n][1])
report["tensors"] = {"ours": len(t_ours), "reference": len(t_ref), "only_ours": len(only_ours),
                     "only_reference": len(only_ref), "shape_diff": len(shape_diff),
                     "type_diff": len(type_diff), "byte_identical": same_bytes}

skip = ("general.", "tokenizer.chat_template")
kv_diff = [k for k in sorted(set(kv_ours) | set(kv_ref)) if not k.startswith(skip) and kv_ours.get(k) != kv_ref.get(k)]
print("\nmetadata keys that differ (general.* left out):", len(kv_diff))
for k in kv_diff[:25]:
    a, b = kv_ours.get(k, "<missing>"), kv_ref.get(k, "<missing>")
    if isinstance(a, list) or isinstance(b, list):
        la = len(a) if isinstance(a, list) else a
        lb = len(b) if isinstance(b, list) else b
        print(f"   {k}: lists of length {la} vs {lb}")
    else:
        print(f"   {k}: {str(a)[:70]!r} vs {str(b)[:70]!r}")
report["metadata_diff"] = kv_diff
tpl_same = kv_ours.get("tokenizer.chat_template") == kv_ref.get("tokenizer.chat_template")
print("chat template identical:", tpl_same)
report["chat_template_identical"] = tpl_same
'''),
    ("code", WAIT_BUILD),
    ("code", r'''
# 2. Compare how the two files answer, with the board's server flags, temperature 0 and seed 42.
PROBLEMS = [
    "How many whole numbers from 1 to 100 are divisible by 4 but not by 8?",
    "Anna has 23 stickers. She gives 5 to each of her 3 friends. How many stickers does she have left?",
    "What is the last digit of 7 x 7 x 7 x 7 x 7?",
    "A rectangle has a perimeter of 30 cm and its length is twice its width. What is its area in square centimetres?",
    "Quants nombres de dues xifres tenen la suma de les xifres igual a 9?",
    "En Marc té 12 anys i la seva germana en té 4. D'aquí a quants anys en Marc tindrà el doble d'anys que ella?",
    "Quin és el nombre més petit més gran que 50 que és múltiple de 6 i de 8?",
    "Una granota puja 3 m cada dia i en baixa 1 cada nit. Quants dies triga a sortir d'un pou de 9 m?",
]
OPTIONS = "\nA) 9\nB) 10\nC) 12\nD) 13\nE) 15\nAnswer:"
POT_SYSTEM = ("You solve math problems by writing a short, flat Python script (no functions, no comments, no input). "
              "Always reply with a ```python code block that prints one number: the final answer.")
POT_INSTRUCTION = "\nWrite a short Python program (no def, no comments) that prints the answer."
LETTER_SYSTEM = ("You solve multiple-choice math competition problems for school students. "
                 "Read the problem and the options, then reply with the letter of the correct option.")


def ask(messages, **extra):
    r = post("/v1/chat/completions", {"messages": messages, "temperature": 0, "seed": 42, **extra})
    t = r.get("timings", {})
    return r["choices"][0]["message"]["content"], (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)


def behave(model):
    rows = []
    with Server(model):
        for text in PROBLEMS:
            code, n1 = ask([{"role": "system", "content": POT_SYSTEM},
                            {"role": "user", "content": text + POT_INSTRUCTION}], max_tokens=120)
            letter, n2 = ask([{"role": "system", "content": LETTER_SYSTEM},
                              {"role": "user", "content": text + OPTIONS}], max_tokens=1, grammar="root ::= [A-E]")
            rows.append({"code": code, "code_prompt_tokens": n1, "letter": letter, "letter_prompt_tokens": n2})
    return rows


b_ours, b_ref = behave(OURS), behave(REF)
n = len(PROBLEMS)
same_tokens = sum(a["code_prompt_tokens"] == b["code_prompt_tokens"] and
                  a["letter_prompt_tokens"] == b["letter_prompt_tokens"] for a, b in zip(b_ours, b_ref))
same_code = sum(a["code"] == b["code"] for a, b in zip(b_ours, b_ref))
same_letter = sum(a["letter"] == b["letter"] for a, b in zip(b_ours, b_ref))
print(f"same prompt token counts: {same_tokens}/{n}   same program text: {same_code}/{n}   same letter: {same_letter}/{n}")
for text, a, b in zip(PROBLEMS, b_ours, b_ref):
    if a["code"] != b["code"]:
        print("\n---", text, "\nOURS:\n" + a["code"], "\nREFERENCE:\n" + b["code"])
report["behaviour"] = {"n": n, "same_prompt_tokens": same_tokens, "same_program": same_code, "same_letter": same_letter}
'''),
    ("code", r'''
t = report["tensors"]
checks = {
    "the reference file is the one in config.env": ref_ok,
    "same architecture": report["architecture"][0] == report["architecture"][1],
    "same tensor names and shapes": t["only_ours"] == 0 and t["only_reference"] == 0 and t["shape_diff"] == 0,
    "same chat template": report["chat_template_identical"],
    "same prompt token counts": report["behaviour"]["same_prompt_tokens"] == n,
    "same letter on at least 7 of 8": report["behaviour"]["same_letter"] >= n - 1,
    "same program text on at least 5 of 8": report["behaviour"]["same_program"] >= 5,
}
report["checks"] = checks
json.dump(report, open(f"{WORK}/gate0_report.json", "w"), indent=1)
print("=" * 60, "\nVERDICT")
for name, ok in checks.items():
    print("  PASS " if ok else "  FAIL ", name)
print(f"  tensors byte-identical: {t['byte_identical']} of {t['ours']}; tensor types that differ: {t['type_diff']}; "
      f"metadata keys that differ: {len(report['metadata_diff'])}")
print("GATE 0:", "PASS - the fine-tuned checkpoint can take this path" if all(checks.values())
      else "CHECK - copy this block and the output of the two cells above")
print("=" * 60)
'''),
]


TRAIN = [
    ("markdown", r'''
# Train Qwen3.5 0.8B to write the program, and turn it into the board's GGUF

M3Kang as a training set: each example is a problem statement (English or Catalan) and a short Python program that
was **run and checked against the labelled answer** (`finetune/build_pairs.py`). The model learns to write the
program; on the board, Python computes the answer.

What this notebook does:

1. LoRA fine-tuning of `Qwen/Qwen3.5-0.8B` (r=16, alpha=32, 3 epochs, lr 2e-4, loss on the program only).
   The prompts are rendered by the board's own `llama-server`, so training and inference see the same text.
2. Merges the adapter and converts to Q8_0 GGUF with `convert_hf_to_gguf.py` at the commit pinned in `config.env`.
3. Measures the new GGUF against the stock one on the practice set, with the board's server flags.
4. Optionally publishes the GGUF on Hugging Face and prints the lines for `config.env`.

**Kaggle settings:** Accelerator **GPU T4**, Internet **on**, and add as input the private dataset made by
`finetune/make_kaggle_dataset.py` (`pot_pairs.jsonl`, `demo.jsonl`, `val.jsonl`, `pot.py`). Run all: about 30 minutes.
'''),
    ("code", COMMON + r'''
import glob, random, shutil
from collections import Counter

FT_FILE = "Qwen3.5-0.8B-M3Kang-PoT-v2-Q8_0.gguf"
FT = f"{WORK}/{FT_FILE}"
MERGED = f"{WORK}/merged"
EPOCHS, LR, RANK, ALPHA, BATCH, ACCUM = 2, 1e-4, 16, 32, 4, 4
HF_REPO = ""   # to publish at the end: "your-user/Qwen3.5-0.8B-M3Kang-PoT-GGUF", and add the Kaggle secret HF_TOKEN

hits = glob.glob("/kaggle/input/**/pot_pairs.jsonl", recursive=True)
assert hits, "Add the dataset with pot_pairs.jsonl as an input of this notebook (Add Input, top right)."
DATA = os.path.dirname(hits[0])
sys.path.insert(0, DATA)
import pot   # copilot/pot.py: the prompts and the sandbox the board uses


def load(name):
    with open(f"{DATA}/{name}", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


pairs, demo, val = load("pot_pairs.jsonl"), load("demo.jsonl"), load("val.jsonl")
print(f"pairs {len(pairs)} ({len({r['id'] for r in pairs})} problems), practice set {len(demo)}, val {len(val)}")
'''),
    ("code", GET_LLAMA),
    ("code", r'''
import numpy as np
sh(f"pip install -q -U 'transformers>=5' safetensors sentencepiece protobuf huggingface_hub 'numpy=={np.__version__}'")
sh(f"pip install -q --no-deps {LLAMA}/gguf-py")
import importlib.metadata as md
print({p: md.version(p) for p in ("transformers", "gguf", "torch", "numpy")})
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
'''),
    ("code", r'''
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

base_sha = HfApi().model_info(BASE_REPO).sha
snapshot_download(BASE_REPO, revision=base_sha, local_dir=f"{WORK}/base")
REF = hf_hub_download(REF_REPO, REF_FILE, revision=REF_REVISION, local_dir=f"{WORK}/ref")
assert sha256(REF) == REF_SHA256, "the reference GGUF is not the one in config.env"
print("base revision:", base_sha)
'''),
    ("code", r'''
import math, torch, torch.nn as nn, transformers
import transformers.utils.import_utils as _iu
_iu._scipy_available = False

torch.manual_seed(42)
random.seed(42)
tok = transformers.AutoTokenizer.from_pretrained(f"{WORK}/base")
model = None
for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
    cls = getattr(transformers, name, None)
    if cls is None:
        continue
    try:
        model = cls.from_pretrained(f"{WORK}/base", dtype=torch.float32)
        print("loaded with", name, "->", type(model).__name__)
        break
    except Exception as error:
        print(name, "failed:", repr(error)[:300])
assert model is not None, "no transformers class could load the checkpoint"
model.requires_grad_(False)


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((self.drop(x.float()) @ self.lora_A.T) @ self.lora_B.T).to(x.dtype) * self.scale

    def merge(self):
        with torch.no_grad():
            self.base.weight.add_((self.lora_B @ self.lora_A).to(self.base.weight.dtype), alpha=self.scale)
        return self.base


# LoRA on every main linear layer of the language model. The vision tower, output head and Gated DeltaNet alpha/beta decay gates stay as they are.
SKIP = ("visual", "vision", "lm_head", "merger", "projector", "embed", "in_proj_a", "in_proj_b", "in_proj_ba")
targets = [n for n, m in model.named_modules()
           if isinstance(m, nn.Linear) and not any(s in n.lower() for s in SKIP)]
for name in targets:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, LoRALinear(getattr(parent, attr), RANK, ALPHA))
model.to("cuda")
N_TARGETS = len(targets)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(len(targets), "linear layers get LoRA:", dict(Counter(n.split(".")[-1] for n in targets)))
print("for example:", targets[:3])
print(f"trainable params: {n_trainable:,} || all params: {n_total:,} || trainable%: {100 * n_trainable / n_total:.2f}%")
'''),
    ("code", WAIT_BUILD),
    ("code", r'''
# The prompt of each example, exactly as the board's server builds it (--jinja --reasoning off).
END = "<|im_end|>"
assert len(tok.encode(END, add_special_tokens=False)) == 1


def hf_prompt(row):
    return tok.apply_chat_template(pot.build_messages(row, "finetuned"), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


prompts, source = None, "the Hugging Face chat template"
with Server(REF):
    try:
        prompts = [post("/apply-template", {"messages": pot.build_messages(r, "finetuned")})["prompt"] for r in pairs]
        source = "llama-server /apply-template"
        same_ids = 0
        for p in prompts[:20]:
            theirs = post("/tokenize", {"content": p, "add_special": False, "parse_special": True})["tokens"]
            same_ids += theirs == tok(p, add_special_tokens=False).input_ids
        print(f"tokenizer check: llama.cpp and Hugging Face give the same ids on {same_ids}/20 prompts")
    except Exception as error:
        print("could not render with llama-server:", repr(error)[:300])
if prompts is None:
    prompts = [hf_prompt(r) for r in pairs]
print("prompts rendered by", source)
print("same as the Hugging Face template:", sum(p == hf_prompt(r) for p, r in zip(prompts[:50], pairs[:50])), "of 50")
print("--- one example ---\n" + prompts[0] + pairs[0]["code"] + END)


def encode(prompt, code):
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(code + END, add_special_tokens=False).input_ids
    return p + c, [-100] * len(p) + c   # the loss only counts the program


ids = sorted({r["id"] for r in pairs})
dev_ids = set(ids[::15])                # both languages of a problem stay on the same side
train = [encode(p, r["code"]) for p, r in zip(prompts, pairs) if r["id"] not in dev_ids]
dev = [encode(p, r["code"]) for p, r in zip(prompts, pairs) if r["id"] in dev_ids]
lengths = sorted(len(x) for x, _ in train)
print(f"train {len(train)}, dev {len(dev)}; tokens per example: median {lengths[len(lengths) // 2]}, max {lengths[-1]}; "
      f"program tokens: mean {sum(sum(t != -100 for t in y) for _, y in train) / len(train):.0f}")
'''),
    ("code", r'''
import math

pad = tok.pad_token_id if tok.pad_token_id is not None else 0


def batch(items):
    n = max(len(x) for x, _ in items)
    ids_t = torch.full((len(items), n), pad, dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    for i, (x, y) in enumerate(items):
        ids_t[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids_t.cuda(), mask.cuda(), labels.cuda()


def loss_of(items):
    ids_t, mask, labels = batch(items)
    logits = model(input_ids=ids_t, attention_mask=mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    active = shift_labels != -100
    return torch.nn.functional.cross_entropy(shift_logits[active].float(), shift_labels[active])


BATCH, ACCUM = 2, 8
torch.cuda.empty_cache()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
updates = EPOCHS * math.ceil(len(train) / (BATCH * ACCUM))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / updates))
t0, step = time.time(), 0
best_dev, best_step, best_lora = float("inf"), 0, None
for epoch in range(EPOCHS):
    random.shuffle(train)
    model.train()
    running = []
    for i in range(0, len(train), BATCH):
        loss = loss_of(train[i:i + BATCH])
        (loss / ACCUM).backward()
        running.append(loss.item())
        loss = None
        step += 1
        if step % ACCUM == 0 or i + BATCH >= len(train):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if step % 140 == 0 or i + BATCH >= len(train):
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                dev_loss = [loss_of(dev[j:j + BATCH]).item() for j in range(0, len(dev), BATCH)]
            mean_dev = sum(dev_loss) / max(len(dev_loss), 1)
            marker = ""
            if mean_dev < best_dev:
                best_dev, best_step = mean_dev, step
                best_lora = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                marker = " *BEST*"
            print(f"ep {epoch + 1} step {step}: train {sum(running[-70:]) / len(running[-70:]):.3f} | dev "
                  f"{mean_dev:.3f}{marker} ({time.time() - t0:.0f} s)", flush=True)
            torch.cuda.empty_cache()
            model.train()
if best_lora is not None:
    for n, p in model.named_parameters():
        if n in best_lora:
            p.data.copy_(best_lora[n].to(p.device))
    print(f"Restored best LoRA checkpoint from step {best_step} (dev loss {best_dev:.3f})", flush=True)
'''),
    ("code", r'''
# A look at what it writes for three held-out problems, then merge the adapter into the weights.
model.eval()
try:
    for x, y in dev[:3]:
        n = sum(t == -100 for t in y)
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([x[:n]]).cuda(), max_new_tokens=100, do_sample=False)
        print("--- written:\n" + tok.decode(out[0][n:], skip_special_tokens=True))
        print("--- reference:\n" + tok.decode(x[n:], skip_special_tokens=True))
except Exception as error:
    print("generation check skipped:", repr(error)[:300])

for name in targets:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, getattr(parent, attr).merge())
state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
del model
torch.cuda.empty_cache()

# Write the merged weights in the layout of the original checkpoint, the one the converter is known to read:
# same files, same tensor names. Only the tensors LoRA changed are replaced (stored as float32).
from safetensors.torch import load_file, save_file

shutil.rmtree(MERGED, ignore_errors=True)
os.makedirs(MERGED)
changed, missing = 0, []
for name in sorted(os.listdir(f"{WORK}/base")):
    src = f"{WORK}/base/{name}"
    if os.path.isdir(src):
        continue
    if not name.endswith(".safetensors"):
        shutil.copy(src, f"{MERGED}/{name}")
        continue
    out = {}
    for key, old in load_file(src).items():
        new = state.get(key)
        if new is None:
            alts = [k for k in state if k.endswith("." + key) or key.endswith("." + k)]
            new = state[alts[0]] if len(alts) == 1 else None
        if new is None or tuple(new.shape) != tuple(old.shape):
            missing.append(key)
            out[key] = old
        elif torch.equal(new.to(torch.float32), old.to(torch.float32)):
            out[key] = old
        else:
            out[key] = new.to(torch.float32).contiguous()
            changed += 1
    save_file(out, f"{MERGED}/{name}", metadata={"format": "pt"})
print(f"tensors replaced: {changed} (LoRA layers: {N_TARGETS}); tensors of the checkpoint not found in the model: {len(missing)}")
print("not found, first ones:", missing[:8])
assert changed == N_TARGETS, "the merged weights don't line up with the checkpoint: stop here and report this output"
del state
'''),
    ("code", r'''
sh(f"cd {WORK} && python {LLAMA}/convert_hf_to_gguf.py merged --outtype q8_0 --outfile {FT} 2>&1 | tail -5")
FT_SHA = sha256(FT)
print(f"{FT_FILE}: {os.path.getsize(FT) // (1 << 20)} MB, sha256 {FT_SHA}")
'''),
    ("code", r'''
# Measure: the model writes a program for each text-only problem with five numeric options, the sandbox runs it,
# and the answer counts when the result equals exactly one option. Same server flags as the board.
def evaluate(model_path, rows, style, max_tokens=140):
    out = []
    with Server(model_path):
        for r in rows:
            if pot.numeric_options(r) is None or not r.get("text_only"):
                continue
            resp = post("/v1/chat/completions", {"messages": pot.build_messages(r, style), "max_tokens": max_tokens,
                                                 "temperature": 0, "seed": 42, "cache_prompt": True})
            t = resp.get("timings", {})
            code = pot.extract_code(resp["choices"][0]["message"]["content"])
            letters, printed, status = pot.run_and_match(code, r)
            out.append({"id": r["id"], "language": r["language"], "level": r["level"], "label": r["label"],
                        "status": status, "printed": printed, "matches": letters,
                        "finish": resp["choices"][0].get("finish_reason"), "read_n": t.get("prompt_n") or 0,
                        "cache_n": t.get("cache_n") or 0, "predicted_n": t.get("predicted_n") or 0, "code": code})
    return out


def summary(name, rows, out):
    matched = [o for o in out if len(o["matches"]) == 1]
    right = [o for o in matched if o["matches"][0] == o["label"]]
    answered = {(o["id"], o["language"]): o["matches"][0] for o in matched}
    score = sum(answered.get((r["id"], r["language"]), "D") == r["label"] for r in rows)
    board = [o["read_n"] / 11 + o["predicted_n"] / 5.5 for o in out[1:]] or [0]
    by_lang = {lang: f"{sum(o['language'] == lang for o in right)}/{sum(o['language'] == lang for o in out)}"
               for lang in ("eng", "cat")}
    print(f"{name}\n  asked {len(out)} of {len(rows)} rows; program printed something {sum(o['printed'] is not None for o in out)}; "
          f"one option matched {len(matched)}; RIGHT {len(right)}; wrong match {len(matched) - len(right)}\n"
          f"  right by language {by_lang}; cut at max_tokens {sum(o['finish'] == 'length' for o in out)}; "
          f"tokens written: mean {sum(o['predicted_n'] for o in out) / max(len(out), 1):.0f}; "
          f"tokens read: mean {sum(o['read_n'] for o in out[1:]) / max(len(out) - 1, 1):.0f}\n"
          f"  estimated board time per problem: mean {sum(board) / len(board):.0f} s, worst {max(board):.0f} s, "
          f"over 50 s: {sum(b > 50 for b in board)}\n"
          f"  score on all {len(rows)} rows, answering D when nothing matches: {score} ({100 * score / len(rows):.1f}%)")
    return len(right), score


results = {}
for name, path, style in (("fine-tuned GGUF (v3)", FT, "finetuned"), ("stock GGUF, three examples in the prompt", REF, "fewshot")):
    out = evaluate(path, demo, style)
    with open(f"{WORK}/eval_demo_{style}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    results[style] = summary(f"PRACTICE SET (Pass@1) - {name}", demo, out)
    if style == "finetuned":
        out_v3 = out

# Also measure Pass@3 conditional with anti-'1' filter on fine-tuned GGUF
by_key = {(r["id"], r["language"]): r for r in demo}
upgraded_v3 = []
with Server(FT):
    for o in out_v3:
        r = by_key[(o["id"], o["language"])]
        board_s = o["read_n"] / 11 + o["predicted_n"] / 5.5
        cur = dict(o, tries=1, board_s=board_s)
        for attempt, seed in ((2, 43), (3, 44)):
            if len(cur["matches"]) == 1 or cur["board_s"] > 42:
                break
            resp = post("/v1/chat/completions", {"messages": pot.build_messages(r, "finetuned"), "max_tokens": 140,
                                                 "temperature": 0.45, "top_p": 0.9, "seed": seed, "cache_prompt": True})
            t = resp.get("timings", {})
            code = pot.extract_code(resp["choices"][0]["message"]["content"])
            letters, printed, status = pot.run_and_match(code, r)
            cur["board_s"] += (t.get("predicted_n") or 0) / 5.5
            cur["tries"] = attempt
            if len(letters) == 1:
                if (printed or "").strip() == "1" and o["printed"] in (None, "", "0"):
                    continue
                cur.update({"matches": letters, "printed": printed, "status": status, "code": code})
                break
        upgraded_v3.append(cur)
summary("PRACTICE SET (Pass@3 conditional) - fine-tuned GGUF (v3)", demo, upgraded_v3)
print("=" * 60)
print(f"VERDICT on the practice set: fine-tuned {results['finetuned'][0]} right, stock {results['fewshot'][0]} right "
      f"(scores {results['finetuned'][1]} vs {results['fewshot'][1]} of {len(demo)})")
print("=" * 60)
'''),
    ("code", r'''
# The same on M3Kang validation problems, which were never trained on (about 10 minutes).
for name, path, style in (("fine-tuned GGUF", FT, "finetuned"), ("stock GGUF, three examples in the prompt", REF, "fewshot")):
    out = evaluate(path, val, style)
    with open(f"{WORK}/eval_val_{style}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    summary(f"M3KANG VAL - {name}", val, out)
'''),
    ("code", r'''
# Publish the GGUF. Runs only if HF_REPO is set in the first cell. The repository is public: that is what
# config.env needs, so the board can download the file without a token.
if not HF_REPO:
    print("HF_REPO is empty: nothing published. The GGUF is at", FT)
else:
    from kaggle_secrets import UserSecretsClient
    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
    api.create_repo(HF_REPO, exist_ok=True)
    card = f"""---
base_model: {BASE_REPO}
license: apache-2.0
language: [en, ca]
tags: [gguf, math, program-of-thought]
---
# Qwen3.5 0.8B, fine-tuned to answer Kangaroo-style math problems with a short Python program

Built for the Datathon FME 2026 (Qualcomm challenge, Arduino UNO Q). Given a problem statement in English or
Catalan, the model writes a short Python program that prints the answer; the copilot runs it and compares the
result with the options.

- Base model: `{BASE_REPO}` at revision `{base_sha}`.
- Training: LoRA (r={RANK}, alpha={ALPHA}, {EPOCHS} epochs, lr {LR}), loss on the program only, on {len(pairs)}
  (statement, program) pairs from {len({r['id'] for r in pairs})} problems of the
  [M3Kang](https://huggingface.co/datasets/qualcomm/M3Kang) training split. Every program was run and kept only
  if its result is the labelled answer. M3Kang is licensed for research use only: use this model for research.
- Conversion: adapter merged, then `convert_hf_to_gguf.py --outtype q8_0` from llama.cpp at commit `{LLAMA_REF[:7]}`.
- System prompt: `{pot.SYSTEM_FINETUNED}` The user message is the problem statement alone.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=HF_REPO)
    api.upload_file(path_or_fileobj=FT, path_in_repo=FT_FILE, repo_id=HF_REPO)
    revision = api.model_info(HF_REPO).sha
    print("\nLines for config.env:\n")
    print(f"LLM_REPO={HF_REPO}\nLLM_REVISION={revision}\nLLM_FILE={FT_FILE}\nLLM_SHA256={FT_SHA}\nPOT_STYLE=finetuned")
'''),
]


def train_cell(marker):
    """The source of the code cell of TRAIN that holds this text."""
    found = [source for kind, source in TRAIN if kind == "code" and marker in source]
    assert len(found) == 1, marker
    return found[0]


def swapped(source, old, new):
    assert old in source, old
    return source.replace(old, new)


# FT4: one run of three epochs, a GGUF after each epoch, and the epoch is chosen by programs that run to the
# right answer on M3Kang val. Notebook 01 chose by token loss on 34 problems, which did not track that.
TRAIN4 = [
    ("markdown", r'''
# FT4: three epochs, one GGUF per epoch, chosen by executed answers on M3Kang val

Same LoRA as `01_train_pot_lora.ipynb` (r=16, alpha=32, the same linear layers), with two changes:

1. **Training recipe:** 3 epochs at lr 2e-4, no dropout (the first fine-tune's recipe), batch 2 with 8 accumulation
   steps. The adapter is saved at the end of every epoch, to memory and to `lora_epoch<N>.pt`.
2. **Choosing the epoch:** every epoch is merged, converted to Q8_0 GGUF and measured with the board's server
   flags on the M3Kang `val` rows it would be asked (163 rows that no model was trained on). The epoch with the
   best expected score wins: a right answer is worth +0.8 over the fallback and a wrong match costs 0.2, so the
   ranking is `right - wrong / 4`. The practice set is measured too, but only shown: it was already used to tune
   the sandbox, so it must not choose the model again.

**Kaggle settings:** Accelerator **GPU T4**, Internet **on**, a **fresh session** (Run > Restart & clear), and the
same private dataset as notebook 01 (`pot_pairs.jsonl`, `demo.jsonl`, `val.jsonl`, `pot.py`). Run all: about one
hour. The table to copy back is printed by the cell that starts with `# Choose the epoch`.
'''),
    ("code", COMMON + r'''
import glob, random, shutil
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before torch touches the GPU
RUN_TAG = "v4"   # "v5" when the dataset also holds the synthetic low-level problems (ids that start with syn-)
FT_NAME = "Qwen3.5-0.8B-M3Kang-PoT-" + RUN_TAG + "-e{epoch}-Q8_0.gguf"
MERGED = f"{WORK}/merged"
EPOCHS, LR, RANK, ALPHA, BATCH, ACCUM, DROPOUT, WEIGHT_DECAY = 3, 2e-4, 16, 32, 2, 8, 0.0, 0.01
HF_REPO = ""   # to publish the chosen epoch: "your-user/Qwen3.5-0.8B-M3Kang-PoT-GGUF", and add the Kaggle secret HF_TOKEN

hits = glob.glob("/kaggle/input/**/pot_pairs.jsonl", recursive=True)
assert hits, "Add the dataset with pot_pairs.jsonl as an input of this notebook (Add Input, top right)."
DATA = os.path.dirname(hits[0])
sys.path.insert(0, DATA)
import pot   # copilot/pot.py: the prompts and the sandbox the board uses


def load(name):
    with open(f"{DATA}/{name}", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


pairs, demo, val = load("pot_pairs.jsonl"), load("demo.jsonl"), load("val.jsonl")
print(f"pairs {len(pairs)} ({len({r['id'] for r in pairs})} problems), practice set {len(demo)}, val {len(val)}")
print("problems by level:", dict(sorted(Counter(r["level"] for r in pairs if r["language"] == "eng").items())),
      "| synthetic:", len({r["id"] for r in pairs if r["id"].startswith("syn-")}))
lines = [len(r["code"].strip().splitlines()) for r in pairs]
print(f"programs of one line: {sum(n == 1 for n in lines)} of {len(lines)}; lines per program: mean {sum(lines) / len(lines):.1f}")
'''),
    ("code", GET_LLAMA),
    ("code", train_cell("pip install -q -U 'transformers>=5' safetensors")),
    ("code", train_cell("base_sha = HfApi().model_info(BASE_REPO).sha")),
    ("code", swapped(train_cell("class LoRALinear(nn.Module):"),
                     "LoRALinear(getattr(parent, attr), RANK, ALPHA)",
                     "LoRALinear(getattr(parent, attr), RANK, ALPHA, DROPOUT)")),
    ("code", WAIT_BUILD),
    ("code", train_cell("def hf_prompt(row):")),
    ("code", r'''
pad = tok.pad_token_id if tok.pad_token_id is not None else 0


def batch(items):
    n = max(len(x) for x, _ in items)
    ids_t = torch.full((len(items), n), pad, dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    for i, (x, y) in enumerate(items):
        ids_t[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids_t.cuda(), mask.cuda(), labels.cuda()


def loss_of(items):
    ids_t, mask, labels = batch(items)
    logits = model(input_ids=ids_t, attention_mask=mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    active = shift_labels != -100
    return torch.nn.functional.cross_entropy(shift_logits[active].float(), shift_labels[active])


torch.cuda.empty_cache()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
per_epoch = math.ceil(len(train) / (BATCH * ACCUM))
updates = EPOCHS * per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / updates))
print(f"{EPOCHS} epochs, {per_epoch} optimizer updates per epoch, {updates} in total", flush=True)
t0, step, skipped = time.time(), 0, 0
snapshots, dev_losses = {}, {}
for epoch in range(1, EPOCHS + 1):
    random.shuffle(train)
    model.train()
    running = []
    for i in range(0, len(train), BATCH):
        last = i + BATCH >= len(train)
        try:
            loss = loss_of(train[i:i + BATCH])
            (loss / ACCUM).backward()
            running.append(loss.item())
        except torch.cuda.OutOfMemoryError:
            skipped += 1                      # drop this batch and what it accumulated, and go on
            opt.zero_grad(set_to_none=True)
        loss = None
        step += 1
        if step % ACCUM == 0 or last:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if step % 100 == 0:
            print(f"ep {epoch} step {step}: train {sum(running[-70:]) / max(len(running[-70:]), 1):.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        dev_loss = [loss_of(dev[j:j + BATCH]).item() for j in range(0, len(dev), BATCH)]
    dev_losses[epoch] = sum(dev_loss) / max(len(dev_loss), 1)
    snapshots[epoch] = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(snapshots[epoch], f"{WORK}/lora_epoch{epoch}.pt")
    print(f"END OF EPOCH {epoch}: train {sum(running[-70:]) / max(len(running[-70:]), 1):.3f} | dev loss "
          f"{dev_losses[epoch]:.3f} (shown only, not used to choose) | batches skipped for memory {skipped} | "
          f"{time.time() - t0:.0f} s", flush=True)
assert len(snapshots[1]) == 2 * N_TARGETS
'''),
    ("code", r'''
# The model is no longer needed on the GPU: each epoch is merged on the CPU from the original checkpoint files.
del model, opt, params
torch.cuda.empty_cache()
from safetensors.torch import load_file, save_file

SCALE = ALPHA / RANK


def write_merged(snapshot):
    """The original checkpoint with base weight + LoRA delta in the trained layers: same files, same tensor names."""
    deltas = {name + ".weight": (snapshot[name + ".lora_B"] @ snapshot[name + ".lora_A"]) * SCALE for name in targets}
    used = Counter()
    shutil.rmtree(MERGED, ignore_errors=True)
    os.makedirs(MERGED)
    for name in sorted(os.listdir(f"{WORK}/base")):
        src = f"{WORK}/base/{name}"
        if os.path.isdir(src):
            continue
        if not name.endswith(".safetensors"):
            shutil.copy(src, f"{MERGED}/{name}")
            continue
        out = {}
        for key, old in load_file(src).items():
            found = [key] if key in deltas else [k for k in deltas if k.endswith("." + key) or key.endswith("." + k)]
            if len(found) == 1 and tuple(deltas[found[0]].shape) == tuple(old.shape):
                out[key] = (old.to(torch.float32) + deltas[found[0]]).contiguous()
                used[found[0]] += 1
            else:
                assert not found, f"{key} matches {found[:3]}: stop here and report this line"
                out[key] = old
        save_file(out, f"{MERGED}/{name}", metadata={"format": "pt"})
    assert len(used) == N_TARGETS and set(used.values()) == {1}, (
        f"merged {len(used)} of {N_TARGETS} layers, some more than once: {[k for k, v in used.items() if v > 1][:3]}")
    return len(used)

''' + train_cell("def evaluate(model_path, rows, style, max_tokens=140):").split("\n\nresults = {}")[0] + r'''


def counts(rows, out):
    """Right and wrong answers by row, problems with a right answer, and one-line programs."""
    matched = [o for o in out if len(o["matches"]) == 1]
    right = [o for o in matched if o["matches"][0] == o["label"]]
    return {"asked": len(out), "right": len(right), "wrong": len(matched) - len(right),
            "problems_right": len({o["id"] for o in right}),
            "one_liners": sum(len(o["code"].strip().splitlines()) == 1 for o in out),
            "net": len(right) - (len(matched) - len(right)) / 4}


table, files = {}, {}
for epoch in sorted(snapshots):
    t1 = time.time()
    path = f"{WORK}/{FT_NAME.format(epoch=epoch)}"
    n = write_merged(snapshots[epoch])
    sh(f"cd {WORK} && python {LLAMA}/convert_hf_to_gguf.py merged --outtype q8_0 --outfile {path} 2>&1 | tail -3")
    shutil.rmtree(MERGED, ignore_errors=True)
    files[epoch] = (path, sha256(path))
    print(f"epoch {epoch}: {n} layers merged, {os.path.getsize(path) // (1 << 20)} MB, sha256 {files[epoch][1]}", flush=True)
    table[epoch] = {}
    for set_name, rows in (("val", val), ("demo", demo)):
        out = evaluate(path, rows, "finetuned")
        with open(f"{WORK}/eval_{set_name}_epoch{epoch}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
        summary(f"EPOCH {epoch} - {'M3KANG VAL' if set_name == 'val' else 'PRACTICE SET'}", rows, out)
        table[epoch][set_name] = counts(rows, out)
    print(f"epoch {epoch} measured in {time.time() - t1:.0f} s", flush=True)
'''),
    ("code", r'''
# Choose the epoch: best expected score on M3Kang val (right - wrong / 4), then more right answers, then the
# earlier epoch. The practice set is printed next to it and takes no part in the choice.
BEST_EPOCH = max(table, key=lambda e: (table[e]["val"]["net"], table[e]["val"]["right"], -e))
print("=" * 78)
print("epoch | dev loss | VAL right/wrong (problems right, one-liners) net | PRACTICE right/wrong (problems right)")
for e in sorted(table):
    v, d = table[e]["val"], table[e]["demo"]
    print(f"  {e}   |  {dev_losses[e]:.3f}   | {v['right']:3d} / {v['wrong']:3d} of {v['asked']} ({v['problems_right']}, "
          f"{v['one_liners']})  {v['net']:+.2f} | {d['right']:3d} / {d['wrong']:3d} of {d['asked']} ({d['problems_right']})"
          f"{'   <- chosen' if e == BEST_EPOCH else ''}")
print("to compare, measured on a laptop with another llama.cpp commit, same sandbox, one program at temperature 0:")
print("  FT1 (first fine-tune, 3 epochs)  val 21 / 16 (+17.00)   practice 15 /  8")
print("  FT2 (-v2-, 1 epoch)              val  8 / 20  (+3.00)   practice 21 / 12")
print("  FT3 (-v3-, 1 epoch)              val 17 / 27 (+10.25)   practice 14 / 10")
FT, FT_SHA = files[BEST_EPOCH]
FT_FILE = os.path.basename(FT)
print(f"CHOSEN: epoch {BEST_EPOCH} -> {FT_FILE}\n  sha256 {FT_SHA}")
print("Keep FT1 instead if the chosen epoch is not above +17 on val." if table[BEST_EPOCH]["val"]["net"] <= 17
      else "Above FT1 on val: download this GGUF and check it on the board before changing config.env.")
print("=" * 78)
json.dump({"table": table, "dev_loss": dev_losses, "chosen": BEST_EPOCH,
           "sha256": {e: s for e, (_, s) in files.items()}}, open(f"{WORK}/ft4_report.json", "w"), indent=1)
'''),
    ("code", swapped(train_cell("# Publish the GGUF."), "{EPOCHS} epochs, lr {LR}",
                     "epoch {BEST_EPOCH} of {EPOCHS}, lr {LR}")),
]


if __name__ == "__main__":
    write("00_gate_conversion.ipynb", GATE0)
    write("01_train_pot_lora.ipynb", TRAIN)
    write("02_train_ft4_epochs.ipynb", TRAIN4)
