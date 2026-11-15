"""Pack what finetune/01_train_pot_lora.ipynb needs into one zip, to upload as a PRIVATE Kaggle dataset.

Usage, on the laptop, from the repository folder:  python finetune/make_kaggle_dataset.py
The zip holds M3Kang problems: keep the dataset private, and don't commit the zip (data/ is ignored by Git).
"""
import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
FILES = {
    "pot_pairs.jsonl": REPO / "data" / "pot_pairs.jsonl",   # finetune/build_pairs.py
    "demo.jsonl": REPO / "data" / "demo" / "train.jsonl",   # the practice set, never trained on
    "val.jsonl": REPO / "data" / "val.jsonl",               # M3Kang validation, never trained on
    "pot.py": REPO / "copilot" / "pot.py",                  # the prompts and the sandbox of the copilot
}

out = REPO / "data" / "kaggle_pot_dataset.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, path in FILES.items():
        z.write(path, name)
        print(f"  {name}: {sum(1 for _ in open(path, encoding='utf-8'))} lines")
print("wrote", out, f"({out.stat().st_size // 1024} KB)")
