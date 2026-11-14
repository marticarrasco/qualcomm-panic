# Revisió visual de l’OCR

Aquest lot conté 100 problemes en anglès amb `text_only = true`. Els resultats OCR són a `data/train.jsonl` i `data/val.jsonl`; les imatges originals del lot són a `data/textonly_100.parquet`.

## Obrir el visor

Des de la carpeta del projecte:

```powershell
.\.venv\Scripts\activate
pip install -r laptop\requirements.txt
python laptop\ocr_review.py --parquet data\textonly_100.parquet
```

Obre `http://127.0.0.1:8765` al navegador. Compara cada imatge amb les opcions A–E detectades, corregeix-les si cal i prem **Desa revisió**. El progrés es desa a `data/ocr_review.json`. En prémer **Exporta JSONL**, el visor crea `train_reviewed.jsonl` i `val_reviewed.jsonl` sense sobreescriure els originals.

Per aturar el visor, torna a la terminal i prem `Ctrl+C`.
