# Dades i OCR

Aquest directori conté les preguntes M3Kang preparades per al projecte, els resultats de l’OCR de les opcions i els fitxers de revisió. Les dades de M3Kang tenen llicència d’ús per a recerca: no publiques els JSONL, les imatges ni l’índex de preguntes. Consulta la [fitxa del conjunt de dades](https://huggingface.co/datasets/qualcomm/M3Kang).

## Estructura

| Fitxer o carpeta | Contingut |
| --- | --- |
| `train.jsonl`, `val.jsonl` | Conjunt inicial de 100 preguntes (87 d’entrenament i 13 de validació). |
| `ocr_batches/eng/batch_0001/` … `batch_0003/` | Lots addicionals en anglès, cadascun amb `train.jsonl` i `val.jsonl`. |
| `train_all_reviewed.jsonl`, `val_all_reviewed.jsonl` | Agregació actual del conjunt inicial i dels lots anglesos: 1.005 preguntes d’entrenament i 190 de validació. No hi ha IDs repetits ni solapament entre les dues particions. |
| `*_reviewed.jsonl` | Còpies dels JSONL d’origen amb les opcions guardades a l’estat de revisió. Els fitxers dels lots tenen la seua còpia al mateix subdirectori. |
| `ocr_review.json` | Estat i opcions associades a cada pregunta revisada pel procés del projecte. |
| `.ocr_cache.jsonl` | Memòria cau de les opcions extretes per OCR; pot incloure IDs que encara no formen part dels JSONL d’entrenament o validació. |
| `textonly_100.parquet` | Imatges originals del lot inicial de 100 preguntes, per comparar-les amb l’OCR al visor. |
| `sample_problems.jsonl` | Cinc preguntes inventades per a proves ràpides; no són dades reals de M3Kang. |

Cada línia dels JSONL és una pregunta amb camps com `id`, `language`, `text`, `options` i `label`. `options` pot ser `{}` quan l’OCR no ha detectat opcions textuals (per exemple, si les opcions són imatges).

## Què vol dir «revisat» ací

Els registres actuals estan marcats com a `reviewed` a `ocr_review.json` per poder generar les còpies `_reviewed.jsonl`. **Aquesta marca no certifica que una persona haja comparat el text amb la imatge.** Els textos OCR s’han conservat; per tant, les lectures encara poden contindre errors, i les opcions buides continuen buides. Per considerar un OCR verificat, cal obrir la pregunta al visor i comprovar visualment les opcions A–E.

Per revisar les imatges del lot inicial, segueix [OCR_REVIEW.md](../laptop/OCR_REVIEW.md). El visor treballa amb `train.jsonl` i `val.jsonl` per defecte; els lots addicionals necessiten que se li passen els JSONL i les imatges corresponents.

## Avaluació

Per avaluar el model amb totes les preguntes angleses de validació agregades, des de la carpeta arrel del projecte executa:

```bash
python3 eval.py --input data/val_all_reviewed.jsonl --languages eng --limit 0 --name current-eng-all
```

`--limit 0` vol dir que s’usen totes les preguntes del fitxer. L’avaluació només és tan fiable com les dades OCR que rep; una marca `reviewed` no elimina possibles errors de lectura.
