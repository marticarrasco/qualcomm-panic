# Dades i OCR

Els fitxers de M3Kang tenen llicència d’ús per a recerca. No publiques els JSONL, les imatges, la memòria cau OCR ni l’índex de preguntes. Consulta la [fitxa del conjunt de dades](https://huggingface.co/datasets/qualcomm/M3Kang).

## Estructura

```text
data/
├── README.md
├── sample_problems.jsonl       # cinc preguntes inventades per a proves ràpides
├── train.jsonl                 # conjunt inicial: 87 preguntes
├── val.jsonl                   # conjunt inicial: 13 preguntes
├── ocr_review.json             # estat del visor OCR; queda ací mentre el visor està obert
├── ocr/
│   ├── .ocr_cache.jsonl        # memòria cau OCR
│   └── source/
│       └── textonly_100.parquet # imatges originals del conjunt inicial
├── ocr_batches/eng/            # lots OCR originals, separats per partició
├── reviewed/
│   ├── initial/                # exportació revisada del conjunt inicial
│   ├── batches/eng/             # exportacions revisades, amb l’estructura dels lots
│   └── all/                     # agregació per entrenament i validació
└── diagnostics/lang_check/     # scripts i resultats de comprovacions exploratòries
```

Els fitxers `train.jsonl` i `val.jsonl` a l’arrel continuen sent les entrades per defecte del pipeline. Els fitxers originals de cada lot es queden a `ocr_batches/`; les seues còpies exportades van a `reviewed/batches/`.

## Fitxers agregats actuals

`reviewed/all/train.jsonl` i `reviewed/all/val.jsonl` ajunten el conjunt inicial i els lots anglesos disponibles. Es regeneren a partir dels originals i no tenen IDs repetits ni solapament entre particions. Recompte actual: **1.184 preguntes d’entrenament i 227 de validació**.

Per avaluar totes les preguntes angleses de validació agregades, des de la carpeta arrel del projecte executa:

```bash
python eval.py --input data/reviewed/all/val.jsonl --languages eng --limit 0 --name current-eng-all
```

## Què vol dir «revisat»

Els registres poden tindre l’estat `reviewed` a `ocr_review.json` i aparéixer a les exportacions. **Aquesta marca no certifica que una persona haja comparat el text amb la imatge.** Les lectures OCR es conserven tal com s’han extret, llevat de correccions fetes al visor. `options` pot ser `{}` si l’OCR no ha detectat opcions textuals.

Per revisar visualment el lot inicial, segueix [OCR_REVIEW.md](../laptop/OCR_REVIEW.md). El visor usa `train.jsonl` i `val.jsonl` per defecte i exporta a `reviewed/initial/`. Els lots addicionals requereixen passar també les seues imatges i JSONL al visor.

## Proves exploratòries

`diagnostics/lang_check/` conté materials auxiliars per a comprovacions entre idiomes; no forma part del pipeline d’entrenament ni d’avaluació.
