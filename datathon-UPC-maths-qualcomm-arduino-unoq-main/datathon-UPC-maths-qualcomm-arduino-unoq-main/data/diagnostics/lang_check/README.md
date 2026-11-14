# Comprovació: mateix problema, mateixes opcions i mateix `label` entre idiomes

**Veredicte: l'assumpció del pipeline (OCR només de l'anglès; opcions i ordre A–E compartits
per tots els idiomes del mateix `id`) es compleix.** Evidència, en tres passes:

## 1. Documentació (dataset card + paper arXiv:2601.16218)

El pipeline de Qualcomm construeix un dataset base en *català* a partir dels PDFs de la SCM
_respecting bounding boxes_, el tradueix a anglès **re-envolcallant només el text de l'enunciat
dins la mateixa bounding box de la imatge**, i després fa el mateix per a cada idioma.
- Les opcions A–E i les figures **no es toquen**: queden a la mateixa posició per a tots els idiomes.
- A més, al crear el dataset base es van **descartar** els problemes amb "Catalan-specific terms
  in images or answers", perquè les opcions hagin quedat neutres (números, símbols, figures).
- El `label` ve de la clau de respostes del PDF original de cada test → el mateix per a tot l'`id`.
- La QA de traduccions (back-translation + LLM-as-judge) descarta traduccions dolentes: per això
  cat/fra/esp/deu tenen menys de 1.747 files (= el que reporta el README del projecte).

## 2. Comprovació global de `label` (shard 0 de `high_quality`, el del projecte)

- 1.747 `id` únics × fins a 14 idiomes (10155 files).
- **0 conflictes**: cap `id` té el mateix `label` diferent entre dos idiomes.

## 3. Comprovació visual + OCR creuat

- 10 problemes (nivells 0–7, amb i sense figura, anys 2007–2016) × 5 idiomes
  (eng, cat, fra, esp, deu), un full de contacte per problema en aquesta carpeta:
  mateix problema traduit, mateixes opcions en el mateix ordre, mateix `label`, i el valor
  de l'opció marcada és correcte (comprovat matemàticament en els casos textuals).
- OCR creuat amb el mateix `parse_options` de `laptop/prepare_m3kang.py`: 60 `id` `text_only`
  (d'un total de 340 amb els 5 idiomes): en els 38 OCR-parsables als 5 idiomes,
  **38/38 el valor de l'opció a la lletra `label` és idèntic en tots els idiomes**
  (33/38 amb les 5 opcions literalment idèntiques). Els 9 «desajustos» restants són soroll d'OCR
  («O» vs «0», signe «−» Unicode, un dígit perdut) o format local — verificat a l'ull que les
  imatges són idèntiques.

## Annexes i scripts

- Fulls de contacte: `data/diagnostics/lang_check/*.png` (nom `lvl-N_YYYY_P.png`).
- Scripts de la comprovació: `check_labels.py` (labels globals), `dump_grid.py` (fulls de
  contacte), `ocr_crosslang.py` (OCR creuat, usa i reomple `ocr_cache.jsonl`).

## Sobre el contingut dels idiomes a les respostes (2a pregunta, ampliació)

**Conclusió: les opcions són neutres lingüísticament — no hi ha respostes "en un idioma",
i per tant es poden creuar entre idiomes sense retocar res.**

Evidència addicional:

1. **Text_only amb opcions llegibles (cau OCR creuat, ~75 `id` × 5 idiomes):** ~79% de
   jocs d'opcions purament numèrics; els pocs casos amb lletres són símbols no traduits:
   - `lvl-2_2018_23`: patrons de lletres (`AAACBB`) idèntics als 5 idiomes.
   - Simbol matemàticpresent en una opció OCR'd: `∈` (U+2208).
   Cap paraula tipus *Yes/No/Sí/Oui* o vocabulari vinculat a un idioma.

2. **Problemes amb figura: la franja inferior (opcions) és idèntica píxel a píxel**
   entre idiomes — 12 problemes amb figura × 4 idiomes vs marglès:
   dif. mitjana de píxels (escala 0-255) **0.00 a la franja d'opcions** ( superior,
   l'enunciat, canvia: mitjana 29.2, per la traducció). Script: `strip_diff.py`.

Això coincideix amb el disseny del dataset: al crear la base catalana es van guardar
els problemes amb termes Catalans específics a imatges/respostes, perquè les opcions
quedessin simbòliques/figures.

### Troballes de QA sobre el vostre pipeline OCR (l'anglès)

- `lvl-0_2020_13`: `ocr_review.json` ha guardat `"E": "Q"` **marcat com revisat** – la imatge
  mostra un «8» clar (lectioni errònia de l'OCR no corregida).
- `lvl-1_2016_17` (i possiblement més dels 21 "sense opcions"): les opcions són legibles
  i numèriques (110–150); el parser n'ha fallat la lectura (no són imatges).



1. La comprovació cobreix el **shard 0** de `high_quality`. Si afegiu idiomes/`--shards all`,
   torneu a llençar `check_labels.py` (triga ~1 min) — el patronat de `label` segurament és
   el mateix, però és barat de confirmar.
2. Si algun dia ajunteu opcions OCR de non-English images directament: normalitzeu la coma
   decimal («0,5%» vs «0.5%») i el separador de milers francès («4 400»).
3. Les traduccions amb text de l'enunciat tenen qualitat variable (ex. «la joie du chat» per
   al gat Joy) — irrellevant per a respostes, però compte si copiu textos en altres idiomes.
