# Comprovació: mateix problema, mateixes opcions i mateix `label` entre idiomes

**Veredicte: l'assumpció del pipeline (OCR només de l'anglès; opcions i ordre A–E compartits
per tots els idiomes del mateix `id`) es compleix. I les respostes són neutres lingüísticament:
cap opció és una paraula en un idioma concret, així que es poden creuar entre idiomes.**

## 1. Documentació (dataset card + paper arXiv:2601.16218)

El pipeline de Qualcomm construeix un dataset base en *català* a partir dels PDFs de la SCM,
el tradueix a anglès **re-envolcallant només el text de l'enunciat dins la mateixa bounding box
de la imatge**, i després fa el mateix per a cada idioma.

- Les opcions A–E i les figures **no es toquen**: queden a la mateixa posició per a tots els idiomes.
- Al crear la base es van descartar els problemes amb "Catalan-specific terms in images or
  answers" → les opcions queden neutral (números, símbols, notació matemàtica, figures).
- El `label` ve de la clau de respostes del PDF original → el mateix per a tot l'`id`.
- La QA de traduccions (back-translation + LLM-as-judge) descarta traduccions dolentes:
  per això fra/esp/deu tenen menys de 1.747 files.

## 2. Comprovació global de `label` (shard 0 de `high_quality`, el del projecte)

- 1.747 `id` únics × fins a 14 idiomes (10.155 files): **0 conflictes**.

## 3. Comprovació visual (10 problemes, nivells 0–7, anys 2007–2016)

10 problemes × 5 idiomes (eng, cat, fra, esp, deu), un full de contacte per problema:
mateixa pregunta traduïda, mateixes opcions en el mateix ordre, mateix `label` (i el valor
de l'opció marcada és correcte, p. ex. lvl-1_2015_1 → 12, lvl-5_2007_15 → 1%).
Fulls: `lvl-*_*.png` en aquesta carpeta. Script: `dump_grid.py`.

## 4. Verificació mecànica de que les opcions són iguals entre idiomes

- **Franja d'opcions idèntica píxel a píxel** en problemes amb figura: 12 `id` × 4 idiomes,
  dif. mitjana de píxels (0–255): **0.00** a la franja inferior (l'enunciat, superior, sí
  canvia: 29.2). Script: `strip_diff.py`.
- **OCR creuat** (el mateix `parse_options` del projecte) sobre mostres aleatòries de
  `text_only` amb els 5 idiomes (225 ids en cau; 188 amb 5 idiomes; **125 completament
  parsables**):
  - opcions normalitzades idèntiques als 5 idiomes: **108/125 (86%)**;
  - l'opció marcada amb `label` té el mateix valor als 5 idiomes: **122/125 (97.6%)**;
  - els 17 "diff" i els 3 "label-diff" revisats a l'ull: **tots són soroll d'OCR o fórmules
    llegides malament, mai diferències del dataset** (les imatges són idèntiques):

    | Tipus | Exemples |
    |---|---|
    | 0 vs lletra O | lvl-4_2013_30 (E), lvl-5_2009_2 (A) |
    | símbol − U+2212 | lvl-5_2020_5 ("−4" vs "-4") |
    | separador de milers | lvl-4_2022_21 ("3 033"), lvl-5_2019_14 ("4 400") |
    | fórmules en cursiva amb subíndex | lvl-7_2021_9 ("2pq"→"2pg"), lvl-7_2017_8 ("g2(x)=x³") |
  - Nota: les opcions sovint són **notació matemàticaamb lletres com a símbols**:
    permutacions (V1,V2,V3), desigualtats (0<k<1), fórmules (g2(x)=x³, a⁵+1=b, 24∈, №12,
    radicals √2) — **idèntiques a tots els idiomes**. Cap paraula vertadera ni vocabulari
    vinculat a un idioma a cap opció de la mostra (~225 ids × 5 idiomes).

## Troballes de QA sobre el vostre pipeline OCR (l'anglès)

- `lvl-0_2020_13`: `ocr_review.json` ha guardat `"E": "Q"` **marcat com revisat** — la
  imatge mostra un «8» clar.
- `lvl-1_2016_17` (i possiblement més dels 21 "sense opcions"): opcions legibles i
  numèriques (110–150); el parser ha fallat la lectura (no són imatges).
- Si OCR-ejeu no-English: normalitzeu coma decimal («0,5%») i el separador de milers («4 400»).

## Annexes i scripts

- Fulls de contacte: `lvl-*.png` (mostra de 10) i `chk_*.png` (casos suspects revisats), noms
  `lvl-N_YYYY_P.png`, etiqueta de cada idioma al marc negre.
- Scripts: `check_labels.py` (labels globals), `dump_grid.py` (fulls), `strip_diff.py`
  (franjes de píxels), `ocr_crosslang.py` (OCR creuat; escriu `ocr_cache.jsonl` UTF-8),
  `classify_cached.py` (classificació textual del contingut).
