# Executar una avaluació

Aquesta guia explica com executar `eval.py` i on trobar els resultats. Les ordres s'han d'executar des de l'arrel del projecte, al dispositiu on s'executa el model (normalment l'Arduino UNO Q).

## Abans de començar

- Comprova que el servidor del model està en marxa a `127.0.0.1:8081`. El port es configura amb `LLM_PORT` a `config.env`.
- Comprova que el fitxer JSONL de validació és dins de `data/`. Per defecte, `eval.py` llegeix `data/val.jsonl`.
- Cada línia del JSONL ha de contenir com a mínim `id`, `text` i `label`; també pot tenir `language`, `level` i `options`.

El model rep el camp `text` i les opcions disponibles al camp `options`. El text de la pregunta prové del conjunt de dades M3Kang; l'OCR s'utilitza abans de l'avaluació per llegir les opcions de la imatge. Durant `eval.py` no es torna a executar OCR ni s'envia la imatge al model. La resposta correcta (`label`) es reserva per comparar-la amb la resposta del model.

## Avaluació ràpida

Des de l'arrel del projecte:

```bash
python eval.py --limit 50 --name baseline
```

Això avalua 50 preguntes del fitxer per defecte. La selecció es barreja amb una llavor fixa, així que les execucions comparables fan servir la mateixa mostra si el fitxer no canvia.

Per fer servir un fitxer concret i totes les preguntes que conté:

```bash
python eval.py --input data/val_all_reviewed.jsonl --limit 0 --name val-all
```

`--limit 0` vol dir que no es limita el nombre de preguntes. Pots restringir la llengua amb `--languages eng` o `--languages cat`.

Per centrar-se només en preguntes marcades com a text (`text_only: true` al JSONL), afegeix `--text-only`:

```bash
python eval.py --input data/val_all_reviewed.jsonl --text-only --limit 0 --name text-only
```

El filtre s'aplica abans de `--limit`, de manera que `--limit 50 --text-only` avalua fins a 50 preguntes de text, si n'hi ha prou al fitxer.

## RAG opcional

RAG afegeix a cada prompt exemples semblants del conjunt d'entrenament. Si encara no tens l'índex, crea'l una vegada:

```bash
python build_index.py --input data/train_all_reviewed.jsonl --db data/index.db --only-with-options
```

Després executa l'avaluació amb un exemple recuperat per pregunta:

```bash
python eval.py --input data/val_all_reviewed.jsonl --limit 0 --rag 1 --index data/index.db --name rag1
```

Si ja tens un índex vàlid, pots saltar-te la primera ordre. `--rag 2` n'afegeix dos exemples. L'índex s'ha de construir amb dades d'entrenament, no amb les de validació.

## Fitxers de resultat

Cada execució crea fitxers nous dins de `results/`; no reutilitza ni sobreescriu el CSV d'una altra execució:

- `eval-<data-hora>-<nom>.csv`: una fila per problema, identificat per `problem_id` i `language`. Inclou resposta triada i correcta, encert, la sortida íntegra del model (`model_output`, amb el raonament i la resposta final), tokens d'entrada i sortida, temps real, estat i identificació dels exemples RAG.
- `eval-<data-hora>-<nom>.json`: manifest de l'execució, amb paràmetres, hora d'inici i final, progrés i estat (`running`, `completed` o `interrupted`).
- `ablations.csv`: resum acumulat de les execucions, amb el commit, la precisió i la latència. Les execucions interrompudes queden marcades al nom.

El CSV també inclou l'estimació de temps a l'Arduino. Es calcula com:

```text
tokens_entrada / ARDUINO_PROMPT_TOKENS_PER_SECOND
+ tokens_sortida / ARDUINO_OUTPUT_TOKENS_PER_SECOND
```

Els valors per defecte són 11 tokens/s de lectura i 6 tokens/s de sortida; es poden canviar a `config.env`. Si el servidor no informa d'algun recompte de tokens, el recompte corresponent i l'estimació queden buits.

`eval.py` desa cada fila immediatament. Si prems Ctrl+C mentre resol una pregunta, conserva els resultats anteriors i registra la pregunta en curs amb l'estat `interrupted`; el manifest indica que l'execució no s'ha completat.

## Opcions més habituals

| Opció | Funció |
| --- | --- |
| `--input` | JSONL d'entrada; per defecte `data/val.jsonl`. |
| `--limit` | Nombre màxim de preguntes; `0` vol dir totes. Per defecte `50`. |
| `--languages` | Filtra per llengua, per exemple `eng` o `eng,cat`. |
| `--text-only` | Avalua només les preguntes amb `text_only: true`. Per defecte no filtra aquest camp. |
| `--name` | Nom descriptiu que apareix als fitxers i al resum. |
| `--rag K` | Afegeix els `K` exemples més semblants de l'índex. Per defecte `0`. |
| `--index` | Ruta de l'índex RAG; per defecte `data/index.db`. |

Per exemple, per avaluar totes les preguntes catalanes del fitxer revisat:

```bash
python eval.py --input data/val_all_reviewed.jsonl --languages cat --limit 0 --name val-cat
```
