# Avaluació només del model (anglès, text only)

`eval_model_only.py` mesura únicament el model. No fa servir regles, calculadora,
ranker d'opcions, mitjanes, mediana, prior ni recuperació d'exemples. Cada pregunta
es resol amb una crida al servidor de `llama.cpp` i la resposta es llegeix del text
que genera el model.

## Conjunt avaluat

- `data/eval_sets/val_text_only_all.jsonl`, filtrat a `language = eng` i
  `text_only = true`.
- 121 files, 121 IDs distints (cada problema anglès apareix una sola vegada).
- Totes tenen les cinc opcions A–E completes.

## Prompt

Es va provar primer el prompt de dues frases. Amb 128 tokens de sortida el model
s'allargava quatre o cinc frases i sovint tallava abans d'arribar a `Answer: X`
(17/121 encerts, 64 respostes sense lletra). Per això la execució final fa servir
el prompt `strict`, d'una frase:

```text
You answer multiple-choice math competition questions. For each question: write ONE
short sentence of reasoning (maximum 25 words), then a new line exactly 'Answer: X'
where X is the correct option letter A, B, C, D or E. Never write more than one
sentence. Never repeat the question. No other text.
```

El prompt inclou dos exemples curts i es repeteix la instrucció després de
l'enunciat. `max_tokens = 128`, `temperature = 0`, `seed = 42`. No hi ha gramàtica
forçada: es va provar i empitjorava el raonament perquè tallava la línia a 240
caràcters.

## Execució

Des de l'arrel del projecte, amb el servidor del model a `127.0.0.1:8081`:

```bash
python eval_model_only.py --prompt strict --limit 0 --name eng-text-only-strict
```

O des de l'arrel del repositori, amb el llançador de Windows:

```bat
run-model-eval.cmd
```

Cada fila del CSV es desa immediatament; el JSON adjunt és el manifest de
l'execució (prompt exacte, model servit, progrés i resum).

## Resultat (15 Nov 2026, Qwen3.5 0.8B Q8_0, portàtil)

| Mètrica | Valor |
|---|---:|
| Encerts | **37/121 = 30.6%** |
| Interval de confiança del 95% (bootstrap) | 22.3% – 38.8% |
| Entre les 114 respostes parsejables | 37/114 = 32.5% |
| Sempre D (referència) | 24/121 = 19.8% |
| Atzar | 20.0% |
| Respostes sense lletra (tallades a 128 tokens) | 7/121 = 5.8% |

Per nivell:

| Nivell | Encerts |
|---:|---:|
| 0 | 2/4 (50.0%) |
| 1 | 2/4 (50.0%) |
| 2 | 0/6 (0.0%) |
| 3 | 2/6 (33.3%) |
| 4 | 8/28 (28.6%) |
| 5 | 10/30 (33.3%) |
| 6 | 5/26 (19.2%) |
| 7 | 8/17 (47.1%) |

Lletres predites / reals: A 42/28, B 13/27, C 29/18, D 23/24, E 7/24. El model
té un biaix clar cap a A i C i evita E.

Cost: 57.4 tokens de sortida de mitjana (mediana 48, màxim 128), 4.2 s per
pregunta de mitjana (mediana 3.7 s, màxim 8.8 s) en aquest portàtil. No és temps
de la placa.

## Compliment del format

El model va escriure exactament una frase de raonament en 56 de 121 casos; en 62
en va escriure dues o més i en 3 va respondre només la lletra. La instrucció de
dues frases no és viable amb 128 tokens en aquest model.

## Comparació amb el programa híbrid

El mateix conjunt de 121 problemes anglesos s'ha passat també pel programa que ja
hi havia, amb la configuració per defecte de `config.env` (`ANSWER_MODE=text`,
`TEXT_ROUTE=uncertain`, `TEXT_PROFILE=shots`, `TEXT_MAX_TOKENS=192`). Aquest
programa fa servir el ranker d'opcions i només crida el model quan el ranker no
arriba a 0,29 de confiança.

| Política | Encerts | % | Model |
|---|---:|---:|---|
| Model sol, prompt estricte, 128 tokens | 37/121 | 30,6% | 121 crides |
| **Híbrid per defecte** (ranker + model si incert) | **44/121** | **36,4%** | 41 crides |
| Només ranker, sense model | 40/121 | 33,1% | 0 |
| Prior de mediana / D (`priors.guess`) | 34/121 | 28,1% | 0 |
| Sempre D | 24/121 | 19,8% | 0 |

Desglossament del híbrid per estat: ranker 27/75 (36,0%), model respost 9/26
(34,6%), fallback a ranker 3/15 (20,0%) i coincidència exacta amb train 5/5
(100%; són problemes gairebé duplicats del train, no independents).

De les 41 files on el híbrid va cridar el model, el model sol també en va encertar
12 i el ranker 8. El guany del híbrid ve sobretot de les 80 files que resol sense
model: 32/80 (40,0%) contra 25/80 (31,3%) del model sol.

Temps del híbrid en aquest portàtil: 3,2 s de mitjana (mediana 0 s perquè 80 files
no fan cap crida), màxim 15,3 s. Estimació per a la UNO Q: 11,2 s de mitjana,
p95 45,2 s i màxim 48,9 s. El model dels 41 casos incerts fa servir el prompt
`shots` de 192 tokens, no el prompt estricte de 128, així que no és una comparació
controlada del component de model.

Ordre:

```bash
python eval.py --input data/eval_sets/val_text_only_all.jsonl \
  --languages eng --text-only --limit 0 --name eng-text-only-hybrid-default

# Només ranker, sense cap crida al model
TEXT_ROUTE=ranker python eval.py --input data/eval_sets/val_text_only_all.jsonl \
  --languages eng --text-only --limit 0 --name eng-text-only-ranker-only
```

## Fitxers

- `eval_model_only.py`: l'avaluador de model sol.
- `run-model-eval.cmd`: llançador amb els paràmetres finals.
- `results/model-only-20261115-002534-791993-eng-text-only-strict.csv`: una fila
  per pregunta, amb sortida completa, tokens i estat.
- `results/model-only-20261115-002534-791993-eng-text-only-strict.json`: manifest.
- `results/eval-20261115-003923-743355-eng-text-only-hybrid-default.csv`: híbrid.
- `results/eval-20261115-004604-263195-eng-text-only-ranker-only.csv`: ranker sol.

Les execucions de prova (`smoke-*`) i la primera passada amb el prompt d'una
frase sense exemples (`eng-text-only-one-sentence`) es conserven a `results/`
per comparar.
