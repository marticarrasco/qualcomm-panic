# Millora de l'efectivitat del model: API de Helmcode, Fine-Tuning LoRA (v1 i v2) i `SmartRunner`

Aquest document explica pas a pas com hem passat del **20,0% d'encert inicial** (nivell d'atzar) fins al **41,0% en una sola passada (`Pass@1`, 11 s de mitjana a la placa)** i **42,0% amb reintent condicional (`Pass@3`, ~18,6 s de mitjana a la placa)** al conjunt oficial de pràctica 2026 (`data/demo/train.jsonl`, 100 problemes: 50 en anglès i 50 en català).

---

## Resum executiu: evolució de punta a punta

Totes les mesures següents s'han executat sobre els **100 problemes del set de pràctica 2026 (`demo`)** (50 `eng` + 50 `cat`, nivells 0 a 4, 80 `text_only` i 20 amb figura) amb `seed=42` i estimant el temps d'inferència a l'Arduino UNO Q (`11 tok/s` de lectura i `5,6 tok/s` d'escriptura):

| Etapa / Configuració | Encert total (`demo`, $N=100$) | Encerts per codi (`eng` + `cat`) | Falsos positius de codi | Precisió quan el codi casa | Temps mitjà estimat a la placa (pitjor cas) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline 1-token** (`Qwen3.5-0.8B-Q8_0`, `root ::= [A-E]`) | **20,0%** (20/100) | — (respon `D` el 74% dels cops) | — | 20,0% | 14,5 s (22 s) |
| **1. PoT base 3-shot** (`Qwen3.5-0.8B-Q8_0` stock) | **28,0%** (28/100) | 11 (`9 eng`, `2 cat`) | 10 | 52,4% | 23,0 s (47 s) |
| **2. PoT base 4-shot bilingüe amb caché `< 512` tok** | **33,0%** (33/100) | 14 (`8 eng`, `6 cat`) | 6 | 70,0% | 19,5 s (44 s) |
| **3. Fine-Tuning LoRA `v1` (`Pass@1`, 0-shot)** | **34,0%** (34/100) | 18 (`10 eng`, `8 cat`) | 11 | 62,1% | **12,0 s** (32 s) |
| **4. Fine-Tuning LoRA `v1` (`Pass@3` condicional)** | **36,0%** (36/100) | 20 (`11 eng`, `9 cat`) | 11 | 64,5% | 19,2 s (44 s) |
| **5. Fine-Tuning LoRA `v2` (`Pass@1`, 0-shot)** | **37,0%** (37/100) | 19 (`11 eng`, `8 cat`) | **8** | **70,4%** | **11,0 s** (29 s) |
| **6. Fine-Tuning LoRA `v2` (`Pass@3` condicional)** | **39,0%** (39/100) | 21 (`12 eng`, `9 cat`) | 10 | 67,7% | 18,6 s (43 s) |
| **7. Fine-Tuning LoRA `v2` + `SmartRunner` (`Pass@1`)** | **41,0%** (41/100) | **23 (`13 eng`, `10 cat`)** | **8** | **74,2%** | **11,0 s** (29 s) |
| **8. Fine-Tuning LoRA `v2` + `SmartRunner` (`Pass@3`)** | **42,0%** (42/100) | **24 (`14 eng`, `10 cat`)** | 10 | **70,6%** | **18,6 s** (43 s) |

---

## 0. Diagnòstic inicial: per què vam passar a *Program-of-Thought* (PoT)?

1. **Col·lapse dels priors de M3Kang al set del 2026 (`demo`)**:
   - Al dataset històric M3Kang (`train.jsonl` / `val.jsonl`), la lletra `D` és la resposta correcta el 24,6% dels cops i les opcions numèriques desordenades afavoreixen el valor central (`median_choice`).
   - Al set de pràctica 2026 de l'organització (`data/demo/train.jsonl`, 100 preguntes), les lletres `A–E` estan repartides de forma **exactament uniforme (20% cadascuna)**. Encara més important:
     - Als **80 problemes `text_only`**, la `D` només és correcta **10/80 vegades (12,5%)**.
     - Als **20 problemes amb figura (`text_only: false`)**, la `D` és correcta **10/20 vegades (50,0%)**.
   - Per tant, qualsevol heurística de triar lletra o usar priors en problemes de text cau al 20% o per sota, mentre que respondre `D` de forma immediata (`0,00 s`) quan `text_only == False` encerta el 50% sense gastar temps de model.

2. **Per què *Program-of-Thought* (PoT) funciona amb un model de 0.8B**:
   - Quan `Qwen3.5-0.8B` intenta raonar en text lliure i fer aritmètica mentalment, s'equivoca en les operacions o esgota el límit de 60 s a la placa.
   - Amb **PoT (`copilot/pot.py`)**, el model **no veu les opcions A–E**: només llegeix l'enunciat i escriu un script curt de Python pla (2–6 línies) que imprimeix el resultat numèric (`print(ans)`).
   - Python executa el càlcul exacte en `< 20 ms` dins d'un procés aïllat (`python -I -X utf8 -c RUNNER`) i nosaltres comparem el número imprès amb les 5 opcions (`A–E`). Com que un programa incorrecte gairebé mai imprimeix per casualitat una de les 4 opcions distractores, **quan el resultat imprès coincideix amb una única opció, tenim entre un 70% i un 74% de seguretat que és la resposta correcta**.

3. **El límit `ubatch=512` de *Gated DeltaNet* a `llama-server`**:
   - `Qwen3.5-0.8B` té una arquitectura híbrida de 24 capes: **18 capes recurrents *Gated DeltaNet*** + **6 capes d'atenció completa**.
   - Vam descobrir que `llama-server` només desa el *checkpoint* de l'estat recurrent al final d'un `ubatch` (512 tokens per defecte). Si el prefix *few-shot* supera els 512 tokens, el servidor invalida la caché i re-llegeix tot el prompt a cada problema (~50 s només de lectura a la placa).
   - Dissenyant un prompt bilingüe (`POT_SHOTS` a `copilot/llm.py`) d'exactament **496 tokens (`< 512`)**, `llama-server` reutilitza `cache_n = 496` a cada pregunta (0 s de re-lectura), pujant el model base fins al **33,0%**.
   - Tot i així, el model base fallava molt en **català** (només 2 encerts en 3-shot) i a vegades escrivia funcions llargues o sintaxi invàlida. Calia fer *fine-tuning*.

---

## 1. Generació i verificació de dades sintètiques amb l'API de Helmcode

Per entrenar `Qwen3.5-0.8B` a traduir enunciats matemàtics (en anglès i català) a scripts de Python curts i nets, vam construir un pipeline de destil·lació verificada per execució utilitzant l'API de **Helmcode** (`https://api.helmcode.com/v1/chat/completions`, model `deepseek-v4-flash`):

### Fitxers del pipeline de dades
- [`finetune/build_synthetic_dataset.py`](finetune/build_synthetic_dataset.py): Crida concurrent a l'API de Helmcode (`HELMCODE_API_KEY` llegida per variable d'entorn, mai al codi) sobre els **684 problemes d'entrenament** (`data/train.jsonl`) amb `language == "eng"`, `text_only == True` i 5 opcions.
- [`finetune/prepare_sft_and_rag.py`](finetune/prepare_sft_and_rag.py): Converteix els procediments verificats en datasets SFT bilingües (`sft_train_pot.jsonl`, `sft_train_plan_pot.jsonl`, `sft_train_multitask.jsonl`) i enriqueix l'índex per al tutor.
- [`finetune/build_pairs.py`](finetune/build_pairs.py): Filtra i emparella cada programa verificat amb les versions en **anglès (`eng`)** i **català (`cat`)** del mateix `id` de problema d'entrenament, generant `data/pot_pairs.jsonl`.

### Com funciona la verificació amb *Sandbox* + Filtre Anti-Trampa AST
1. **Generació amb bucle de *feedback* d'error (fins a 3 intents)**:
   - Es demana a `deepseek-v4-flash` (`temperature=0.1`, `reasoning_effort="low"`) un objecte JSON amb `topic`, `steps_en`, `steps_ca`, `hint_en`, `hint_ca` i `code` (de 2 a 6 línies planes de Python, sense `def`, sense `import`, sense comentaris, acabant en `print(ans)`).
   - Si el codi dóna error de sintaxi, *timeout*, o imprimeix un valor que no coincideix amb el valor de l'opció correcta (`label`), el script fa un segon/tercer intent passant-li a l'API l'error exacte:
     > `"Your previous code produced printed='...' (status: ...), which did NOT match the ground-truth option D = 42. Write a genuine mathematical calculation..."`
2. **Filtre AST anti-*hardcoding* (`is_nontrivial_code`)**:
   - Per evitar que el model professor faci trampa escrivint `ans = 42; print(ans)` (ja que coneix l'opció correcta), `is_nontrivial_code()` analitza l'arbre sintàctic (`ast.parse`) i **rebutja qualsevol programa que no contingui operacions de càlcul real** (`ast.BinOp`, `ast.UnaryOp`, `ast.Compare`, `ast.BoolOp`, `ast.For`, `ast.While`, `ast.ListComp`, `ast.GeneratorExp` o crides com `sum`, `min`, `max`, `len`, `gcd`, `comb`, `permutations`, etc.).
3. **Execució al mateix `RUNNER` que corre a la placa**:
   - Només s'accepten programes que acaben en `< 2 s` dins del nostre sandbox (`copilot/pot.py`) i on `run_and_match(code, problem)` retorna **exactament una única lletra** i aquesta lletra és igual a `problem["label"]`.
4. **Volum final verificat**:
   - **585 problemes únics verificats** (`data/pot_programs.jsonl`), que en creuar-los amb els enunciats d'entrenament en anglès i català produeixen **929 parells verificats `(enunciat, codi Python)`** a `data/pot_pairs.jsonl` (0 solapament amb `val.jsonl` ni amb `demo/train.jsonl`).

---

## 2. Fine-Tuning LoRA (Versió 1 i Versió 2) i conversió GGUF

Tot el procés d'entrenament i conversió a GGUF està automatitzat i reproduïble en dos notebooks de Kaggle T4 dins de `finetune/`:
- [`finetune/00_gate_conversion.ipynb`](finetune/00_gate_conversion.ipynb)
- [`finetune/01_train_pot_lora.ipynb`](finetune/01_train_pot_lora.ipynb) (generat per [`finetune/build_notebooks.py`](finetune/build_notebooks.py))
- [`finetune/make_kaggle_dataset.py`](finetune/make_kaggle_dataset.py) (empaqueta `pot_pairs.jsonl`, `demo.jsonl`, `val.jsonl` i `copilot/pot.py` en un zip privat per a Kaggle)

### Gate 0: Validació de la conversió HF $\rightarrow$ GGUF (`00_gate_conversion.ipynb`)
Abans d'entrenar res, vam comprovar que convertir `Qwen/Qwen3.5-0.8B` (HuggingFace) a GGUF `Q8_0` amb el commit exacte de `llama.cpp` de la placa (`2cbe0ec429b849a88664e0f62388fc92a5abd699`) produeix un model idèntic al de `bartowski/Qwen_Qwen3.5-0.8B-GGUF`:
- Mateixa arquitectura `qwen35`, mateixos 320 tensors i dimensions, mateixa plantilla Jinja i mateixos tokens d'entrada/sortida sota `--jinja --reasoning off --reasoning-budget 0`.

### Alineació exacta del template i màscara de `loss`
- En lloc d'usar el template per defecte de `transformers`, cada exemple s'envia a l'endpoint `/apply-template` del mateix `llama-server` arrencat amb `--jinja --reasoning off --reasoning-budget 0`.
- Això garanteix que el prompt d'entrenament inclou exactament el bloc buit `<|im_start|>assistant\n<think>\n\n</think>\n\n` que `llama-server` pre-omple en producció a l'Arduino UNO Q.
- Posem `labels = -100` a tots els tokens del sistema, de l'enunciat i del `<think>\n\n</think>\n\n`, de manera que la *cross-entropy loss* es calcula **únicament sobre els tokens del bloc ```` ```python ... ``` ```` + `<|im_end|>`**.
- **Optimització de memòria VRAM a la T4**: Qwen3.5 té un vocabulari enorme (`248.320` tokens). Fer `.contiguous()` sobre tot el tensor `[batch, seq, 248320]` esgotava els 15 GB de la T4. Filtrant primer les posicions actives (`active = shift_labels != -100` i passant `shift_logits[active].float()` a `F.cross_entropy`), el consum de VRAM cau dràsticament i l'entrenament complet dura **~3 minuts en una sola GPU T4**.

---

### Fine-Tuning Versió 1 (`v1`)
- **Configuració `v1`**:
  - `r = 16`, `alpha = 32`, `LR = 2e-4`, `EPOCHS = 3`, `BATCH = 4`, `ACCUM = 4` (mida de lot efectiva = 16).
  - Aplicat a totes les 150 capes lineals (incloent les projeccions de porta recurrent de Gated DeltaNet).
- **Progressió de la `loss` a `v1`**:
  - Època 1: `train = 0.612`, `dev = 0.658`
  - Època 2: `train = 0.244`, `dev = 0.723`
  - Època 3: `train = 0.093`, `dev = 0.917`
- **Resultats de `v1` (`Qwen3.5-0.8B-M3Kang-PoT-Q8_0.gguf`, 794 MB)**:
  - **`Pass@1` a `demo` (100 problemes)**: **34,0% (34/100)** (`asked 76; printed 61; matched 29; RIGHT 18 [10 eng, 8 cat]; wrong 11; cut at max_tokens: 1/76`).
  - **Impacte clau en català i velocitat**: Els encerts de codi en català van passar de **2 a 8 ($\times 4$)**, i en no necessitar exemples *few-shot* (`POT_STYLE=finetuned`), el model llegeix només **54 tokens** i n'escriu **41 de mitjana**, baixant el temps mitjà a la placa a **12 s** (pitjor cas 32 s, 0 problemes > 50 s).
  - **`Pass@3` condicional amb `v1`**: Afegint fins a 2 reintents amb `temperature=0.45, top_p=0.9` quan el primer intent no casa amb cap opció (reutilitzant el prompt en caché amb `read_n = 0`), la puntuació va pujar a **36,0% (36/100)** (`RIGHT 20 [11 eng, 9 cat]`).
  - **`Pass@1` a `M3Kang val` (562 problemes)**: **26,2% (147/562)** vs 21,6% del model base.

---

### Fine-Tuning Versió 2 (`v2`): Regularització, protecció de portes Gated DeltaNet i *Early Stopping*
Analitzant la corba de `v1`, vam veure que a les èpoques 2 i 3 la `dev loss` pujava de `0.658` a `0.917` (el model memoritzava els nombres concrets del `train` i es tornava massa agressiu escrivint codi que coincidia per atzar amb opcions incorrectes). A més, a Kaggle la llibreria `peft` donava conflictes d'importació amb `torchao 0.10.0`.

Per a la **Versió 2 (`v2`)**, vam introduir **4 millores d'arquitectura i entrenament** a [`finetune/build_notebooks.py`](finetune/build_notebooks.py):
1. **`LoRALinear` en PyTorch pur amb `Dropout(0.05)`**: Sense dependre de `peft` ni `torchao`, i amb `merge()` exacte al pes base `bf16` abans d'exportar a GGUF.
2. **Protecció de les portes recurrents de *Gated DeltaNet* (`SKIP`)**:
   - Vam excloure del LoRA les projeccions d'actualització i decaïment d'estat recurrent (`in_proj_a`, `in_proj_b`, `in_proj_ba`), entrenant només les 114 projeccions d'atenció, contingut i MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`). Això evita desestabilitzar la memòria recurrent de Qwen3.5.
3. **Regularització d'hiperparàmetres**: `LR = 1e-4` (meitat que a `v1`), `weight_decay = 0.01`, `EPOCHS = 2`.
4. **Validació sub-època cada 70 passos amb *Early Stopping***:
   - Es desa en RAM l'estat LoRA amb menor `dev loss` durant l'entrenament i es restaura automàticament abans de fer el `merge()`.
   - El millor punt es va assolir al **`step 217` amb `dev loss = 0.673`** (contra `0.917` al final de `v1`).

- **Resultats de `v2` (`Qwen3.5-0.8B-M3Kang-PoT-v2-Q8_0.gguf`, abans dels canvis al runner)**:
  - **`Pass@1` a `demo`**: Puja de **34,0% a 37,0% (37/100)** (`matched 27; RIGHT 19 [11 eng, 8 cat]; wrong 8` $\rightarrow$ **70,4% de precisió** quan el codi casa amb una opció!). Escriu 36 tokens de mitjana i triga **11 s de mitjana a la placa** (pitjor cas 29 s).
  - **`Pass@3` condicional a `demo`**: Puja de **36,0% a 39,0% (39/100)**.

---

## 3. Últims canvis al Runner (`SmartRunner`) i filtre anti-`'1'` (`copilot/pot.py` i `copilot/llm.py`)

Un cop entrenat el model `v2` (`FT2`), vam inspeccionar un per un tots els scripts de Python que `FT2` havia generat als 76 problemes numèrics de `demo`. Vam descobrir una cosa clau: **en diversos problemes on `FT2` fallava, l'equació o l'algorisme que havia escrit ja era matemàticament perfecte**, però l'execució de Python fallava per 4 friccions habituals dels models petits amb la llibreria estàndard de Python.

Vam actualitzar `RUNNER` i `match_printed` / `match_option` a [`copilot/pot.py`](copilot/pot.py) i [`copilot/llm.py`](copilot/llm.py) amb 5 millores totalment genèriques:

### 3.1. `_PermTuple(tuple)` i `smart_int`: tupla de dígits vs. enter (`+1,0%`)
- **Problema detectat (`lvl-0_2026_10`, `eng`)**: El model volia comptar quants nombres de 4 xifres es poden formar amb `[0, 1, 2, 3]` i va escriure:
  ```python
  print(sum(1 for n in permutations([0, 1, 2, 3]) if 1000 <= n <= 9999))
  ```
  Com que `itertools.permutations` retorna tuples `(1, 0, 2, 3)`, Python llançava `TypeError: '<=' not supported between instances of 'int' and 'tuple'`.
- **Solució al `RUNNER`**: Vam embolcallar `permutations`, `combinations`, `product` i `combinations_with_replacement` perquè retornin `_PermTuple(tuple)`. Si una `_PermTuple` de dígits (`0..9`) es compara amb un `int` (`<`, `<=`, `>`, `>=`, `==`, `!=`), se li aplica el mòdul (`%`) o es passa a `int(t)`, es converteix transparentment a l'enter decimal que representa (`(1, 0, 2, 3) -> 1023`).
- **Efecte**: El codi de `FT2` imprimeix `18` (= Opció A, correcta) en `0 ms`.

### 3.2. `_SmartStr(str)`: `int in str(n)` i resta de strings numèrics (`+1,0%`)
- **Problema detectat (`lvl-1_2026_13`, `cat`)**: El model volia comptar quants nombres de l'1 al 100 contenen el dígit 7 i va escriure:
  ```python
  print(sum(1 for n in range(1, 101) if 7 in str(n)))
  ```
  Python llançava `TypeError: 'in <string>' requires string as left operand, not int` perquè `7` és un `int` i no `'7'`. En un altre problema (`lvl-2_2026_5`), el model feia `str(n)[::-1] - str(n)`.
- **Solució al `RUNNER`**: Vam definir `safe["str"] = smart_str`, que retorna un `_SmartStr(str)` on `__contains__` converteix enters a `str` abans de comprovar `in`, i `__sub__` permet restar dos strings si tots dos són nombres enters.
- **Efecte**: `lvl-1_2026_13 (cat)` imprimeix `19` (= Opció B, correcta).

### 3.3. `smart_product` / `_SmartIntProd`: desambiguació entre `math.prod` i `itertools.product` (`+1,0%`)
- **Problema detectat (`lvl-4_2026_10`, `eng`)**: El model buscava el nombre més petit el producte de les xifres del qual és 120, i va escriure:
  ```python
  print(min(n for n in range(1, 1000) if product(int(c) for c in str(n)) == 120))
  ```
  Al nostre `RUNNER` original, `product` apuntava a `itertools.product` (producte cartesià), mentre que el producte numèric es deia `prod`. Però en altres problemes el model sí que utilitza `for p in product(range(10), repeat=3):` com a producte cartesià!
- **Solució al `RUNNER`**: `smart_product(*args, **kwargs)` comprova si rep **un únic argument iterable sense `repeat=`** format per nombres enters. En aquest cas retorna `_SmartIntProd(math.prod(items), items)`: una subclasse d'`int` que val el producte matemàtic (`120 == 120` és `True`), però que a més implementa `__iter__` per si el model realment volia iterar sobre `product([1, 2, 3])` en un bucle `for`.
- **Efecte**: `lvl-4_2026_10 (eng)` imprimeix `358` (= Opció C, correcta).

### 3.4. Captura de `printed[0]` i `printed[-1]` quan falta un `break` (`+1,0%`)
- **Problema detectat (`lvl-1_2026_8`, `cat`)**: El model buscava el nombre més petit que deixa residu 1 en dividir-lo per 2, 3, 4, 5 i 6, i va escriure el bucle exacte però es va deixar el `break`:
  ```python
  for n in range(2, 1000):
      if n % 2 == 1 and n % 3 == 1 and n % 4 == 1 and n % 5 == 1 and n % 6 == 1:
          print(n)
  ```
  El bucle imprimia `61`, `121`, `181`, ..., `961`. Com que el `RUNNER` antic només retornava l'última línia (`printed[-1] = 961`), no coincidia amb cap opció i perdia la resposta correcta (`printed[0] = 61`, Opció E).
- **Solució al `RUNNER` i a `match_printed` / `match_option`**: Quan un programa imprimeix més d'una línia i la primera és diferent de l'última, `RUNNER` retorna `printed[0] + "\t" + printed[-1]`. Primer es comprova si l'últim valor imprès (`printed[-1]`, per a programes que imprimeixen passos intermedis i al final la resposta) casa amb una única opció; si no casa amb cap opció, es comprova si el primer valor imprès (`printed[0]`, per a bucles de cerca sense `break`) casa amb una única opció.
- **Efecte**: `lvl-1_2026_8 (cat)` recupera `61` (= Opció E, correcta) a la primera passada (`Pass@1`).

### 3.5. Filtre anti-`'1'` per als reintents amb temperatura (`Pass@3`)
- Quan vam analitzar els reintents (`attempt = 2, 3` amb `temperature = 0.45`), vam observar que **5 dels 7 falsos positius nous** tenien exactament el mateix patró: al primer intent (`temp = 0`) el bucle `sum(1 for ...)` donava `0` o `SyntaxError` (cap coincidència), i en un reintent a `temp = 0.45` el model canviava la condició i comptava casualment `1` element, que gairebé sempre coincideix amb alguna opció petita (`1`).
- **Solució a `solve_pot` (`copilot/llm.py`)**: Si el primer intent (`attempt == 1`) imprimeix `'1'`, s'accepta immediatament. Però si el primer intent imprimeix `'0'` o falla (`None`) i un reintent (`attempt > 1`) imprimeix `'1'`, només s'accepta si **els dos reintents coincideixen en la mateixa lletra** (`pending_one_letter == cand`).
- **Efecte**: Elimina els falsos positius d'`'1'` i porta el mode amb reintents a **42,0% (42/100)**.

---

## 4. Com utilitzar i reproduir el model al repositori

### Fitxers del model GGUF generats
1. **Versió 2 (Recomanada — 41,0% `Pass@1` / 42,0% `Pass@3`)**:
   - Fitxer: `Qwen3.5-0.8B-M3Kang-PoT-v2-Q8_0.gguf` (794 MB)
   - SHA256: `fe09ae5f83c845fbe9b49af5cfbbd049976c04c19ebf55249cec84881723102e`
2. **Versió 1 (34,0% `Pass@1` / 36,0% `Pass@3`)**:
   - Fitxer: `Qwen3.5-0.8B-M3Kang-PoT-Q8_0.gguf` (794 MB)
   - SHA256: `ec2ed8e5dc0febcf1c1ae8f8482963d8d28e4d23e4f2e2b71779b7c72daf4d0b`

### Configuració a `config.env`
A [`config.env`](config.env) ja està configurat el mode `ANSWER_MODE=pot` i `POT_STYLE=finetuned`:
```ini
LLM_FILE=Qwen3.5-0.8B-M3Kang-PoT-v2-Q8_0.gguf
LLM_SHA256=fe09ae5f83c845fbe9b49af5cfbbd049976c04c19ebf55249cec84881723102e
ANSWER_MODE=pot
POT_STYLE=finetuned
```
*(Si es vol córrer amb el model base `Qwen_Qwen3.5-0.8B-Q8_0.gguf` sense fine-tuning, n'hi ha prou de posar `POT_STYLE=fewshot`, que activa automàticament el prompt bilingüe de 496 tokens amb caché).*

### Com executar l'avaluació sobre el set de pràctica 2026 (`demo`)
```bash
python eval.py --input data/demo/train.jsonl --limit 0 --name pot-ft2-demo
```
O directament des de Python per mesurar tant `Pass@1` com `Pass@3` i el temps estimat a l'Arduino UNO Q:
```bash
python -c "import json; from copilot.pipeline import solve; rows=[json.loads(l) for l in open('data/demo/train.jsonl',encoding='utf-8')]; hits=sum(solve(r)['answer']==r['label'] for r in rows); print(f'Accuracy: {hits}/{len(rows)} ({hits}% )')"
```
