# Resum PoT fine-tuning v1 → v4 (per preparar la presentació final)

Aquest fitxer és el punt d'entrada per a qui hagi de fer la presentació. Recull què vam fer amb el fine-tuning
"Program of Thought" (PoT) de `Qwen3.5-0.8B` i on són els fitxers. Tot el codi és a
`datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/` (a partir d'ara `APP/`).
La documentació detallada de v1/v2 és a `APP/POT_FINETUNING_I_RUNNER.md`.

## La idea

El model petit (0.8B, Q8_0, corre a la placa Arduino UNO Q amb llama.cpp) no resol bé el problema de Cangur
raonant en text. En comptes d'això, li fem escriure **un programa curt de Python** que calcula la resposta; el
programa s'executa en un sandbox (`APP/copilot/pot.py`, el "SmartRunner") i el número que imprimeix s'aparella amb
una de les opcions A–E. Si no coincideix amb cap opció, es fa servir una resposta de reserva.

## Les versions

| Versió | GGUF (no és al repo, 833 MB cadascun) | Què canvia |
|---|---|---|
| stock | `Qwen_Qwen3.5-0.8B-Q8_0.gguf` | model base amb prompt PoT |
| v1 (ft1) | `Qwen3.5-0.8B-M3Kang-PoT-Q8_0.gguf` | LoRA r=16, alpha=32, 3 epochs, lr 2e-4. Parelles enunciat→programa destil·lades amb `deepseek-v4-flash` (API Helmcode) i **verificades executant-les** contra la resposta correcta, en anglès i català |
| v2 (ft2) | `...-PoT-v2-Q8_0.gguf` | 2 epochs, validació cada 70 passos i restauració del millor checkpoint (dev loss 0.673) |
| v3 (ft3) | `...-PoT-v3-Q8_0.gguf` | dataset v3 (`mesures/build_pot_v3.py`): 507 problemes / 1.011 parelles bilingües, programes 1-línia convertits en scripts multi-línia amb variables amb nom o bucles de força bruta, plantilles canòniques per famílies (rellotges digitals, cues/curses, simulacions, sistemes diofàntics), 0 % de solapament amb demo i val |
| v4 (ft4, epoch 1) | `...-PoT-v4-e1-Q8_0.gguf` | dades v3 + problemes nous de nivell baix (0–3) generats per un model professor (`mesures/gen_synth_lowlevel.py`, 440 problemes mantinguts, filtrats per: el programa s'executa i dona exactament una opció, una segona crida independent tria la mateixa opció, el programa calcula de debò, i l'enunciat no s'assembla a demo/val). Entrenament `APP/finetune/02_train_ft4_epochs.ipynb`: 3 epochs, un GGUF per epoch, i es tria l'epoch per respostes executades a M3Kang val (puntuació `right - wrong/4`). Es va quedar l'epoch 1 |

## Resultats mesurats (portàtil, llama.cpp, greedy, un programa per pregunta, SmartRunner)

Font: `APP/mesures/ft_score_output.txt` (generat per `APP/mesures/ft_score.py`). "Score amb D" = encerts quan
s'utilitza D com a resposta de reserva si el programa no coincideix amb cap opció.

| Model | demo (100 files): encerts PoT / errors | demo: score amb D | val (562 files): encerts / errors | val: score amb D | precisió (demo) |
|---|---|---|---|---|---|
| stock | – | – | 2 / 16 | 23.0 % | – |
| v1 | 15 / 8 | 32 % | 21 / 16 | 26.0 % | 65 % |
| v2 | 21 / 12 | **39 %** | 8 / 20 | 23.8 % | 64 % |
| v3 | 14 / 10 | 32 % | 17 / 27 | 24.6 % | 58 % |
| **v4** | **24 / 8** | **42 %** | 16 / 6 | 25.3 % | **75 %** |

Puntuació "esperada" del notebook de la v4 (`encerts - errors/4`, una resposta errònia costa més que no respondre):

| Model | demo | val |
|---|---|---|
| v1 | 13.0 | **17.0** |
| v2 | 18.0 | 3.0 |
| v3 | 11.5 | 10.3 |
| **v4** | **22.0** | 14.5 |

Sobre la v4 (fitxers `APP/mesures/eval_{demo,val}_epoch1.jsonl`, generats a Kaggle i re-executats aquí amb el
mateix sandbox com a `ft-ft4-*.jsonl`; Kaggle donava 25 encerts a demo, aquí 24):
- **És la millor a demo**: 42 % amb D de reserva (v2: 39 %), 17 problemes de 50 resolts pel programa. Guanya la v3
  de forma significativa (14 files contra 4, p = 0.03) i la v1 gairebé (13 contra 4, p = 0.05).
- **Gairebé no s'equivoca quan respon**: a val, 6 errors contra 16 de la v1 i 27 de la v3, precisió 73 % (v1: 57 %).
  A demo, precisió 75 %. Útil per a un copilot: quan diu una lletra, és fiable.
- Puja a nivells alts a demo (nivell 7: 5 de 8 bé, 0 malament).
- A val fa una mica menys encerts que la v1 (16 contra 21, diferència no significativa, p = 0.40).
- Matís honest: l'epoch 1 es va triar mirant val, així que el val de la v4 és lleugerament optimista; demo no es va
  fer servir per triar res.
- Temps estimat a la placa: 14 s de mitjana a demo, pitjor cas 29 s.

Lectures clau per a la presentació:
- El fine-tuning fa que el model **intenti** molts més problemes (val: de 39 a 163 preguntes amb programa) i que,
  quan el programa coincideix amb una opció, encerti amb molta més freqüència (precisió ~60 % a demo).
- Abans de la v4, v2 era la millor a demo però la pitjor a val; v1 era la més estable (val: v1 guanya v2 amb p = 0.01, McNemar).
  Demo només té 50 problemes: les diferències entre versions són petites i sorolloses.
- La majoria de fallades són "el programa imprimeix un número que no és cap opció" (no errors de Python).
- Combinar models (dos d'acord) dona encerts molt fiables: demo v1+v2+v3, quan dos coincideixen: 18 bé / 3 malament.
- Temps estimat a la placa: ~11–16 s de mitjana per pregunta, pitjor cas < 40 s.
- Els números antics del README (41 % Pass@1 / 42 % Pass@3 a demo per a v2) es van mesurar abans d'alguns canvis al
  runner; els d'aquesta taula són els actuals.

## On és cada cosa

- `APP/copilot/pot.py`, `APP/copilot/llm.py`, `APP/copilot/pipeline.py`: el pipeline PoT + SmartRunner integrat al copilot.
- `APP/finetune/`: notebooks de Kaggle (00 conversió, 01 LoRA v1–v3, 02 v4 per epochs) i scripts per construir les dades.
- `APP/mesures/`: tots els scripts d'experiments i els resultats en brut (`*.jsonl`). `ft_gen.py` genera programes
  amb cada GGUF, `ft_score.py` els puntua. Nota: els scripts busquen el codi a `../datathon-UPC-maths-qualcomm-arduino-unoq`
  (la nostra còpia local), cal ajustar `ROOT` per executar-los des d'aquí.
- `APP/chat/`, `APP/laptop/fix_options.py`, `APP/analyze_logprobs.py`, `APP/dump_logprobs.py`: eines auxiliars (xat local, correcció d'opcions, anàlisi de logprobs).
- No són al repo: els GGUF (massa grans) i les dades d'entrenament (`data/pot_pairs.jsonl`, `kaggle_pot_dataset*.zip`), que deriven de M3Kang.
