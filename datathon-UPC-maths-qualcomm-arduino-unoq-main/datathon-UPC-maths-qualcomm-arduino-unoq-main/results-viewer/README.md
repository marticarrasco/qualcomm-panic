# Visor de resultats

App web molt senzilla per mirar els CSV que genera `eval.py` dins de `results/`.
No porta cap dependència: HTML + JS purs, i el servidor és només llibreria estàndard.

## Amb servidor

Des de l'arrel del projecte:

```bash
python results-viewer/server.py
```

I obre `http://127.0.0.1:8001`. La pàgina mostra la llista de tots els CSV de
`results/` (els més nous primer); clica'n un per carregar-lo. Pots canviar la
carpeta amb `--results` i el port amb `--port`.

## Sense servidor

També pots obrir `results-viewer/index.html` directament al navegador i arrastrar-hi
qualsevol CSV (per exemple `results/ablations.csv`).

## Què hi trobaràs

Per a un CSV d'avaluació (`eval-*.csv`):

- Targetes amb encerts, precisió sobre respostes vàlides, tokens i temps mitjans.
- Precisió desglossada per nivell, llengua, opcions OCR i estat.
- Diagnòstic de sortida: quantes respostes toquen el límit de tokens i quina
  precisió tenen respecte les que acaben dins el límit, i tokens mitjans
  d'encertades vs fallades.
- Matriu de confusió de la lletra triada vs la correcta.
- Les 5 preguntes més lentes (temps real i estimació Arduino).
- Comparació de precisió amb i sense exemples RAG, si l'execució va usar RAG.
- Filtres (llengua, nivell, estat, cerca, només no encertades) i taula ordenable;
  clica una fila per veure la sortida íntegra del model.

Els CSV d'avaluació amb esquemes antics (`id,language,level,has_options,label,answer,
correct,seconds,...`) s'adapten automàticament al format actual. Els que no tenen
resposta correcta (`local-reasoning-*.csv`) es mostren com a taula genèrica.
Per a qualsevol altre CSV també es mostra una taula genèrica amb mitjanes de les
columnes numèriques.
