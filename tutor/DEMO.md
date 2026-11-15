# Pista · Prova amb el banc complet de M3Kang

Pista treballa amb 1.747 problemes reals en català i anglès. Conserva els enunciats originals, les figures i les opcions. Les 1.466 fitxes de train fan servir les deu categories del projecte.

Des de la carpeta que conté `tutor/`, prepara el banc i arrenca el servidor amb el servei local de model en marxa:

```bash
python3 tutor/build_bank.py --lexical
python3 tutor/server.py --host 0.0.0.0
```

Al port 8000 hi trobaràs el tutor. El port 8081 serveix el model per defecte. Pots indicar un altre endpoint amb `--llm` i activar `--vision` amb el servei multimodal. Per fer servir el servei multimodal del projecte, passa `--vision --llm http://127.0.0.1:8083/v1/chat/completions`. Si vols fer la prova amb un historial nou, passa `--students data/pista-prova.db`.

1. Entra amb un perfil, selecciona el català, un tema i un nivell. Demana una pista. El model genera un primer pas a partir de la fitxa actual.
2. Prova una resposta incorrecta. La clau de M3Kang la corregeix i el model rep el context de l'opció que has triat. Escriu al xat què has intentat o què no entens.
3. Resol el problema o demana veure la resposta. Si apareix un problema de seguiment, prova'l sense pistes.
4. Obre la vista del professor per veure el progrés per tema i els problemes de seguiment resolts a la primera sense ajuda. Canvia a l'anglès per comprovar que es conserva l'intent i es carrega la fitxa en l'altre idioma.

A l'Arduino UNO Q, l'app `applab/math-matrix` pot mostrar els esdeveniments de pràctica a la matriu LED. Els enunciats, el perfil i les peticions del model es processen localment.
