@echo off
rem Pure model evaluation on the English text-only problems.
rem Extra arguments are passed through, e.g.  run-model-eval.cmd --limit 10
set ROOT=%~dp0datathon-UPC-maths-qualcomm-arduino-unoq-main\datathon-UPC-maths-qualcomm-arduino-unoq-main
cd /d "%ROOT%"
python eval_model_only.py --prompt strict %*
