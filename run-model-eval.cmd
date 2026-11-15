@echo off
setlocal
rem Pure model evaluation on the English text-only problems.
rem Extra arguments are passed through, e.g.  run-model-eval.cmd --limit 10
pushd "%~dp0" || exit /b 1
python -X utf8 eval_model_only.py --prompt strict %*
set "status=%errorlevel%"
popd
exit /b %status%
