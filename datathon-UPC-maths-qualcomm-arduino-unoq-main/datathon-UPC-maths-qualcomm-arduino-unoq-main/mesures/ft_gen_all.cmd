@echo off
rem One server at a time (several at once starve each other). Resumes where it stopped.
set PY=%~dp0..\datathon-UPC-maths-qualcomm-arduino-unoq\.venv\Scripts\python.exe
set D=%~dp0..
set G="%PY%" -X utf8 "%~dp0ft_gen.py" --port 8091 --threads 8
set FT1="%D%\Qwen3.5-0.8B-M3Kang-PoT-Q8_0.gguf"
set FT2="%D%\Qwen3.5-0.8B-M3Kang-PoT-v2-Q8_0.gguf"
set FT3="%D%\Qwen3.5-0.8B-M3Kang-PoT-v3-Q8_0.gguf"
%G% --model %FT2% --style finetuned --tag ft2 || exit /b 1
%G% --model %FT3% --style finetuned --tag ft3 || exit /b 1
%G% --model %FT1% --style finetuned --tag ft1 || exit /b 1
echo GREEDY DONE
%G% --model %FT2% --style finetuned --tag ft2 --samples 2 --sets demo,val || exit /b 1
echo SAMPLES DONE
%G% --model %FT1% --style finetuned --tag ft1 --gate figure --sets demo,val || exit /b 1
%G% --model %FT1% --style finetuned --tag ft1 --gate units --sets demo,val || exit /b 1
echo GATES DONE
%G% --model %FT1% --style finetuned --tag ft1 --samples 2 --sets demo,val || exit /b 1
echo ALL DONE
