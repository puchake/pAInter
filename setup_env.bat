@echo off

if "%~1" == "" (
    echo You have to specify type of pytorch installation. CUDA will install pytorch with CUDA support and CPU will install pytorch without it.
    exit /B
)

python3 -m venv venv
call "%~dp0venv\Scripts\activate.bat"

if "%~1" == "CPU" (
    pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp35-cp35m-win_amd64.whl
    pip install torchvision
) else if "%~1" == "CUDA" (
    pip install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp35-cp35m-win_amd64.whl
    pip install torchvision
) else (
    echo Unsupported value of first argument = $1. Supported values are CPU and CUDA.
    exit /B
)

pip install -r requirements.txt
deactivate