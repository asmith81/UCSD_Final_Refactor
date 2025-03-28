@echo off
REM Setup script for configuring environment on Windows
REM This script validates dependencies, creates necessary directories,
REM and sets up environment variables needed for the project.

setlocal

REM Detect environment
IF NOT "%RUNPOD_POD_ID%"=="" (
    SET ENVIRONMENT=runpod
    echo 📋 Detected RunPod environment
) ELSE (
    SET ENVIRONMENT=local
    echo 📋 Detected local Windows environment
)

REM Find project root directory
SET SCRIPT_DIR=%~dp0
SET PROJECT_ROOT=%SCRIPT_DIR%..
echo 📂 Project root: %PROJECT_ROOT%

REM Export project root as environment variable
setx PROJECT_ROOT "%PROJECT_ROOT%"

REM Create logs directory
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"

REM Check Python version
python --version > temp.txt
set /p PYTHON_VERSION=<temp.txt
del temp.txt
echo 🐍 %PYTHON_VERSION%

REM Check for GPU
python -c "import torch; print(f'PyTorch detected, CUDA available: {torch.cuda.is_available()}')"
IF %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Warning: PyTorch not installed or error checking CUDA
)

REM Install or verify required packages
echo 📦 Checking required packages...

REM Create a temporary requirements file
echo transformers>=4.34.0 > temp_requirements.txt
echo torch>=2.0.0 >> temp_requirements.txt
echo pillow>=10.0.0 >> temp_requirements.txt
echo matplotlib>=3.7.2 >> temp_requirements.txt
echo pandas>=2.0.3 >> temp_requirements.txt
echo pyyaml>=6.0 >> temp_requirements.txt

REM Install packages
pip install -r temp_requirements.txt
del temp_requirements.txt

REM Verify installation
echo ✅ Validating installations...
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import yaml; print(f'PyYAML {yaml.__version__}')"

REM Create necessary local directories
if not exist "%PROJECT_ROOT%\models\cache" mkdir "%PROJECT_ROOT%\models\cache"
echo 📂 Created model cache directory: %PROJECT_ROOT%\models\cache

if not exist "%PROJECT_ROOT%\data\images" mkdir "%PROJECT_ROOT%\data\images"
if not exist "%PROJECT_ROOT%\results\raw" mkdir "%PROJECT_ROOT%\results\raw"
if not exist "%PROJECT_ROOT%\results\processed" mkdir "%PROJECT_ROOT%\results\processed"

REM Set up Python path to include project
set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%
echo 🔄 Added %PROJECT_ROOT% to PYTHONPATH

REM Run a Python validation script if it exists
SET VALIDATION_SCRIPT=%PROJECT_ROOT%\src\config\validate_env.py
IF EXIST %VALIDATION_SCRIPT% (
    echo 🔍 Running environment validation...
    python %VALIDATION_SCRIPT%
)

echo ✨ Environment setup complete for %ENVIRONMENT%!
echo 📋 To use this configuration in your notebooks, import the environment modules.

endlocal