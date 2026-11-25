@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo USED CAR PRICE PREDICTION - WINDOWS LAUNCHER
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Python 3.11 was not found.
        echo Install Python 3.11 from https://www.python.org/downloads/ and enable "Add Python to PATH".
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist "model\used_car_price_model.cbm" (
    echo No trained model found. Training from the supplied CSV...
    python train_model.py
    if errorlevel 1 goto :error
)

echo.
echo Starting the web application. Keep this window open.
python -m streamlit run app.py
if errorlevel 1 goto :error
goto :end

:error
echo.
echo The application could not be started. Read the error above.
pause
exit /b 1

:end
endlocal
