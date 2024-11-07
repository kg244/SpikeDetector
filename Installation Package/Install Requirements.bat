@echo off
echo Installing required Python packages...

REM Check if requirements.txt exists
IF NOT EXIST requirements.txt (
    echo requirements.txt file not found!
    exit /b
)

REM Install packages from requirements.txt
pip install -r requirements.txt

REM Install xlwings add-in
echo Installing xlwings add-in...
xlwings addin install

echo Installation complete!
pause