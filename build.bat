@echo off
REM ============================================================
REM 女装电商图片智能裁切 - 一键打包成 Windows EXE
REM
REM 先决条件：
REM   1. 已安装 Python 3.10+ 并加入 PATH
REM   2. pip install -r requirements.txt
REM   3. pip install pyinstaller
REM ============================================================

setlocal enabledelayedexpansion

set APP_NAME=CropTool
set ENTRY=main.py

echo.
echo [1/3] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist %APP_NAME%.spec del %APP_NAME%.spec

echo.
echo [2/3] Locating MediaPipe resources...
for /f "delims=" %%i in ('python -c "import os, mediapipe; print(os.path.dirname(mediapipe.__file__))"') do set MP_DIR=%%i
if "%MP_DIR%"=="" (
    echo [ERROR] MediaPipe is not installed. Run: pip install -r requirements.txt
    exit /b 1
)
echo     MediaPipe: %MP_DIR%

echo.
echo [3/3] Running PyInstaller...
pyinstaller ^
    --name %APP_NAME% ^
    --onefile ^
    --noconsole ^
    --clean ^
    --add-data "%MP_DIR%\modules;mediapipe\modules" ^
    --hidden-import mediapipe ^
    --hidden-import cv2 ^
    --collect-all mediapipe ^
    %ENTRY%

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Build succeeded: dist\%APP_NAME%.exe
echo ============================================================
endlocal
