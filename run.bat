@echo off
title CineMatch AI Recommendation Engine Launcher
color 0A

echo ========================================================
echo   CineMatch AI Recommendation Engine - Setup ^& Launch
echo ========================================================
echo.

:: Step 1: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH!
    echo Please install Python and try again.
    pause
    exit /b
)

:: Step 2: Initialize Virtual Environment
if not exist "venv" (
    echo [SETUP] Creating a clean Python Virtual Environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b
    )
    echo [SUCCESS] Virtual environment created successfully.
    echo.
)

:: Step 3: Activate venv and install dependencies
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate

echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip >nul

echo [SETUP] Installing project dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed!
    pause
    exit /b
)
echo [SUCCESS] Dependencies satisfied.
echo.

:: Step 4: Run Main Orchestrator
echo ========================================================
echo   Starting CineMatch AI Recommender Pipeline...
echo ========================================================
echo.
python main.py

pause
