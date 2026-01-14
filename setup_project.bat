@echo off
REM ITI123 Project Setup Script (Windows)
REM AI-Based Badminton Stroke Technique Assessment
REM This script creates the complete folder structure for the project

echo ==========================================
echo ITI123 Project Folder Structure Setup
echo ==========================================
echo.

echo Creating project folder structure...
echo.

REM Create main data folders
echo Creating data folders...
mkdir data\raw_videos 2>nul
mkdir data\annotations 2>nul
mkdir data\processed\clips 2>nul
mkdir data\processed\poses 2>nul
mkdir data\processed\features 2>nul

REM Create notebooks folder
echo Creating notebooks folder...
mkdir notebooks 2>nul

REM Create source code folders
echo Creating source code folders...
mkdir src\data_processing 2>nul
mkdir src\models 2>nul
mkdir src\evaluation 2>nul
mkdir src\deployment 2>nul

REM Create experiments folder (for MLflow)
echo Creating experiments folder...
mkdir experiments 2>nul

REM Create models folder (for saved models)
echo Creating models folder...
mkdir models 2>nul

REM Create outputs folder
echo Creating outputs folder...
mkdir outputs\plots 2>nul
mkdir outputs\reports 2>nul
mkdir outputs\visualizations 2>nul

REM Create docs folder
echo Creating documentation folder...
mkdir docs 2>nul

REM Create placeholder __init__.py files for Python packages
echo Creating Python package files...
type nul > src\__init__.py
type nul > src\data_processing\__init__.py
type nul > src\models\__init__.py
type nul > src\evaluation\__init__.py
type nul > src\deployment\__init__.py

REM Create .gitkeep files to preserve empty folders in git
echo Creating .gitkeep files for git...
type nul > data\raw_videos\.gitkeep
type nul > data\annotations\.gitkeep
type nul > data\processed\clips\.gitkeep
type nul > data\processed\poses\.gitkeep
type nul > data\processed\features\.gitkeep
type nul > experiments\.gitkeep
type nul > models\.gitkeep
type nul > outputs\plots\.gitkeep
type nul > outputs\reports\.gitkeep
type nul > outputs\visualizations\.gitkeep

echo.
echo ==========================================
echo Folder structure created successfully!
echo ==========================================
echo.
echo Next steps:
echo 1. Create virtual environment: python -m venv venv
echo 2. Activate it: venv\Scripts\activate
echo 3. Install requirements: pip install -r requirements.txt
echo.
echo Note: Raw video files should be placed in: data\raw_videos\
echo       ShuttleSet annotations should be placed in: data\annotations\
echo.
pause
