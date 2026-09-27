@echo off
rem Double-click this file to train the model (writes model.npz and ..\web_version\model.js)
cd /d "%~dp0"
"%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" train.py
pause
