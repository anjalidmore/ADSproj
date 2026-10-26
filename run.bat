@echo off
rem Windows launcher: runs run.sh with Git Bash. Arguments are passed through.
setlocal
set "BASH="
for %%P in ("%ProgramFiles%\Git\bin\bash.exe" "%ProgramFiles(x86)%\Git\bin\bash.exe" "%LocalAppData%\Programs\Git\bin\bash.exe") do (
  if exist %%P if not defined BASH set "BASH=%%~P"
)
if not defined BASH (
  where git >nul 2>&1 && for /f "delims=" %%G in ('where git') do (
    if not defined BASH if exist "%%~dpG..\bin\bash.exe" set "BASH=%%~dpG..\bin\bash.exe"
  )
)
if not defined BASH (
  echo Git Bash not found. Install Git for Windows: https://git-scm.com/download/win
  pause
  exit /b 1
)
"%BASH%" "%~dp0run.sh" %*
pause
