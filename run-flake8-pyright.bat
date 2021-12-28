REM nodemon -w radiobee -w tests -x "pyright radiobee && flake8.bat radiobee"
nodemon -w radiobee -w tests -x "run-p pyright flake8"
