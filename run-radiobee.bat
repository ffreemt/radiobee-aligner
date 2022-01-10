REM nodemon -V -w radiobee -x "sleep 3 && python -m radiobee"
REM nodemon -V -w radiobee -x python -m radiobee
REM nodemon -V -w radiobee -x py -3.8 -m radiobee
REM nodemon -V -w radiobee -x "run-p pyright flake8 && py -3.8 -m radiobee"
REM nodemon -V -w radiobee -x "run-p pyright-radiobee && py -3.8 -m radiobee"
REM nodemon -V -w radiobee -x "pyright radiobee && py -3.8 -m radiobee"

REM E501 line-too-long, F401, unused import, W293 blank line contains whitespace
nodemon -V -w radiobee -x "flake8.exe --max-complexity=55 --ignore=E501,F401,W293 radiobee && py -3.8 -m radiobee"
